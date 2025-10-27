# app/services/vector_store.py
import os
import time
import logging
import json
from typing import Any, Dict, List, Optional, Iterable

from ..config import AppConfig  # kept for parity (unused but preserved)
from .embeddings import embed_text, OPENAI_EMBED_MODEL

try:
    from pinecone import Pinecone
except Exception as e:
    raise RuntimeError("Missing dependency: pip install pinecone-client") from e

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------
# Area helpers
# ---------------------------

TOKYO_SYNONYMS = {"tokyo", "東京", "東京都", "tokyo metropolis"}
WARDS = {
    "渋谷区","新宿区","目黒区","世田谷区","中野区","杉並区","練馬区","港区","千代田区","中央区",
    "台東区","墨田区","江東区","品川区","大田区","豊島区","北区","荒川区","板橋区","足立区",
    "葛飾区","江戸川区","文京区"
}
CITIES = {
    "三鷹市","武蔵野市","調布市","府中市","小金井市","西東京市","小平市","東久留米市","国分寺市","国立市"
}

def _normalize_area(a: str) -> str:
    if not a:
        return ""
    s = str(a).strip()
    if s.lower() in TOKYO_SYNONYMS or s in {"東京"}:
        return "東京都"
    for w in WARDS:
        if w in s:
            return w
    for c in CITIES:
        if c in s:
            return c
    return s

def _areas_from_where(where: Optional[Dict[str, Any]]) -> List[str]:
    if not where:
        return []
    out: List[str] = []
    for k in ("areas", "area", "prefecture", "city", "ward", "neighborhood", "area_like"):
        v = where.get(k)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out.extend([_normalize_area(x) for x in v if x])
        else:
            out.append(_normalize_area(v))
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def _matches_area(meta: Dict[str, Any], desired: List[str]) -> bool:
    if not desired:
        return True
    hay = []
    for k in ("prefecture", "city", "ward", "area", "neighborhood"):
        v = meta.get(k)
        if isinstance(v, str):
            hay.append(v)
    h = " ".join(hay)
    if not h:
        return False
    for d in desired:
        if d and (d == h or d in h):
            return True
    return False

def _merge_unique(lists: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for L in lists:
        for r in L:
            rid = r.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            out.append(r)
    return out

# ---------------------------
# Filters
# ---------------------------

def _num(v):
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return None

def _meta_filter(where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not where:
        return {}
    f: Dict[str, Any] = {}
    if where.get("max_budget_yen") is not None:
        v = _num(where["max_budget_yen"])
        if v is not None:
            f["price_yen"] = {"$lte": v}
    elif where.get("max_budget_usd") is not None:
        v = _num(where["max_budget_usd"])
        if v is not None:
            f["price_usd"] = {"$lte": v}
    if "pet_dog" in where and where.get("pet_dog") is not None:
        f["pet_dog"] = {"$eq": bool(where["pet_dog"])}
    return f

# ---------------------------
# Namespace selection
# ---------------------------

def _namespaces(default_ns: str, where: Optional[Dict[str, Any]]) -> List[str]:
    explicit = (where or {}).get("namespace")
    if explicit:
        return [explicit]
    areas = _areas_from_where(where)
    if not areas:
        return [default_ns or "listings_openai"]
    return [default_ns or "listings_openai"]

# ---------------------------
# Pinecone response utils
# ---------------------------

def _resp_to_dict(resp: Any) -> Dict[str, Any]:
    try:
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
        if hasattr(resp, "vectors"):
            try:
                return {"vectors": dict(resp.vectors)}
            except Exception:
                return {"vectors": {k: v for k, v in resp.vectors.items()}}
        return dict(resp)
    except Exception:
        out = {}
        if hasattr(resp, "vectors"):
            try:
                out["vectors"] = dict(resp.vectors)
            except Exception:
                out["vectors"] = {}
        return out

# ---------------------------
# Metadata shrinking (to avoid 40KB limit)
# ---------------------------

META_HARD_LIMIT = 40960
META_SOFT_BUDGET = 38000

def _estimate_meta_size(meta: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(meta, ensure_ascii=False))
    except Exception:
        return 0

def _shrink_meta(meta: Dict[str, Any], text: str, soft_budget: int = META_SOFT_BUDGET) -> Dict[str, Any]:
    m = dict(meta) if meta else {}

    def size_ok(limit=soft_budget) -> bool:
        return _estimate_meta_size(m) <= limit

    if size_ok():
        return m

    # drop heavy/rare fields first
    for k in ["source_html", "raw_html", "raw_text", "price_text"]:
        if k in m and not size_ok():
            m.pop(k, None)

    # cap images list + total char budget
    if "images" in m and isinstance(m["images"], list):
        if len(m["images"]) > 8 and not size_ok():
            m["images"] = m["images"][:8]
        if not size_ok():
            total = 0
            capped = []
            for u in m["images"]:
                s = str(u)
                if total + len(s) > 4000:
                    break
                total += len(s)
                capped.append(s)
            m["images"] = capped

    # replace document with snippet
    if not size_ok():
        if "document" in m:
            m.pop("document", None)
        snippet = (text or "")[:1000]
        if snippet:
            m["snippet"] = snippet

    # truncate long strings
    if not size_ok():
        for k, v in list(m.items()):
            if isinstance(v, str) and len(v) > 1024 and not size_ok():
                m[k] = v[:1024]

    # drop lower-importance fields
    for k in ["url", "source", "description", "neighborhood"]:
        if not size_ok():
            m.pop(k, None)

    # final guard against hard limit
    if not size_ok(META_HARD_LIMIT):
        for k, v in list(m.items()):
            if isinstance(v, (list, dict)):
                m.pop(k, None)
            elif isinstance(v, str) and len(v) > 512:
                m[k] = v[:512]

    return m

# ---------------------------
# Store
# ---------------------------

SKIP_EXISTING = os.getenv("SKIP_EXISTING", "true").lower() in ("1", "true", "yes")
MINIMAL_META = os.getenv("MINIMAL_META", "false").lower() in ("1", "true", "yes")

class ListingStore:
    def __init__(self, collection: str = None, force_recreate: bool = False):
        self.index_name = os.getenv("PINECONE_INDEX", "listing")
        self.default_ns = os.getenv("PINECONE_NAMESPACE", collection) or "listings_openai"

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        try:
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            log.exception("Could not initialize Pinecone client/index: %s", e)
            raise

        try:
            stats = self.index.describe_index_stats()
            sd = _resp_to_dict(stats)
            ns_keys = list((sd.get("namespaces") or {}).keys())
            log.info(
                "Pinecone OK: index=%s total=%s namespaces=%s",
                self.index_name, sd.get("total_vector_count"), ns_keys
            )
            if self.default_ns not in ns_keys:
                log.warning(
                    "Default namespace '%s' not in index stats; set PINECONE_NAMESPACE correctly.",
                    self.default_ns
                )
        except Exception as e:
            log.warning("describe_index_stats failed: %s", e)

        self.collection = None
        log.info(
            "Embed model=%s | namespace(default)=%s",
            os.getenv("OPENAI_EMBED_MODEL", OPENAI_EMBED_MODEL), self.default_ns
        )

    # --- Upsert ---

    def upsert(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> int:
        if not items:
            log.warning("No items to upsert")
            return 0

        ns = namespace or self.default_ns

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for it in items:
            if "id" not in it or not it.get("text"):
                continue
            ids.append(str(it["id"]))
            docs.append(it["text"])
            m = it.get("metadata") or {}
            clean: Dict[str, Any] = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    clean[k] = v
                else:
                    try:
                        clean[k] = str(v)
                    except Exception:
                        pass
            clean.setdefault("document", it["text"])
            metas.append(clean)

        if not ids:
            log.warning("No valid items (missing id/text)")
            return 0

        to_process_idxs = list(range(len(ids)))
        if SKIP_EXISTING:
            try:
                t0 = time.perf_counter()
                resp = self.index.fetch(ids=ids, namespace=ns)
                d = _resp_to_dict(resp)
                existing = set(d.get("vectors", {}).keys())
                t1 = time.perf_counter()
                log.info(
                    "Fetch existing ids: checked=%d existed=%d (%.2fs)",
                    len(ids), len(existing), t1 - t0
                )
                if existing:
                    new_idxs = [i for i, _id in enumerate(ids) if _id not in existing]
                    skipped = len(ids) - len(new_idxs)
                    to_process_idxs = new_idxs
                    log.info("Skipping %d already-present items", skipped)
                else:
                    to_process_idxs = list(range(len(ids)))
            except Exception as e:
                log.warning(
                    "Could not fetch existing ids from Pinecone (proceeding to upsert all): %s",
                    e
                )
                to_process_idxs = list(range(len(ids)))

        if not to_process_idxs:
            log.info("All items already present in namespace=%s; nothing to upsert.", ns)
            return 0

        # Optionally trim aggressively
        if MINIMAL_META:
            for i in to_process_idxs:
                meta = metas[i]
                for k in ("images", "source", "url"):
                    if k in meta:
                        meta.pop(k, None)
                meta.pop("document", None)
                meta["snippet"] = (docs[i])[:800]
                metas[i] = meta

        # Prepare embed + metadata (shrunken under 40KB)
        docs_to_embed = [docs[i] for i in to_process_idxs]
        ids_to_upsert = [ids[i] for i in to_process_idxs]
        metas_to_upsert: List[Dict[str, Any]] = []
        for idx in to_process_idxs:
            compact = _shrink_meta(metas[idx], docs[idx], soft_budget=META_SOFT_BUDGET)
            metas_to_upsert.append(compact)

        # Embed
        t0 = time.perf_counter()
        try:
            log.info("Generating embeddings for %d texts", len(docs_to_embed))
            embs = embed_text(docs_to_embed)
            log.info("Successfully generated %d embeddings", len(embs))
        except Exception as e:
            log.exception("embed_text failed for %d items: %s", len(docs_to_embed), e)
            raise
        t1 = time.perf_counter()
        log.info("Embedded %d texts in %.2fs", len(docs_to_embed), t1 - t0)

        # Upsert
        vectors = []
        for _id, emb, meta in zip(ids_to_upsert, embs, metas_to_upsert):
            vectors.append((_id, emb, meta))

        t_up_s = time.perf_counter()
        try:
            self.index.upsert(vectors=vectors, namespace=ns)
            t_up_e = time.perf_counter()
            log.info(
                "Upserted %d vectors into %s/%s in %.2fs",
                len(vectors), self.index_name, ns, t_up_e - t_up_s
            )
            return len(vectors)
        except Exception as e:
            log.exception("Pinecone upsert failed: %s", e)
            raise

    # --- Search ---

    def search(self, query_text: str, n: int = 20, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            log.info("Empty query -> []")
            return []

        areas = _areas_from_where(where)
        namespaces = _namespaces(self.default_ns, where)
        meta_filter = _meta_filter(where)

        try:
            q = embed_text([query_text])[0]
        except Exception as e:
            log.error("embed_text failed: %s", e, exc_info=True)
            return []

        results_per_ns: List[List[Dict[str, Any]]] = []
        for ns in namespaces:
            try:
                res = self.index.query(
                    vector=q,
                    top_k=min(max(n * 6, 120), 400),
                    include_values=False,
                    include_metadata=True,
                    namespace=ns,
                    filter=meta_filter or {}
                )
            except Exception as e:
                log.error("Pinecone query failed (ns=%s): %s", ns, e, exc_info=True)
                continue
            hits = []
            for m in getattr(res, "matches", []) or []:
                md = dict(getattr(m, "metadata", {}) or {})
                item = {"id": getattr(m, "id", None), "_score": float(getattr(m, "score", 0.0))}
                if "document" in md:
                    item["document"] = md["document"]
                item.update(md)
                hits.append(item)
            results_per_ns.append(hits)

        filtered_merged = _merge_unique(results_per_ns)
        if areas:
            filtered_merged = [r for r in filtered_merged if _matches_area(r, areas)]

        results_per_ns = []
        for ns in namespaces:
            try:
                res = self.index.query(
                    vector=q,
                    top_k=min(max(n * 8, 160), 400),
                    include_values=False,
                    include_metadata=True,
                    namespace=ns
                )
            except Exception as e:
                log.error("Pinecone wide query failed (ns=%s): %s", ns, e, exc_info=True)
                continue
            hits = []
            for m in getattr(res, "matches", []) or []:
                md = dict(getattr(m, "metadata", {}) or {})
                item = {"id": getattr(m, "id", None), "_score": float(getattr(m, "score", 0.0))}
                if "document" in md:
                    item["document"] = md["document"]
                item.update(md)
                hits.append(item)
            results_per_ns.append(hits)

        wide_merged = _merge_unique(results_per_ns)
        if areas:
            wide_merged = [r for r in wide_merged if _matches_area(r, areas)]

        combined = {(r.get("id") or r.get("url") or r.get("details_url") or str(i)): r for i, r in enumerate(wide_merged)}
        for r in filtered_merged:
            k = r.get("id") or r.get("url") or r.get("details_url") or None
            if k is not None:
                combined[k] = r
            else:
                combined[str(id(r))] = r

        out = list(combined.values())
        out.sort(
            key=lambda x: (
                -float(x.get("_score", 0.0)),
                str(x.get("id") or x.get("url") or x.get("details_url") or "")
            )
        )
        return out[:n]

    # --- All / Count ---

    def all(self, limit: int = 50) -> List[Dict[str, Any]]:
        log.warning("ListingStore.all() not supported with Pinecone. Return [].")
        return []

    def count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            sd = _resp_to_dict(stats)
            return int(sd.get("total_vector_count") or 0)
        except Exception as e:
            log.warning("Could not describe index stats: %s", e)
            return 0
