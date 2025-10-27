# app/services/vector_store.py
import os
import time
import logging
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
# Area helpers (unchanged)
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
        if w in s: return w
    for c in CITIES:
        if c in s: return c
    return s

def _areas_from_where(where: Optional[Dict[str, Any]]) -> List[str]:
    if not where: return []
    out: List[str] = []
    for k in ("areas","area","prefecture","city","ward","neighborhood","area_like"):
        v = where.get(k)
        if v is None: continue
        if isinstance(v, (list,tuple)):
            out.extend([_normalize_area(x) for x in v if x])
        else:
            out.append(_normalize_area(v))
    # unique preserve order
    seen=set(); uniq=[]
    for x in out:
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def _matches_area(meta: Dict[str, Any], desired: List[str]) -> bool:
    if not desired: return True
    hay=[]
    for k in ("prefecture","city","ward","area","neighborhood"):
        v = meta.get(k)
        if isinstance(v,str): hay.append(v)
    h = " ".join(hay)
    if not h: return False
    for d in desired:
        if d and (d == h or d in h): return True
    return False

def _merge_unique(lists: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen=set(); out=[]
    for L in lists:
        for r in L:
            rid=r.get("id")
            if rid in seen: continue
            seen.add(rid); out.append(r)
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
    """
    Pinecone metadata filter. Keep it conservative.
    NOTE: We do NOT add 'bedrooms' filter because your JP docs don't store it.
    """
    if not where: return {}
    f: Dict[str, Any] = {}

    # Prefer yen. Do not include USD when yen is provided.
    if where.get("max_budget_yen") is not None:
        v=_num(where["max_budget_yen"])
        if v is not None:
            f["price_yen"] = {"$lte": v}
    elif where.get("max_budget_usd") is not None:
        v=_num(where["max_budget_usd"])
        if v is not None:
            f["price_usd"] = {"$lte": v}

    # Optional booleans you actually store
    if "pet_dog" in where and where.get("pet_dog") is not None:
        f["pet_dog"] = {"$eq": bool(where["pet_dog"])}

    return f

# ---------------------------
# Namespace selection
# ---------------------------

def _namespaces(default_ns: str, where: Optional[Dict[str, Any]]) -> List[str]:
    # IMPORTANT: you have data in 'listings_openai'
    explicit = (where or {}).get("namespace")
    if explicit: return [explicit]

    areas = _areas_from_where(where)
    if not areas: return [default_ns or "listings_openai"]

    # If you actually shard per-area, map here; else always use your data namespace.
    return [default_ns or "listings_openai"]

# ---------------------------
# Utilities for Pinecone typed responses
# ---------------------------

def _resp_to_dict(resp: Any) -> Dict[str, Any]:
    """
    Convert various pinecone response types to a plain dict safely.
    Works with typed objects (has .to_dict()), older dict-like returns, or objects with .vectors attr.
    """
    try:
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
        if hasattr(resp, "vectors"):
            try:
                return {"vectors": dict(resp.vectors)}
            except Exception:
                # try iterating
                return {"vectors": {k: v for k, v in resp.vectors.items()}}
        # attempt to cast
        return dict(resp)
    except Exception:
        # last-ditch: build minimal dict
        out = {}
        if hasattr(resp, "vectors"):
            try:
                out["vectors"] = dict(resp.vectors)
            except Exception:
                out["vectors"] = {}
        return out

# ---------------------------
# Store (rewritten to use new Pinecone client + batching)
# ---------------------------

# runtime options
SKIP_EXISTING = os.getenv("SKIP_EXISTING", "true").lower() in ("1","true","yes")
MINIMAL_META = os.getenv("MINIMAL_META", "false").lower() in ("1","true","yes")

class ListingStore:
    """
    Pinecone-backed listing store.
    Expects metadata fields like: prefecture/city/ward/area/neighborhood, price_yen, etc.
    """

    def __init__(self, collection: str = None, force_recreate: bool = False):
        self.index_name = os.getenv("PINECONE_INDEX", "listing")  # ← your index
        # prefer env var PINECONE_NAMESPACE if set, otherwise use supplied collection or default
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

        # Log stats so you can confirm we’re hitting the right place
        try:
            stats = self.index.describe_index_stats()
            sd = _resp_to_dict(stats)
            ns_keys = list((sd.get("namespaces") or {}).keys())
            log.info("Pinecone OK: index=%s total=%s namespaces=%s",
                     self.index_name, sd.get("total_vector_count"), ns_keys)
            if self.default_ns not in ns_keys:
                log.warning("Default namespace '%s' not in index stats; set PINECONE_NAMESPACE correctly.", self.default_ns)
        except Exception as e:
            log.warning("describe_index_stats failed: %s", e)

        # For compatibility with the rest of your app
        self.collection = None
        log.info("Embed model=%s | namespace(default)=%s", os.getenv("OPENAI_EMBED_MODEL", OPENAI_EMBED_MODEL), self.default_ns)

    # --- Upsert (improved) ---

    def upsert(self, items: List[Dict[str, Any]], namespace: Optional[str] = None) -> int:
        """
        items: list of {"id": id_str, "text": "...", "metadata": {...}}
        Returns: number of vectors actually upserted (0 if none).
        """
        if not items:
            log.warning("No items to upsert")
            return 0

        ns = namespace or self.default_ns

        ids = []
        docs = []
        metas = []
        for it in items:
            if "id" not in it or not it.get("text"):
                continue
            ids.append(str(it["id"]))
            docs.append(it["text"])
            m = it.get("metadata") or {}
            clean = {}
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

        # 1) Fetch existing ids in one call and skip them if configured
        to_process_idxs = list(range(len(ids)))
        if SKIP_EXISTING:
            try:
                t0 = time.perf_counter()
                resp = self.index.fetch(ids=ids, namespace=ns)
                d = _resp_to_dict(resp)
                existing = set(d.get("vectors", {}).keys())
                t1 = time.perf_counter()
                log.info("Fetch existing ids: checked=%d existed=%d (%.2fs)", len(ids), len(existing), t1 - t0)
                if existing:
                    new_idxs = [i for i, _id in enumerate(ids) if _id not in existing]
                    skipped = len(ids) - len(new_idxs)
                    to_process_idxs = new_idxs
                    log.info("Skipping %d already-present items", skipped)
                else:
                    to_process_idxs = list(range(len(ids)))
            except Exception as e:
                log.warning("Could not fetch existing ids from Pinecone (proceeding to upsert all): %s", e)
                to_process_idxs = list(range(len(ids)))

        if not to_process_idxs:
            log.info("All items already present in namespace=%s; nothing to upsert.", ns)
            return 0

        # 2) Optionally trim metadata to reduce payload size
        if MINIMAL_META:
            for i in to_process_idxs:
                meta = metas[i]
                for k in ("images", "source", "url"):
                    if k in meta:
                        meta.pop(k, None)
                metas[i] = meta

        # 3) Prepare lists to embed & upsert
        docs_to_embed = [docs[i] for i in to_process_idxs]
        ids_to_upsert = [ids[i] for i in to_process_idxs]
        metas_to_upsert = [metas[i] for i in to_process_idxs]

        # 4) Embed
        t0 = time.perf_counter()
        try:
            embs = embed_text(docs_to_embed)
        except Exception as e:
            log.exception("embed_text failed for %d items: %s", len(docs_to_embed), e)
            raise
        t1 = time.perf_counter()
        log.info("Embedded %d texts in %.2fs", len(docs_to_embed), t1 - t0)

        # 5) Upsert batch
        vectors = []
        for _id, emb, meta in zip(ids_to_upsert, embs, metas_to_upsert):
            vectors.append((_id, emb, meta))

        t_up_s = time.perf_counter()
        try:
            self.index.upsert(vectors=vectors, namespace=ns)
            t_up_e = time.perf_counter()
            log.info("Upserted %d vectors into %s/%s in %.2fs", len(vectors), self.index_name, ns, t_up_e - t_up_s)
            return len(vectors)
        except Exception as e:
            log.exception("Pinecone upsert failed: %s", e)
            raise

    # --- Search (keeps existing behavior) ---

    def search(self, query_text: str, n: int = 20, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            log.info("Empty query -> []")
            return []

        areas = _areas_from_where(where)
        namespaces = _namespaces(self.default_ns, where)
        meta_filter = _meta_filter(where)

        # Query embedding
        try:
            q = embed_text([query_text])[0]
        except Exception as e:
            log.error("embed_text failed: %s", e, exc_info=True)
            return []

        # Stage 1: filtered vector query (higher recall)
        results_per_ns: List[List[Dict[str, Any]]] = []
        for ns in namespaces:
            try:
                res = self.index.query(
                    vector=q,
                    top_k=min(max(n*6, 120), 400),
                    include_values=False,
                    include_metadata=True,
                    namespace=ns,
                    filter=meta_filter or {}
                )
            except Exception as e:
                log.error("Pinecone query failed (ns=%s): %s", ns, e, exc_info=True)
                continue
            hits=[]
            for m in getattr(res, "matches", []) or []:
                md = dict(getattr(m, "metadata", {}) or {})
                item={"id": getattr(m, "id", None), "_score": float(getattr(m, "score", 0.0))}
                if "document" in md: item["document"]=md["document"]
                item.update(md)
                hits.append(item)
            results_per_ns.append(hits)

        filtered_merged = _merge_unique(results_per_ns)
        if areas:
            filtered_merged = [r for r in filtered_merged if _matches_area(r, areas)]

        # Stage 2: wide vector (no numeric filters)
        results_per_ns=[]
        for ns in namespaces:
            try:
                res = self.index.query(
                    vector=q,
                    top_k=min(max(n*8, 160), 400),
                    include_values=False,
                    include_metadata=True,
                    namespace=ns
                )
            except Exception as e:
                log.error("Pinecone wide query failed (ns=%s): %s", ns, e, exc_info=True)
                continue
            hits=[]
            for m in getattr(res, "matches", []) or []:
                md = dict(getattr(m, "metadata", {}) or {})
                item={"id": getattr(m, "id", None), "_score": float(getattr(m, "score", 0.0))}
                if "document" in md: item["document"]=md["document"]
                item.update(md)
                hits.append(item)
            results_per_ns.append(hits)

        wide_merged = _merge_unique(results_per_ns)
        if areas:
            wide_merged = [r for r in wide_merged if _matches_area(r, areas)]

        # Combine both passes deterministically: prefer filtered scores, break ties by id/url
        combined = { (r.get("id") or r.get("url") or r.get("details_url") or str(i)) : r for i, r in enumerate(wide_merged) }
        for r in filtered_merged:
            k = r.get("id") or r.get("url") or r.get("details_url") or None
            if k is not None:
                combined[k] = r
            else:
                combined[str(id(r))] = r
        out = list(combined.values())
        out.sort(key=lambda x: (-float(x.get("_score", 0.0)), str(x.get("id") or x.get("url") or x.get("details_url") or "")))
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
