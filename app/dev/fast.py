#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# airflow dags 
import argparse
import hashlib
import io
import json
import os
import sys
from typing import List, Dict, Any, Iterable

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ---- Supported embedding models and their dimensions ----
MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}

def stable_id(title: str, details_url: str) -> str:
    raw = (title or "") + "||" + (details_url or "")
    import hashlib
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def coerce_image_urls(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [v]
    return [str(v)]

def build_document(rec: Dict[str, Any]) -> str:
    title = rec.get("title") or ""
    location = rec.get("location") or ""
    description = rec.get("description") or ""
    details_url = rec.get("details_url") or rec.get("url") or ""
    source_url = rec.get("source_url") or rec.get("source") or ""
    image_urls = coerce_image_urls(rec.get("image_urls") or rec.get("images"))
    image_blob = " ".join(image_urls)
    parts = [
        f"物件名: {title}".strip(),
        f"所在地: {location}".strip(),
        f"説明: {description}".strip(),
        f"詳細URL: {details_url}".strip(),
        f"出典: {source_url}".strip(),
        f"画像: {image_blob}".strip(),
    ]
    for k in ["city", "prefecture", "neighborhood", "deal_type", "timestamp", "pet_dog"]:
        if k in rec and rec[k] is not None:
            parts.append(f"{k}: {rec[k]}")
    compact = " / ".join([title, location, description, details_url, source_url]).strip(" /")
    parts.append(f"コンパクト: {compact}")
    return "\n".join(p for p in parts if p)

def prepare_metadata(rec: Dict[str, Any], document_text: str) -> Dict[str, Any]:
    md = {
        "title": rec.get("title"),
        "location": rec.get("location"),
        "description": rec.get("description"),
        "details_url": rec.get("details_url") or rec.get("url"),
        "source_url": rec.get("source_url") or rec.get("source"),
        "images": coerce_image_urls(rec.get("image_urls") or rec.get("images")),
        "document": document_text,
    }
    for k in ["city", "prefecture", "neighborhood", "deal_type", "timestamp", "pet_dog", "url"]:
        if k in rec and rec[k] is not None:
            md[k] = rec[k]
    return md

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            s = raw.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                excerpt = raw[:120].replace("\n", "\\n")
                raise ValueError(
                    f"Bad JSON at line {lineno}: {e.msg} (col {e.colno}). Line excerpt: {excerpt!r}"
                ) from e

def read_records(path: str) -> List[Dict[str, Any]]:
    # Prefer JSONL; fall back to pretty-printed array/object
    try:
        recs = list(iter_jsonl(path))
        if recs:
            return recs
    except ValueError as jsonl_err:
        jsonl_error = jsonl_err
    else:
        jsonl_error = None
    try:
        with io.open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return list(data)
        raise ValueError("Top-level JSON must be an object or an array.")
    except Exception as e:
        if jsonl_error:
            raise RuntimeError(f"Failed JSONL parse and full-file JSON parse.\n"
                               f"- JSONL error: {jsonl_error}\n- Full-file error: {e}") from e
        raise

# ---- Pinecone helpers ----

def get_index_dim(pc: Pinecone, index_name: str) -> int | None:
    """Return index dimension if index exists, else None."""
    try:
        desc = pc.describe_index(index_name)
        # v5 returns dict-like with 'dimension'
        return int(desc.get("dimension")) if isinstance(desc, dict) else getattr(desc, "dimension", None)
    except Exception:
        # fallback via list_indexes
        try:
            li = pc.list_indexes()
            try:
                # For some SDK versions, list returns objects with .name and .dimension
                for it in li:
                    name = getattr(it, "name", None) or (isinstance(it, dict) and it.get("name"))
                    if name == index_name:
                        dim = getattr(it, "dimension", None) or (isinstance(it, dict) and it.get("dimension"))
                        return int(dim) if dim is not None else None
            except Exception:
                pass
        except Exception:
            pass
    return None

def ensure_index(pc: Pinecone, index_name: str, target_dim: int, recreate_on_mismatch: bool, region="us-east-1"):
    current_dim = get_index_dim(pc, index_name)
    if current_dim is None:
        pc.create_index(
            name=index_name,
            dimension=target_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
        return

    if current_dim != target_dim:
        if recreate_on_mismatch:
            print(f"[warn] Index '{index_name}' exists with dim {current_dim}, recreating with dim {target_dim}...")
            pc.delete_index(index_name)
            pc.create_index(
                name=index_name,
                dimension=target_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=region),
            )
            return
        else:
            sys.exit(
                f"Index '{index_name}' dimension is {current_dim}, but model requires {target_dim}.\n"
                f"Fix it by either:\n"
                f"  1) Use --model to match the existing index (dim {current_dim}), or\n"
                f"  2) Pass --recreate-on-mismatch to delete & recreate the index at {target_dim}, or\n"
                f"  3) Choose a different --index name.\n"
                f"Models: {', '.join([f'{m} ({d})' for m,d in MODEL_DIMS.items()])}"
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Pinecone index name")
    ap.add_argument("--input", required=True, help="Path to JSONL/JSON file (e.g., jp_listings.jsonl)")
    ap.add_argument("--namespace", default="listings_openai", help="Pinecone namespace")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--model", default="text-embedding-3-large", choices=list(MODEL_DIMS.keys()),
                    help="Embedding model to use")
    ap.add_argument("--region", default="us-east-1", help="Pinecone serverless region")
    ap.add_argument("--recreate-on-mismatch", action="store_true",
                    help="If index exists with different dimension, delete & recreate it to match the model")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Missing OPENAI_API_KEY")
    if not os.environ.get("PINECONE_API_KEY"):
        sys.exit("Missing PINECONE_API_KEY")

    target_dim = MODEL_DIMS[args.model]

    # clients
    oai = OpenAI()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # ensure index dimension aligns with model
    ensure_index(pc, args.index, target_dim, args.recreate_on_mismatch, region=args.region)
    index = pc.Index(args.index)

    # load records
    try:
        records = read_records(args.input)
    except Exception as e:
        sys.exit(f"Failed to read {args.input}: {e}")
    if not records:
        sys.exit("No valid records found in file.")

    # build docs + metadata + ids
    docs, metas, ids = [], [], []
    for rec in records:
        doc = build_document(rec)
        meta = prepare_metadata(rec, doc)
        doc_id = stable_id(meta.get("title") or "", meta.get("details_url") or "")
        docs.append(doc)
        metas.append(meta)
        ids.append(doc_id)

    # embed + upsert
    for bnum, idxs in enumerate(chunk(list(range(len(docs))), args.batch), start=1):
        inputs = [docs[i] for i in idxs]
        resp = oai.embeddings.create(model=args.model, input=inputs)
        embs = [d.embedding for d in resp.data]

        vectors = [
            {"id": ids[i], "values": embs[pos], "metadata": metas[i]}
            for pos, i in enumerate(idxs)
        ]
        index.upsert(vectors=vectors, namespace=args.namespace)
        print(f"Upserted batch {bnum}: {len(vectors)} vectors → ns={args.namespace}")

    print("Done ✅")

if __name__ == "__main__":
    main()
