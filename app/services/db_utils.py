# app/services/db_utils.py
"""
Postgres utilities for scraped listings (NO EMBEDDING/NO VECTOR WORK).

Usage examples (shell):
    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
    python -c "from app.services.db_utils import init_db; init_db()"

Typical flow:
  1) init_db()
  2) insert_listings_from_jsonl('/abs/path/jp_listings.jsonl')
  3) query listings with fetch_by_ids or other helpers
"""

from __future__ import annotations
import os
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger("db_utils")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL", None)
if not DATABASE_URL:
    log.warning("DATABASE_URL not set; DB operations will fail until set.")


# ---------- Helpers ----------

def _get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # use RealDictCursor for convenience (dict-like rows)
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _id_from_record(rec: Dict[str, Any]) -> str:
    """
    Deterministic id used by seed pipeline: md5(details_url) if details_url present,
    otherwise md5(title + location).
    """
    url = (rec.get("details_url") or "").strip()
    if url:
        base = url
    else:
        base = f"{rec.get('title','')}-{rec.get('location','')}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


# ---------- Schema / init ----------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS listings (
  id TEXT PRIMARY KEY,
  title TEXT,
  address TEXT,
  location TEXT,
  layout TEXT,
  size_m2 NUMERIC,
  deposit TEXT,
  key_money TEXT,
  price_text TEXT,
  price_yen_min INTEGER,
  price_yen_max INTEGER,
  image_urls JSONB,
  details_url TEXT,
  source_url TEXT,
  city_label TEXT,
  scraped_at TIMESTAMP,
  inserted_at TIMESTAMP DEFAULT now(),
  embedded BOOLEAN DEFAULT false,
  embedded_at TIMESTAMP,
  extras JSONB
);
CREATE INDEX IF NOT EXISTS idx_listings_details_url ON listings(details_url);
CREATE INDEX IF NOT EXISTS idx_listings_embedded ON listings(embedded);
"""

def init_db():
    """Create table and indexes if not present."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
        log.info("DB initialized (table 'listings' created if not exists).")
    finally:
        conn.close()


# ---------- Insert / upsert / delete ----------

def upsert_listing(record: Dict[str, Any]) -> str:
    """
    Insert or update a single listing. Returns the id used.
    """
    rec = dict(record)
    idv = _id_from_record(rec)
    rec['id'] = idv

    # normalize some fields
    if 'image_urls' in rec and isinstance(rec['image_urls'], (list, tuple)):
        # store as real JSON object; psycopg2 will adapt
        rec['image_urls'] = rec['image_urls']

    # extras: keep any unexpected fields
    allowed = {"id","title","address","location","layout","size_m2","deposit","key_money","price_text",
               "price_yen_min","price_yen_max","image_urls","details_url","source_url","city_label","scraped_at","extras"}
    extras = {}
    for k in list(rec.keys()):
        if k not in allowed:
            extras[k] = rec.pop(k)

    rec['extras'] = extras or None

    # Build SQL dynamically and use parameters
    cols = list(rec.keys())
    placeholders = ", ".join([f"%({k})s" for k in cols])
    cols_sql = ", ".join(cols)
    updates = ", ".join([f"{k} = EXCLUDED.{k}" for k in cols if k != "id"])

    sql = f"""
    INSERT INTO listings ({cols_sql})
    VALUES ({placeholders})
    ON CONFLICT (id) DO UPDATE SET {updates};
    """

    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, rec)
        log.debug("Upserted listing id=%s", idv)
    finally:
        conn.close()
    return idv


def insert_listings(records: List[Dict[str, Any]], batch_size: int = 200) -> int:
    """
    Bulk insert/upsert many records using execute_values for speed.
    Returns number processed.
    """
    if not records:
        return 0

    rows = []
    for rec in records:
        r = dict(rec)
        _id = _id_from_record(r)
        r['id'] = _id
        # move unknown keys into extras
        allowed = {"id","title","address","location","layout","size_m2","deposit","key_money","price_text",
                   "price_yen_min","price_yen_max","image_urls","details_url","source_url","city_label","scraped_at"}
        extras = {}
        for k in list(r.keys()):
            if k not in allowed:
                extras[k] = r.pop(k)
        r['extras'] = extras or None
        rows.append(r)

    conn = _get_conn()
    inserted = 0
    try:
        with conn:
            with conn.cursor() as cur:
                # prepare columns - allow missing keys by mapping
                cols = ["id","title","address","location","layout","size_m2","deposit","key_money","price_text",
                        "price_yen_min","price_yen_max","image_urls","details_url","source_url","city_label","scraped_at","extras"]
                template = "(" + ",".join([f"%({c})s" for c in cols]) + ")"
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i:i+batch_size]
                    psycopg2.extras.execute_values(
                        cur,
                        f"""
                        INSERT INTO listings ({", ".join(cols)})
                        VALUES %s
                        ON CONFLICT (id) DO UPDATE SET
                          title = EXCLUDED.title,
                          address = EXCLUDED.address,
                          location = EXCLUDED.location,
                          layout = EXCLUDED.layout,
                          size_m2 = EXCLUDED.size_m2,
                          deposit = EXCLUDED.deposit,
                          key_money = EXCLUDED.key_money,
                          price_text = EXCLUDED.price_text,
                          price_yen_min = EXCLUDED.price_yen_min,
                          price_yen_max = EXCLUDED.price_yen_max,
                          image_urls = EXCLUDED.image_urls,
                          details_url = EXCLUDED.details_url,
                          source_url = EXCLUDED.source_url,
                          city_label = EXCLUDED.city_label,
                          scraped_at = EXCLUDED.scraped_at,
                          extras = EXCLUDED.extras
                        """,
                        chunk,
                        template=template,
                        page_size=100
                    )
                    inserted += len(chunk)
        log.info("Inserted/Upserted %d listings", inserted)
    finally:
        conn.close()
    return inserted


def delete_listing(id_value: str) -> bool:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM listings WHERE id = %s", (id_value,))
                return cur.rowcount > 0
    finally:
        conn.close()


def update_listing(id_value: str, updates: Dict[str, Any]) -> bool:
    if not updates:
        return False
    set_parts = []
    params = {}
    for i, (k, v) in enumerate(updates.items()):
        param = f"v{i}"
        set_parts.append(f"{k} = %({param})s")
        params[param] = v
    params["idv"] = id_value
    sql = f"UPDATE listings SET {', '.join(set_parts)} WHERE id = %(idv)s"
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0
    finally:
        conn.close()


# ---------- Query helpers ----------

def fetch_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    """Return rows matching given ids (preserves order of results not guaranteed)."""
    if not ids:
        return []
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Use IN with tuple
                cur.execute("SELECT * FROM listings WHERE id = ANY(%s)", (ids,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_recent(limit: int = 200) -> List[Dict[str, Any]]:
    """Return most recently inserted/scraped rows (useful for inspection)."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM listings
                    ORDER BY scraped_at NULLS LAST, inserted_at DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------- JSONL loader (convenience) ----------

def insert_listings_from_jsonl(path: str, batch_size: int = 200) -> int:
    """
    Read a JSONL file (one JSON object per line) and upsert to the database.
    Returns number processed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    items = []
    count = 0
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                log.warning("Skipping invalid JSON line")
                continue
            items.append(obj)
            if len(items) >= batch_size:
                insert_listings(items, batch_size=batch_size)
                count += len(items)
                items = []
        if items:
            insert_listings(items, batch_size=batch_size)
            count += len(items)
    log.info("Loaded %d lines from %s", count, path)
    return count


# ---------- convenience CLI helpers ----------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true", help="create tables")
    p.add_argument("--load-jsonl", help="path to jsonl file to load")
    p.add_argument("--batch", type=int, default=200)
    args = p.parse_args()

    if args.init:
        init_db()
    if args.load_jsonl:
        insert_listings_from_jsonl(args.load_jsonl, batch_size=args.batch)
