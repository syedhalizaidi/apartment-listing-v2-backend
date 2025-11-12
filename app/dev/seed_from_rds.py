#!/usr/bin/env python3
"""
Postgres RDS → Vector Store seeding with resume support.

- Reads listings from a Postgres table
- Embeds/upserts into your ListingStore (Pinecone, etc.)
- Tracks progress so subsequent runs resume cleanly
- Optional in-table flags: embedded / embedded_at

ENV:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname
  LISTINGS_TABLE=listings
  LISTINGS_ID_COLUMN=id
  VECTOR_COLLECTION=listings_openai
  SEED_BATCH=64
  FORCE_RECREATE=false
  USE_EMBEDDED_FLAGS=0            # if 1 → WHERE embedded=false and then mark true
  LISTINGS_WHERE=                 # optional extra predicate

  # Optional column overrides ("" to disable a field)
  COL_TITLE=title
  COL_DESCRIPTION=description
  COL_LOCATION=location
  COL_PRICE_YEN_MIN=price_yen_min
  COL_PRICE_YEN=price_yen
  COL_RENT_YEN=rent_yen
  COL_PRICE_TEXT=price_text
  COL_DETAILS_URL=details_url
  COL_SOURCE_URL=source_url
  COL_IMAGES=image_urls
  COL_BEDROOMS=bedrooms
  COL_BATHROOMS=bathrooms
  COL_PET_DOG=pet_dog
  COL_CITY=city
  COL_PREFECTURE=prefecture
  COL_NEIGHBORHOOD=neighborhood
"""

import os
import re
import sys
import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple
from hashlib import md5

try:
    import orjson as jsonlib
except Exception:
    import json as jsonlib  # fallback

import psycopg2
import psycopg2.extras

# ---------- flexible imports (works in package or flat script) ----------
ListingStore = None
_imp_err = []
for mod in (
    ("app.services.vector_store", "ListingStore"),
    ("services.vector_store", "ListingStore"),
    ("vector_store", "ListingStore"),
):
    try:
        ListingStore = __import__(mod[0], fromlist=[mod[1]]).__dict__[mod[1]]
        break
    except Exception as e:
        _imp_err.append(f"{mod[0]}: {e!r}")
        continue
if ListingStore is None:
    raise RuntimeError("Could not import ListingStore. Tried: " + " | ".join(_imp_err))

# -----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("seed_rds")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    log.error("DATABASE_URL is required")
    sys.exit(1)

TBL = os.getenv("LISTINGS_TABLE", "listings")
IDCOL = os.getenv("LISTINGS_ID_COLUMN", "id")
COL = lambda k, d: os.getenv(k, d)

VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "listings_openai")
BATCH = int(os.getenv("SEED_BATCH", "64"))
FORCE_RECREATE = os.getenv("FORCE_RECREATE", "false").lower() in ("1", "true", "yes")
USE_EMBEDDED_FLAGS = os.getenv("USE_EMBEDDED_FLAGS", "0").lower() in ("1", "true", "yes")
LISTINGS_WHERE = (os.getenv("LISTINGS_WHERE") or "").strip()

# column names (override via env if needed)
C_TITLE         = COL("COL_TITLE", "title")
C_DESC          = COL("COL_DESCRIPTION", "description")
C_LOC           = COL("COL_LOCATION", "location")
C_PRICE_MIN     = COL("COL_PRICE_YEN_MIN", "price_yen_min")
C_PRICE_YEN     = COL("COL_PRICE_YEN", "price_yen")
C_RENT_YEN      = COL("COL_RENT_YEN", "rent_yen")
C_PRICE_TEXT    = COL("COL_PRICE_TEXT", "price_text")
C_DETAILS_URL   = COL("COL_DETAILS_URL", "details_url")
C_SOURCE_URL    = COL("COL_SOURCE_URL", "source_url")
C_IMAGES        = COL("COL_IMAGES", "image_urls")
C_BEDROOMS      = COL("COL_BEDROOMS", "bedrooms")
C_BATHROOMS     = COL("COL_BATHROOMS", "bathrooms")
C_PET_DOG       = COL("COL_PET_DOG", "pet_dog")
C_CITY          = COL("COL_CITY", "city")
C_PREF          = COL("COL_PREFECTURE", "prefecture")
C_HOOD          = COL("COL_NEIGHBORHOOD", "neighborhood")

YEN_RX = re.compile(r"(?:¥|円|万|千|,|\s)")

PROG_TABLE = "embedding_progress"
PROG_KEY = TBL  # one progress row per source table

# ---------- PG helpers ----------
def _pg():
    return psycopg2.connect(DATABASE_URL)

def _ensure_progress_table(cur):
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {PROG_TABLE} (
        source_table TEXT PRIMARY KEY,
        last_id BIGINT DEFAULT 0,
        total_embedded BIGINT DEFAULT 0,
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    """)

def _get_progress(cur) -> Tuple[int, int]:
    cur.execute(f"SELECT last_id, total_embedded FROM {PROG_TABLE} WHERE source_table=%s;", (PROG_KEY,))
    row = cur.fetchone()
    if not row:
        cur.execute(
            f"""INSERT INTO {PROG_TABLE}(source_table, last_id, total_embedded)
                VALUES (%s, 0, 0)
                ON CONFLICT (source_table) DO NOTHING;""",
            (PROG_KEY,)
        )
        return 0, 0
    # RealDictCursor returns dict
    if isinstance(row, dict):
        return int(row.get("last_id", 0)), int(row.get("total_embedded", 0))
    return int(row[0]), int(row[1])

def _set_progress(cur, last_id: int, add_count: int):
    cur.execute(
        f"""UPDATE {PROG_TABLE}
              SET last_id = last_id + %s,
                  total_embedded = total_embedded + %s,
                  updated_at = now()
            WHERE source_table=%s;""",
        (add_count, add_count, PROG_KEY)
    )
def _mark_embedded(cur, ids: List[Any]):
    if not ids:
        return
    # Use a parameterized query with IN clause
    params = tuple(ids)
    placeholders = ','.join(['%s'] * len(params))
    q = f"""
        UPDATE {TBL}
           SET embedded = true,
               embedded_at = now()
         WHERE {IDCOL} IN ({placeholders});
    """
    cur.execute(q, params)

def _table_exists(cur, table: str) -> bool:
    cur.execute("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='public' AND table_name=%s
        LIMIT 1;
    """, (table,))
    row = cur.fetchone()
    return bool((row or {}).get("?column?") if isinstance(row, dict) else row)

def _get_columns(cur, table: str) -> List[str]:
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s;
    """, (table,))
    rows = cur.fetchall()
    cols = []
    for r in rows:
        if isinstance(r, dict):
            cols.append(r.get("column_name"))
        else:
            cols.append(r[0])
    return [c for c in cols if c]

def _assert_source_table(cur):
    if not _table_exists(cur, TBL):
        raise RuntimeError(f'Source table "{TBL}" not found in schema "public". Set LISTINGS_TABLE or create the table.')

    cols = set(_get_columns(cur, TBL))
    required = {IDCOL}
    # title/location/url are strongly recommended but you may disable by setting env to ""
    if C_TITLE:       required.add(C_TITLE)
    if C_LOC:         required.add(C_LOC)
    if C_DETAILS_URL: required.add(C_DETAILS_URL)

    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(
            f"Missing required columns in {TBL}: {', '.join(missing)}. Present: {', '.join(sorted(cols))}"
        )

# ---------- value helpers ----------
def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return int(v)
        except Exception:
            return None
    if isinstance(v, str):
        try:
            vv = YEN_RX.sub("", v)
            return int(float(vv))
        except Exception:
            return None
    return None

def _pick_price_yen_row(row: Dict[str, Any]) -> Optional[int]:
    for key in (C_PRICE_MIN, C_PRICE_YEN, C_RENT_YEN):
        if key and key in row:
            val = _as_int(row.get(key))
            if val:
                return val
    s = (row.get(C_PRICE_TEXT) or "") if (C_PRICE_TEXT and C_PRICE_TEXT in row) else ""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*万", s)
    if m:
        try:
            return int(float(m.group(1)) * 10000)
        except Exception:
            pass
    m2 = re.search(r"(?:¥|円)\s*([\d,]+)", s)
    if m2:
        try:
            return int(m2.group(1).replace(",", ""))
        except Exception:
            pass
    return None

def _parse_images(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, (dict,)):
        out = []
        for k in ("images", "image_urls", "urls", "photos"):
            if v.get(k):
                if isinstance(v[k], list):
                    out.extend([str(x) for x in v[k] if x])
                else:
                    out.append(str(v[k]))
        return out
    if isinstance(v, (bytes, bytearray)):
        try:
            obj = jsonlib.loads(v)
            return _parse_images(obj)
        except Exception:
            s = v.decode("utf-8", "ignore")
            return [x.strip() for x in s.split(",") if x.strip()]
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = jsonlib.loads(s)
                return _parse_images(obj)
            except Exception:
                pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []

def _format_text(row: Dict[str, Any], price_yen: Optional[int]) -> str:
    parts = []
    if C_TITLE: parts.append(str(row.get(C_TITLE) or ""))
    if C_DESC:  parts.append(str(row.get(C_DESC) or ""))
    if C_LOC:   parts.append(str(row.get(C_LOC) or ""))
    if price_yen is not None:
        parts.append(f"Price: ¥{price_yen:,}")
    if C_PRICE_TEXT and row.get(C_PRICE_TEXT):
        parts.append(str(row[C_PRICE_TEXT]))
    if C_DETAILS_URL and row.get(C_DETAILS_URL):
        parts.append(str(row[C_DETAILS_URL]))

    # Optional extras (common in your schema)
    for extra_key, label in [
        ("station", "Nearest"),
        ("layout", "Layout"),
        ("floor_plan", "Layout"),
        ("size_m2", "Size m²"),
        ("deposit", "Deposit"),
        ("key_money", "Key money"),
        ("city_label", "City"),
        ("area", "Area"),
        ("built_year", "Built"),
    ]:
        if extra_key in row and row.get(extra_key) not in (None, ""):
            parts.append(f"{label}: {row.get(extra_key)}")

    return " ".join(p for p in parts if p)

def _rows_to_items(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, List[Any]]:
    items: List[Dict[str, Any]] = []
    max_id = 0
    ids_for_flag: List[Any] = []

    for r in rows:
        rid = r.get(IDCOL)
        if rid is not None:
            ids_for_flag.append(rid)

        # derive a stable string id
        if rid is None:
            base = str((C_DETAILS_URL and r.get(C_DETAILS_URL)) or f"{r.get(C_TITLE,'')}-{r.get(C_LOC,'')}")
            rid_final = md5(base.encode("utf-8")).hexdigest()
        else:
            rid_final = str(rid)
            try:
                iv = int(rid)
                if iv > max_id:
                    max_id = iv
            except Exception:
                pass

        price_yen = _pick_price_yen_row(r)
        images = _parse_images(r.get(C_IMAGES)) if C_IMAGES else []

        meta = {
            "title": r.get(C_TITLE) or "",
            "description": r.get(C_DESC) or "",
            "price_yen": price_yen,
            "bedrooms": r.get(C_BEDROOMS) if C_BEDROOMS else None,
            "bathrooms": r.get(C_BATHROOMS) if C_BATHROOMS else None,
            "pet_dog": bool(r.get(C_PET_DOG)) if (C_PET_DOG and r.get(C_PET_DOG) is not None) else None,
            "neighborhood": r.get(C_HOOD) or "" if C_HOOD else "",
            "city": r.get(C_CITY) or "" if C_CITY else "",
            "prefecture": r.get(C_PREF) or "" if C_PREF else "",
            "deal_type": "rent",
            "url": r.get(C_DETAILS_URL) or "" if C_DETAILS_URL else "",
            "images": images,
            "source": r.get(C_SOURCE_URL) or "" if C_SOURCE_URL else "",
            "timestamp": dt.datetime.now().isoformat(),
        }
        meta = {k: v for k, v in meta.items() if v not in (None, "", [])}

        text = _format_text(r, price_yen)
        items.append({"id": rid_final, "text": text, "metadata": meta})

    return items, max_id, ids_for_flag

# ---------- Column & query composition ----------
def _compose_select_columns(existing_cols: List[str]) -> List[str]:
    wanted = [
        IDCOL, C_TITLE, C_DESC, C_LOC, C_PRICE_MIN, C_PRICE_YEN, C_RENT_YEN, C_PRICE_TEXT,
        C_DETAILS_URL, C_SOURCE_URL, C_IMAGES, C_BEDROOMS, C_BATHROOMS, C_PET_DOG,
        C_CITY, C_PREF, C_HOOD,
        # extras common in your table:
        "station", "layout", "floor_plan", "size_m2", "deposit", "key_money", "city_label", "area", "built_year",
    ]
    if USE_EMBEDDED_FLAGS:
        wanted += ["embedded", "embedded_at"]

    have = set(existing_cols)
    cols = [c for c in wanted if c and c in have]
    cols = [IDCOL] + [c for c in cols if c != IDCOL]
    # dedupe while preserving order
    seen = set(); out = []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _fetch_batch(cur, last_id: int, limit: int, select_cols: List[str]) -> List[Dict[str, Any]]:
    collist = ", ".join(select_cols)
    predicates = []
    params: List[Any] = []

    if USE_EMBEDDED_FLAGS and "embedded" in select_cols:
        predicates.append("embedded = false")

    if LISTINGS_WHERE:
        predicates.append(f"({LISTINGS_WHERE})")

    # Instead of relying on numeric id > last_id, we use OFFSET pagination
    where_sql = ("WHERE " + " AND ".join(predicates)) if predicates else ""
    order_sql = f"ORDER BY {IDCOL} ASC" if IDCOL in select_cols else ""
    limit_sql = "LIMIT %s OFFSET %s"
    params.extend([limit, last_id])  # use last_id as offset

    q = f"SELECT {collist} FROM {TBL} {where_sql} {order_sql} {limit_sql};"

    try:
        cur.execute(q, tuple(params))
    except Exception as e:
        raise RuntimeError(f"SELECT failed: {e} | table={TBL} | idcol={IDCOL} | where={where_sql} | cols=({collist})")

    return cur.fetchall()

# ---------- Main ----------
def main() -> int:
    log.info(
        "Starting RDS seeding: table=%s idcol=%s collection=%s batch=%d mode=%s",
        TBL, IDCOL, VECTOR_COLLECTION, BATCH, "flags" if USE_EMBEDDED_FLAGS else "progress_table"
    )
    store = ListingStore(collection=VECTOR_COLLECTION, force_recreate=FORCE_RECREATE)

    with _pg() as conn:
        conn.autocommit = False
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            _assert_source_table(cur)
            existing_cols = _get_columns(cur, TBL)
            select_cols = _compose_select_columns(existing_cols)

            if not USE_EMBEDDED_FLAGS:
                _ensure_progress_table(cur)
                last_id, total_done = _get_progress(cur)
            else:
                last_id, total_done = 0, 0

            log.info("Progress: last_id=%s total_embedded=%s", last_id, total_done)

            grand_added = 0
            while True:
                rows = _fetch_batch(cur, last_id, BATCH, select_cols)
                if not rows:
                    break

                items, max_id, ids_for_flag = _rows_to_items(rows)
                if not items:
                    break

                inserted = store.upsert(items)  # should return number inserted
                grand_added += inserted

                if USE_EMBEDDED_FLAGS:
                    _mark_embedded(cur, ids_for_flag)
                else:
                    last_id = max(last_id, max_id)
                    _set_progress(cur, last_id, inserted)

                conn.commit()
                log.info(
                    "Batch upserted=%d  new_last_id=%s  total_so_far=%d  mode=%s",
                    inserted, last_id, total_done + grand_added, "flags" if USE_EMBEDDED_FLAGS else "progress"
                )

    # Post-run stats
    try:
        cnt = store.count()
        log.info("Vector store count (approx): %s", cnt)
    except Exception as e:
        log.info("Could not fetch vector count: %s", e)

    try:
        probe = store.search("賃貸 マンション ワンルーム 東京", n=3)
        log.info("Probe results: %d", len(probe))
    except Exception as e:
        log.info("Probe search failed: %s", e)

    log.info("Done. Added %d new items.", grand_added)
    return int(grand_added)

if __name__ == "__main__":
    added = main()
    print(f"Added {added} items")
