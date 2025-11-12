#!/usr/bin/env python3
"""
Seed listings into the vector store from a JSONL file.
Supports:
- Legacy schema (title/location/details_url/price_text/price_yen_min...)
- Record format 1 (Village House)
- Record format 2 (Leopalace)

Usage:
  python seed.py
  DATA_PATH=/abs/path/to/jp_listings.jsonl python seed.py
  FORCE_RECREATE=true python seed.py
"""
import os
import re
import datetime as dt
import logging
from typing import Iterable, Dict, Any, Optional
from hashlib import md5

try:
    import orjson as jsonlib
except Exception:
    import json as jsonlib  # fallback

# ---------- flexible imports ----------
ListingStore = None
_imp_err = []

for mod in (
    ("app.services.vector_store", "ListingStore"),
    ("services.vector_store", "ListingStore"),
    ("vector_store", "ListingStore"),
    ("vectore_store", "ListingStore"),  # fallback if file name was misspelled
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
log = logging.getLogger("seed")

DEFAULT_FILE = "jp_listings.jsonl"
DATA_PATH = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), DEFAULT_FILE))

YEN_RX = re.compile(r"(?:¥|円|万|千|,|\s)")

MAX_IMAGES = int(os.getenv("SEED_MAX_IMAGES", "8"))
MAX_DESC_CHARS = int(os.getenv("SEED_MAX_DESC", "1200"))

def _cap_images(imgs):
    if not isinstance(imgs, (list, tuple)):
        return []
    uniq = []
    seen = set()
    for u in imgs:
        s = str(u)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
        if len(uniq) >= MAX_IMAGES:
            break
    return uniq

def _trim(s: Optional[str], n: int = MAX_DESC_CHARS) -> str:
    if not s:
        return ""
    ss = str(s)
    return ss if len(ss) <= n else (ss[:n] + "…")

def _parse_jp_location(s: str) -> Dict[str, str]:
    out = {"prefecture": "", "city": "", "neighborhood": ""}
    if not s:
        return out
    m = re.match(r"^(.*?[都道府県])(?:(.*?[市区町村]))?(.*)$", s)
    if m:
        out["prefecture"] = (m.group(1) or "").strip()
        out["city"] = (m.group(2) or "").strip()
        out["neighborhood"] = (m.group(3) or "").strip()
    else:
        out["neighborhood"] = s.strip()
    return out

# Romaji helpers for format 1/2 (e.g., "Yubari-shi, Hokkaido")
_ROMAJI_CITY_RX = re.compile(r"([A-Za-z\-]+(?:-shi|-ku|-cho|-mura))", re.I)
_ROMAJI_PREF_RX = re.compile(
    r"\b(Hokkaido|Tokyo|Osaka|Kyoto|Aichi|Fukuoka|Okinawa|Kagoshima|Hiroshima|Miyagi|Saitama|Chiba|Kanagawa|Hyogo|Nara|Shizuoka|Niigata|Nagano|Gifu|Ibaraki|Tochigi|Gunma|Mie|Okayama|Kumamoto|Oita|Yamagata|Yamanashi|Fukushima|Ishikawa|Toyama|Fukui|Shiga|Wakayama|Tottori|Shimane|Yamaguchi|Kagawa|Tokushima|Ehime|Kochi|Akita|Aomori|Iwate)\b",
    re.I
)

def _parse_romaji_addr(addr: str) -> Dict[str, str]:
    out = {"prefecture": "", "city": "", "neighborhood": ""}
    if not addr:
        return out
    m_city = _ROMAJI_CITY_RX.search(addr or "")
    m_pref = _ROMAJI_PREF_RX.search(addr or "")
    if m_city:
        out["city"] = m_city.group(1)
    if m_pref:
        out["prefecture"] = m_pref.group(1)
    out["neighborhood"] = (addr or "").strip()
    return out

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

def _pick_price_yen(item: Dict[str, Any]) -> Optional[int]:
    for key in ("price_yen_min", "price_yen", "rent_yen", "price", "rent"):
        val = _as_int(item.get(key))
        if val:
            return val
    offers = item.get("offers")
    if isinstance(offers, (list, tuple)) and offers:
        nums = [_as_int(x) for x in offers]
        nums = [x for x in nums if isinstance(x, int) and x > 0]
        if nums:
            return min(nums)
    s = item.get("price_text") or ""
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

def _format_listing_text(item: Dict[str, Any], price_yen: Optional[int]) -> str:
    parts = [
        str(item.get("title") or ""),
        str(item.get("description") or ""),
        str(item.get("location") or ""),
    ]
    if price_yen is not None:
        parts.append(f"Price: ¥{price_yen:,}")
    if item.get("price_text"):
        parts.append(str(item["price_text"]))
    if item.get("details_url"):
        parts.append(str(item["details_url"]))
    return " ".join(p for p in parts if p)

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH does not exist: {path}")
    if path.endswith(".json"):
        with open(path, "rb") as f:
            data = jsonlib.loads(f.read())
        if isinstance(data, dict):
            yield data
        else:
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        return
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = jsonlib.loads(line)
            except Exception as e:
                log.warning("Skipping line %d (invalid JSON): %s", i, e)
                continue
            if not isinstance(obj, dict):
                log.warning("Skipping line %d (not an object)", i)
                continue
            yield obj

def _clean_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            sv = [str(x) for x in v if x is not None]
            if sv:
                out[k] = sv
        else:
            try:
                out[k] = str(v)
            except Exception:
                pass
    return out

# ---------- Normalizers for new record formats ----------

def _normalize_format1(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Village House record format 1"""
    addr = rec.get("address") or ""
    loc = _parse_romaji_addr(addr)
    if not any(loc.values()):
        loc = _parse_jp_location(addr)

    details_url = rec.get("source_url") or ""
    price_yen = _as_int(rec.get("rent"))

    meta = _clean_meta({
        "title": rec.get("title") or "",
        "description": _trim(f"Year built: {rec.get('year_built') or ''} | Floorplan: {rec.get('floorplan') or ''} | Size: {rec.get('size_sqm') or ''}"),
        "price_yen": price_yen,
        "pet_dog": False,
        "neighborhood": loc.get("neighborhood") or "",
        "city": loc.get("city") or "",
        "prefecture": loc.get("prefecture") or "",
        "deal_type": "rent",
        "url": details_url,
        "images": _cap_images(rec.get("image_urls") or []),
        "source": rec.get("source_url") or "",
        "timestamp": rec.get("scraped_at") or dt.datetime.now().isoformat(),
    })

    price_txt = f"¥{price_yen:,}" if price_yen else ""
    text = " ".join(x for x in [
        rec.get("title") or "",
        addr,
        f"Price: {price_txt}" if price_txt else "",
        details_url
    ] if x)

    base_id = details_url or f"{rec.get('listing_id','')}-{addr}"
    _id = md5(base_id.encode("utf-8")).hexdigest()
    return {"id": _id, "text": text, "metadata": meta}

def _normalize_format2(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Leopalace record format 2"""
    addr = rec.get("address") or ""
    if not addr:
        p = (rec.get("prefecture") or "").strip()
        c = (rec.get("city") or "").strip()
        addr = ", ".join(x for x in [c, p] if x)

    loc = _parse_romaji_addr(addr) if addr else {"prefecture": "", "city": "", "neighborhood": ""}
    if not any(loc.values()):
        loc = _parse_jp_location(addr)

    details_url = rec.get("source_url") or ""
    price_yen = _pick_price_yen(rec)

    meta = _clean_meta({
        "title": rec.get("title") or "",
        "description": _trim(f"Floorplan: {rec.get('floorplan') or ''} | Size: {rec.get('size_sqm') or ''}"),
        "price_yen": price_yen,
        "pet_dog": False,
        "neighborhood": loc.get("neighborhood") or "",
        "city": loc.get("city") or "",
        "prefecture": loc.get("prefecture") or "",
        "deal_type": "rent",
        "url": details_url,
        "images": _cap_images(rec.get("image_urls") or []),
        "source": rec.get("source_url") or "",
        "timestamp": rec.get("scraped_at") or dt.datetime.now().isoformat(),
    })

    price_txt = f"¥{price_yen:,}" if price_yen else ""
    text = " ".join(x for x in [
        rec.get("title") or "",
        addr,
        f"Price: {price_txt}" if price_txt else "",
        details_url
    ] if x)

    base_id = details_url or f"{rec.get('property_number','')}-{addr}"
    _id = md5(base_id.encode("utf-8")).hexdigest()
    return {"id": _id, "text": text, "metadata": meta}

# ---------- Seeding ----------

def seed_examples(force_recreate: bool = False) -> int:
    log.info("Seeding from %s", DATA_PATH)

    items = []
    for idx, rec in enumerate(_iter_jsonl(DATA_PATH), 1):
        try:
            if isinstance(rec, dict) and "listing_id" in rec and "address" in rec:
                norm = _normalize_format1(rec)
            elif isinstance(rec, dict) and "offers" in rec and "title" in rec:
                norm = _normalize_format2(rec)
            else:
                # Legacy/other schema
                url = rec.get("details_url") or rec.get("source_url") or ""
                base = url or f"{rec.get('title','')}-{rec.get('location','')}"
                _id = md5(base.encode("utf-8")).hexdigest()
                price_yen = _pick_price_yen(rec)
                loc = _parse_jp_location(rec.get("location") or "")

                meta = _clean_meta({
                    "title": rec.get("title") or "",
                    "description": _trim(rec.get("description") or ""),
                    "price_yen": price_yen,
                    "pet_dog": bool(rec.get("pet_dog", False)),
                    "neighborhood": loc.get("neighborhood") or "",
                    "city": loc.get("city") or "",
                    "prefecture": loc.get("prefecture") or "",
                    "deal_type": rec.get("deal_type") or "rent",
                    "url": url,
                    "images": _cap_images(rec.get("image_urls") or rec.get("images") or []),
                    "source": rec.get("source_url") or "",
                    "timestamp": dt.datetime.now().isoformat(),
                })
                text = _format_listing_text(rec, price_yen)
                norm = {"id": _id, "text": text, "metadata": meta}

            items.append(norm)
        except Exception as e:
            log.warning("Skipping record %d due to error: %s", idx, e)

    if not items:
        log.warning("No items parsed from %s", DATA_PATH)
        return 0

    store = ListingStore(collection="listings_openai_2", force_recreate=force_recreate)
    log.info("Vector store ready (index/namespace may vary by backend).")

    BATCH = int(os.getenv("SEED_BATCH", "64"))
    total = 0
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        inserted = store.upsert(batch)
        total += inserted
        log.info("Upserted %d/%d (added %d this batch)", total, len(items), inserted)

    try:
        cnt = store.count()
        log.info("Vector store count (approx): %s", cnt)
    except Exception as e:
        log.info("Could not fetch count: %s", e)

    try:
        probe = store.search("賃貸 マンション ワンルーム 東京", n=3)
        log.info("Probe results: %d", len(probe))
    except Exception as e:
        log.info("Probe search failed: %s", e)

    return total

if __name__ == "__main__":
    created = seed_examples(force_recreate=os.getenv("FORCE_RECREATE", "false").lower() in ("1","true","yes"))
    print(f"Seeded {created} items from {DATA_PATH}")
