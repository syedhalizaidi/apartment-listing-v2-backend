#!/usr/bin/env python3
# suumo_jp_scraping.py  (fixed)
"""
Suumo scraper with DB integration (fixed):
 - added 'scraped_at' to CSV header
 - convert scraped_at to ISO string for CSV
 - more robust import of app.services.db_utils (tries multiple fallbacks)
 - use timezone-aware datetime for scraped_at
"""

import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
from datetime import datetime, timezone
import logging
import sys
import importlib

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# ----------------- CONFIG -----------------
RESULTS_URL = "https://suumo.jp/chintai/nagasaki/new/"
START_PAGE = 0
HEADLESS = True
PAGE_LOAD_TIMEOUT = 60
SCROLL_PAUSES = 5
SCROLL_SLEEP = 2.0
REQUEST_DELAY_PER_CARD = 0.15
MAX_PAGINATED_PAGES = 2304

RETRY_ON_EMPTY = 4
RETRY_SLEEP = 2.0
DO_HARD_REFRESH = True

OUT_CSV = "suumo_listings.csv"
OUT_JSON = "suumo_listings.json"
OUT_JSONL = "suumo_listings.jsonl"

DB_BATCH_SIZE = 200
# ------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("suumo_scraper")


# ---- Try to import db_utils robustly ----
DB_ENABLED = False
insert_listings = None
init_db = None

def _try_import_db_utils():
    global DB_ENABLED, insert_listings, init_db
    tried = []
    # candidate module names to try
    candidates = [
        "app.services.db_utils",
        "services.db_utils",
        "db_utils",
        "app.db_utils",
    ]
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
            insert_listings = getattr(m, "insert_listings", None)
            init_db = getattr(m, "init_db", None)
            if insert_listings:
                DB_ENABLED = True
                logger.info("Imported %s for DB writes.", mod)
                return
        except Exception as e:
            tried.append((mod, str(e)))
    # fallback: try to add project root to sys.path (if running script inside repo)
    try:
        # assume project structure: <repo-root>/app/services/db_utils.py
        this = Path(__file__).resolve()
        root = this.parents[2]  # adjust if file is at repo_root/... ; parents[2] => repo root if file is scraping/<file>
        sys.path.insert(0, str(root))
        for mod in candidates:
            try:
                m = importlib.import_module(mod)
                insert_listings = getattr(m, "insert_listings", None)
                init_db = getattr(m, "init_db", None)
                if insert_listings:
                    DB_ENABLED = True
                    logger.info("Imported %s for DB writes after adding %s to sys.path.", mod, root)
                    return
            except Exception as e:
                tried.append((mod + " (after adding root)", str(e)))
    except Exception as e:
        tried.append(("sys.path adjustment", str(e)))

    # if we reach here, import failed
    logger.warning("Could not import app.services.db_utils (attempted: %s). DB writes disabled.", tried)


_try_import_db_utils()


# ---------- Selenium helpers ----------
def setup_driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1366,900")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    d = webdriver.Chrome(service=service, options=opts)
    d.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return d

def scroll_to_load(driver, times=SCROLL_PAUSES, sleep=SCROLL_SLEEP):
    last_h = 0
    for _ in range(times):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep)
        new_h = driver.execute_script("return document.body.scrollHeight;")
        if new_h == last_h:
            break
        last_h = new_h

def text_clean(s): return re.sub(r"\s+", " ", s or "").strip()

def extract_images_from_card(card_el):
    urls = set()
    for img in card_el.find_elements(By.TAG_NAME, "img"):
        for attr in ("src", "data-src", "data-original", "data-img", "data-lazy"):
            val = img.get_attribute(attr)
            if val and val.startswith("http"):
                urls.add(val)
    for div in card_el.find_elements(By.XPATH, ".//*[contains(@style,'background-image')]"):
        style = div.get_attribute("style") or ""
        m = re.search(r'url\(["\']?(https?[^"\')]+)', style)
        if m:
            urls.add(m.group(1))
    return list(urls)

def extract_text_candidates(card_html):
    soup = BeautifulSoup(card_html, "html.parser")
    title = ""
    for sel in ["h2", "h3", ".property_ttl", ".cassetteitem_content-title", ".building-header__title"]:
        el = soup.select_one(sel)
        if el and text_clean(el.get_text()):
            title = text_clean(el.get_text()); break
    if not title:
        el = soup.find(["strong", "b"])
        if el:
            t = text_clean(el.get_text())
            if t: title = t

    location = ""
    addr_like = []
    for el in soup.find_all(string=True):
        t = text_clean(el)
        if t and re.search(r"(都|道|府|県|区|市|町|村)", t):
            addr_like.append(t)
    if addr_like:
        location = addr_like[0]

    desc = ""
    for sel in [".property_body", ".cassetteitem_detail", ".property_content", ".building-content", ".detailbody"]:
        el = soup.select_one(sel)
        if el:
            desc = text_clean(el.get_text(separator=" ")); break
    if not desc:
        desc = text_clean(soup.get_text(separator=" "))
    for b in ["詳細を見る", "お問い合わせ", "お気に入り", "動画", "パノラマ", "まとめて問い合わせ", "おすすめ順", "新着順"]:
        desc = desc.replace(b, " ")
    desc = re.sub(r"\s{2,}", " ", desc).strip()
    if len(desc) > 600:
        desc = desc[:580].rstrip() + "..."
    return title, location, desc

# ---------- presence / end checks ----------
def listings_present(driver):
    if driver.find_elements(By.XPATH, "//a[contains(., '詳細を見る')]"): return True
    for sel in [".cassetteitem", ".property_unit", ".property", ".content-inner"]:
        if driver.find_elements(By.CSS_SELECTOR, sel): return True
    return False

def is_no_results_or_end(driver):
    for p in ["該当する物件はありません", "条件に一致する物件は見つかりません"]:
        if driver.find_elements(By.XPATH, f"//*[contains(., '{p}')]"): return True
    return not listings_present(driver)

def wait_for_listings_or_end(driver, timeout=25):
    try:
        WebDriverWait(driver, timeout).until(lambda d: listings_present(d) or is_no_results_or_end(d))
        return "listings" if listings_present(driver) else "end"
    except TimeoutException:
        return "end"

def get_listing_cards(driver):
    links = driver.find_elements(By.XPATH, "//a[contains(., '詳細を見る')]")
    cards, seen = [], set()
    for link in links:
        card = None
        for tag in ["article", "section", "li", "div"]:
            try:
                card = link.find_element(By.XPATH, f"./ancestor::{tag}[1]"); break
            except Exception:
                continue
        if card and card.id not in seen:
            seen.add(card.id); cards.append(card)
    return cards

# ---------- url / paging ----------
def set_query_param(url, key, value):
    p = urlparse(url); q = parse_qs(p.query, keep_blank_values=True)
    q[str(key)] = [str(value)]
    new_q = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_q, p.fragment))

def get_page_num(url):
    p = urlparse(url); q = parse_qs(p.query)
    try: return int(q.get("page", [1])[0])
    except Exception: return 1

def next_results_url(current_url):
    return set_query_param(current_url, "page", get_page_num(current_url) + 1)

# ---------- writers (CSV now includes scraped_at) ----------
def csv_writer_init(csv_path):
    exists = Path(csv_path).exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    # added 'scraped_at' to fieldnames
    writer = csv.DictWriter(f, fieldnames=["title", "location", "image_urls", "description", "details_url", "source_url", "scraped_at"])
    if not exists: writer.writeheader()
    return f, writer

def append_rows_to_csv(writer, rows):
    for r in rows:
        r2 = r.copy()
        # image_urls -> join
        r2["image_urls"] = " | ".join(r2.get("image_urls") or [])
        sa = r2.get("scraped_at")
        if isinstance(sa, datetime):
            # write ISO string
            r2["scraped_at"] = sa.astimezone(timezone.utc).isoformat()
        else:
            r2["scraped_at"] = str(sa or "")
        writer.writerow(r2)

def append_rows_to_jsonl(jsonl_path, rows):
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for r in rows:
            # convert scraped_at to iso if datetime
            rec = dict(r)
            sa = rec.get("scraped_at")
            if isinstance(sa, datetime):
                rec["scraped_at"] = sa.astimezone(timezone.utc).isoformat()
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- page scrape ----------
def scrape_current_page(driver, source_url, seen_detail_urls):
    scroll_to_load(driver)
    status = wait_for_listings_or_end(driver)
    if status == "end": return []
    cards = get_listing_cards(driver)
    page_rows = []
    for card in cards:
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", card)
        except Exception: pass
        time.sleep(0.15)
        card_html = card.get_attribute("outerHTML") or ""
        title, location, desc = extract_text_candidates(card_html)
        image_urls = extract_images_from_card(card)
        details_url = ""
        try:
            a = card.find_element(By.XPATH, ".//a[contains(., '詳細を見る')]")
            href = a.get_attribute("href")
            if href: details_url = href
        except Exception: pass
        if details_url and details_url in seen_detail_urls: continue
        if details_url: seen_detail_urls.add(details_url)
        page_rows.append({
            "title": title,
            "location": location,
            "image_urls": image_urls,
            "description": desc,
            "details_url": details_url,
            "source_url": source_url,
            # timezone-aware UTC
            "scraped_at": datetime.now(timezone.utc)
        })
        time.sleep(REQUEST_DELAY_PER_CARD)
    return page_rows

# -------- retry helper --------
def ensure_listings_or_retry(driver, target_url, retries=RETRY_ON_EMPTY, sleep_s=RETRY_SLEEP):
    attempt = 0
    while attempt <= retries:
        if attempt == 0:
            if driver.current_url != target_url:
                driver.get(target_url)
        else:
            if DO_HARD_REFRESH:
                try:
                    driver.refresh()
                except Exception:
                    driver.get(target_url)
            else:
                driver.get(target_url)

        status = wait_for_listings_or_end(driver)
        if status == "listings":
            return True

        attempt += 1
        if attempt <= retries:
            logger.info("[retry] page looked empty, retry %d/%d -> %s", attempt, retries, target_url)
            time.sleep(sleep_s)

    return False

# -------- main scraper with DB integration --------
def scrape_suumo_results(url=RESULTS_URL, db_batch_size=DB_BATCH_SIZE):
    if DB_ENABLED and init_db:
        try:
            init_db()
        except Exception as e:
            logger.warning("init_db() failed: %s (continuing)", e)

    driver = setup_driver()
    results_all = []
    seen_detail_urls = set()
    csv_f, csv_w = csv_writer_init(OUT_CSV)
    pending_db_rows = []

    try:
        start_url = set_query_param(url, "page", START_PAGE) if START_PAGE and START_PAGE > 1 else url
        ok = ensure_listings_or_retry(driver, start_url)
        if not ok:
            logger.info("[done] no listings on start page even after retries (%s)", start_url)
            return results_all

        page_idx = get_page_num(driver.current_url)
        pages_done = 0

        while True:
            cur_url = driver.current_url
            logger.info("\n[page] %d: %s", page_idx, cur_url)

            rows = scrape_current_page(driver, cur_url, seen_detail_urls)
            logger.info("[info] collected %d listings on page %d", len(rows), page_idx)

            if rows:
                append_rows_to_csv(csv_w, rows)
                append_rows_to_jsonl(OUT_JSONL, rows)
                results_all.extend(rows)

            if DB_ENABLED and insert_listings and rows:
                db_records = []
                for r in rows:
                    rec = {
                        "title": r.get("title"),
                        "location": r.get("location"),
                        "image_urls": r.get("image_urls"),
                        "details_url": r.get("details_url"),
                        "source_url": r.get("source_url"),
                        "scraped_at": r.get("scraped_at"),
                        # other keys will go into extras
                        "description": r.get("description")
                    }
                    db_records.append(rec)

                pending_db_rows.extend(db_records)

                if len(pending_db_rows) >= db_batch_size:
                    try:
                        inserted = insert_listings(pending_db_rows, batch_size=db_batch_size)
                        logger.info("[db] inserted/upserted %d rows", inserted)
                    except Exception as e:
                        logger.error("[db] insert_listings failed: %s", e)
                    pending_db_rows = []

            pages_done += 1
            if pages_done >= MAX_PAGINATED_PAGES:
                logger.info("[stop] reached MAX_PAGINATED_PAGES")
                break

            nxt = next_results_url(cur_url)
            ok = ensure_listings_or_retry(driver, nxt)
            if not ok:
                logger.info("[done] reached the end (no listings on next page, after retries)")
                break

            if driver.current_url == cur_url:
                logger.info("[done] pagination did not advance (same URL) — stopping")
                break

            page_idx += 1

    finally:
        if DB_ENABLED and insert_listings and pending_db_rows:
            try:
                inserted = insert_listings(pending_db_rows, batch_size=db_batch_size)
                logger.info("[db] final inserted/upserted %d rows", inserted)
            except Exception as e:
                logger.error("[db] final insert_listings failed: %s", e)
        try: csv_f.close()
        except Exception: pass
        try: driver.quit()
        except Exception: pass

    # write combined JSON with ISO timestamps
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        out_dump = []
        for r in results_all:
            r2 = dict(r)
            sa = r2.get("scraped_at")
            if isinstance(sa, datetime):
                r2["scraped_at"] = sa.astimezone(timezone.utc).isoformat()
            out_dump.append(r2)
        json.dump(out_dump, f, ensure_ascii=False, indent=2)

    logger.info("Streaming JSONL (per page): %s", OUT_JSONL)
    logger.info("Streaming CSV (per page):  %s", OUT_CSV)
    logger.info("Combined JSON (this run):  %s  | items: %d", OUT_JSON, len(results_all))
    return results_all

if __name__ == "__main__":
    scrape_suumo_results(RESULTS_URL)
