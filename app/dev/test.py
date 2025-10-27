# check_ids.py
import os, json, hashlib
from pinecone import Pinecone

DATA_PATH = os.getenv("DATA_PATH", "jp_listings.jsonl")
N = int(os.getenv("CHECK_N", "10"))
IDX_NAME = os.getenv("PINECONE_INDEX", "listing")
NAMESPACE = os.getenv("SEED_NAMESPACE", "listings_openai")

def md5_of(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def first_n_ids(path, n):
    ids = []
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            if i > n:
                break
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            url = obj.get("details_url") or ""
            if url:
                _id = md5_of(url)
            else:
                base = f"{obj.get('title','')}-{obj.get('location','')}"
                _id = md5_of(base)
            ids.append((_id, obj.get("title") or "", obj.get("details_url") or ""))
    return ids

def main():
    pc = Pinecone(api_key="pcsk_4fZ4oZ_hPseoG8T8iggW7qmNZXJXfUGXSQzDuS748ozxkGEgLVDYJ3XXgFNdUvjLRJHzh")

    idx = pc.Index(IDX_NAME)
    ids = first_n_ids(DATA_PATH, N)
    id_list = [i[0] for i in ids]
    print("Checking IDs (first %d items):" % len(id_list))
    for _id, title, url in ids:
        print("-", _id, "|", title[:60].replace("\n"," "), "|", url)
    resp = idx.fetch(ids=id_list, namespace=NAMESPACE)
    # normalize response
    present = set()
    if hasattr(resp, "to_dict"):
        d = resp.to_dict()
        present = set(d.get("vectors", {}).keys())
    elif hasattr(resp, "vectors"):
        present = set(dict(resp.vectors).keys())
    else:
        try:
            d = dict(resp)
            present = set(d.get("vectors", {}).keys())
        except Exception:
            pass
    print("\nPresent IDs:", present)
    missing = set(id_list) - present
    print("Missing IDs (not in index):", list(missing))

if __name__ == "__main__":
    main()
