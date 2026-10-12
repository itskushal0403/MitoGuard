"""Download MIDOG++ (images + annotations) from figshare collection 6615571.

Usage: python scripts/download_midogpp.py --out /path/to/midogpp [--workers 8]
Resumable: skips files whose size already matches.
"""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.figshare.com/v2"
COLLECTION = 6615571
ANNOTATIONS = {"MIDOG++.json": 41265615, "datasets_xvalidation.csv": 41332611}


def get_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise


def list_files():
    arts, page = [], 1
    while True:
        batch = get_json(f"{API}/collections/{COLLECTION}/articles?page={page}&page_size=1000")
        if not batch:
            break
        arts += batch
        page += 1
    files = []
    with ThreadPoolExecutor(16) as ex:
        for meta in ex.map(lambda a: get_json(f"{API}/articles/{a['id']}"), arts):
            files += [(f["name"], f["size"], f["download_url"]) for f in meta.get("files", [])]
    return files


def download(name, size, url, dest):
    path = os.path.join(dest, name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        return name, "skip"
    tmp = path + ".part"
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) == size:
                os.replace(tmp, path)
                return name, "ok"
        except Exception:
            pass
    return name, "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for name, fid in ANNOTATIONS.items():
        urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{fid}", os.path.join(args.out, name))
    files = [f for f in list_files() if f[0].endswith(".tiff")]
    print(f"{len(files)} images, {sum(f[1] for f in files) / 1e9:.1f} GB", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(download, n, s, u, img_dir) for n, s, u in files]
        for i, fut in enumerate(as_completed(futs)):
            name, status = fut.result()
            print(f"[{i + 1}/{len(files)}] {name} {status}", flush=True)


if __name__ == "__main__":
    main()
