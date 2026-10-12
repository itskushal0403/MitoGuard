"""Cache every crop-bank point as a 224x224 uint8 crop (WORK/bank/crops224.npy, ~150 KB per crop).

Reading 503 large TIFFs is the slow, CPU-bound part of embedding; with the cache, any number of feature
variants (zoom levels, token pooling, TTA) can be embedded on a GPU node with very few CPUs.
Run on a CPU node after embed_crops.py has (re)built bank.pkl.
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, image_path  # noqa: E402
from mitoguard.tiling import crop_centered, read_region  # noqa: E402

SIZE = 224
D = os.path.join(WORK_ROOT, "bank")


def work(cid):
    bank = pickle.load(open(os.path.join(D, "bank.pkl"), "rb"))
    off = pickle.load(open(os.path.join(D, "offsets.pkl"), "rb"))
    a, b = off[cid]
    if a == b:
        return cid
    img = read_region(image_path(cid))
    mm = np.load(os.path.join(D, "crops224.npy.tmp"), mmap_mode="r+")
    mm[a:b] = np.stack([crop_centered(img, x, y, SIZE) for x, y in bank[cid]])
    mm.flush()
    open(os.path.join(D, "crops224.done", str(cid)), "w").close()  # resumable: one marker per finished case
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    off = pickle.load(open(os.path.join(D, "offsets.pkl"), "rb"))
    total = max(b for _, b in off.values())
    tmp = os.path.join(D, "crops224.npy.tmp")
    os.makedirs(os.path.join(D, "crops224.done"), exist_ok=True)
    if not os.path.exists(tmp):
        np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(total, SIZE, SIZE, 3)).flush()
    done = {int(f) for f in os.listdir(os.path.join(D, "crops224.done"))}
    if not done:  # job killed before markers existed: a case is done if its first and last crops are written
        mm = np.load(tmp, mmap_mode="r")
        done = {c for c, (a, b) in off.items() if a == b or (mm[a].any() and mm[b - 1].any())}
    todo = [c for c in sorted(off) if c not in done]
    print(f"{len(done)} cases already cached, {len(todo)} to go", flush=True)
    with Pool(args.workers) as pool:
        for i, cid in enumerate(pool.imap_unordered(work, todo)):
            if i % 50 == 0:
                print(f"[{i + 1}/{len(off)}] case {cid}", flush=True)
    os.replace(tmp, os.path.join(D, "crops224.npy"))
    import shutil
    shutil.rmtree(os.path.join(D, "crops224.done"))
    print("cached", total, "crops")


if __name__ == "__main__":
    main()
