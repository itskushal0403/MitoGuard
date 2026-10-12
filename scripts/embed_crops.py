"""Build the per-case crop bank and embed every crop with each frozen backbone.

Crop bank for a case = all labelled cells (mitoses + imposters) plus every Stage 1 candidate of every
fold model, de-duplicated within 4 px.  Each bank point is embedded once per backbone, so all folds
and all compared systems share identical features.

Output: WORK/bank/bank.pkl  {case_id: (M,2) xy}
        WORK/bank/<backbone>.npy  float16 (sum_M, dim), rows ordered by case_id then bank index
        WORK/bank/offsets.pkl {case_id: (start, end)}
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, image_path, load_annotations, load_cases  # noqa: E402
from mitoguard.features import Backbone  # noqa: E402
from mitoguard.tiling import crop_centered, read_region  # noqa: E402

DEDUP_PX = 4.0


def dedup(xy, radius):
    keep, taken = [], np.zeros(len(xy), bool)
    tree = cKDTree(xy)
    for i in range(len(xy)):
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(xy[i], radius)] = True
    return xy[keep]


def build_bank():
    ann = load_annotations()
    cand_files = sorted(glob.glob(os.path.join(WORK_ROOT, "cands", "*.pkl")))
    cands = [pickle.load(open(f, "rb")) for f in cand_files]
    print("candidate files:", [os.path.basename(f) for f in cand_files], flush=True)
    bank = {}
    for cid in load_cases().case_id:
        pts = [ann[ann.case_id == cid][["x", "y"]].to_numpy(np.float32)]
        pts += [c[cid]["xy"] for c in cands if cid in c]
        xy = np.concatenate(pts) if pts else np.zeros((0, 2), np.float32)
        bank[int(cid)] = dedup(xy, DEDUP_PX).astype(np.float32) if len(xy) else xy
    return bank


class Crops(Dataset):
    def __init__(self, bank, size):
        self.ids, self.bank, self.size = sorted(bank), bank, size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        img = read_region(image_path(cid))
        xy = self.bank[cid]
        crops = np.stack([crop_centered(img, x, y, self.size) for x, y in xy]) if len(xy) else np.zeros((0, self.size, self.size, 3), np.uint8)
        return cid, crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["phikon", "phikon2", "resnet50"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rebuild-bank", action="store_true")
    args = ap.parse_args()
    out = os.path.join(WORK_ROOT, "bank")
    os.makedirs(out, exist_ok=True)
    bank_file = os.path.join(out, "bank.pkl")
    if args.rebuild_bank or not os.path.exists(bank_file):
        bank = build_bank()
        pickle.dump(bank, open(bank_file, "wb"))
    bank = pickle.load(open(bank_file, "rb"))
    total = sum(len(v) for v in bank.values())
    print(f"bank: {total} crops over {len(bank)} cases", flush=True)
    offsets, s = {}, 0
    for cid in sorted(bank):
        offsets[cid] = (s, s + len(bank[cid]))
        s += len(bank[cid])
    pickle.dump(offsets, open(os.path.join(out, "offsets.pkl"), "wb"))

    suffix = "" if args.size == 224 else f"_{args.size}"
    models = {b: Backbone(b) for b in args.backbones}
    feats = {b: np.lib.format.open_memmap(os.path.join(out, f"{b}{suffix}.npy.tmp"), mode="w+", dtype=np.float16,
                                          shape=(total, models[b].dim)) for b in models}
    loader = DataLoader(Crops(bank, args.size), batch_size=None, num_workers=8, prefetch_factor=2)
    t0 = time.time()
    for i, (cid, crops) in enumerate(loader):
        crops = crops.numpy()
        if args.size != 224:
            import cv2
            crops = np.stack([cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA) for c in crops]) if len(crops) else np.zeros((0, 224, 224, 3), np.uint8)
        a, b_ = offsets[int(cid)]
        for name, m in models.items():
            for k in range(0, len(crops), args.batch):
                feats[name][a + k:a + k + len(crops[k:k + args.batch])] = m(crops[k:k + args.batch]).astype(np.float16)
        if i % 25 == 0:
            print(f"[{i + 1}/{len(bank)}] case {int(cid)} crops={len(crops)} {time.time() - t0:.0f}s", flush=True)
    for mm in feats.values():
        mm.flush()
    del feats
    for name in models:
        os.replace(os.path.join(out, f"{name}{suffix}.npy.tmp"), os.path.join(out, f"{name}{suffix}.npy"))


if __name__ == "__main__":
    main()
