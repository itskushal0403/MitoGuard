"""Run every trained Stage 1 model over every MIDOG++ case and cache the candidates.

Each region is read from disk once and passed through all fold models, which is far cheaper than
one pass per model.  Output: WORK/cands/<run>.pkl = {case_id: {xy, p_mit, p_imp}}.
Detections on training cases provide Stage 2 hard negatives; on validation cases they drive
threshold tuning; on held-out cases they are the test predictions.
"""
import argparse
import glob
import os
import pickle
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, image_path, load_cases  # noqa: E402
from mitoguard.detector import CandidateDetector  # noqa: E402
from mitoguard.tiling import read_region  # noqa: E402


class Regions(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], read_region(image_path(self.ids[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="YOLO run names (default: all finished runs)")
    ap.add_argument("--weights", default="last.pt")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(WORK_ROOT, "cands")
    os.makedirs(out_dir, exist_ok=True)
    # finished = DONE marker without a PENDING marker (PENDING = queued/claimed in slurm/queue_worker.sh)
    runs = args.runs or sorted(os.path.basename(os.path.dirname(p))
                               for p in glob.glob(os.path.join(WORK_ROOT, "yolo", "*", "DONE"))
                               if not os.path.exists(os.path.join(os.path.dirname(p), "PENDING")))
    runs = [r for r in runs if args.overwrite or not os.path.exists(os.path.join(out_dir, r + ".pkl"))]
    print("runs:", runs, flush=True)
    if not runs:
        return
    dets = {r: CandidateDetector(os.path.join(WORK_ROOT, "yolo", r, "weights", args.weights)) for r in runs}
    results = {r: {} for r in runs}
    ids = load_cases().case_id.tolist()
    loader = DataLoader(Regions(ids), batch_size=None, num_workers=8, prefetch_factor=2)
    t0 = time.time()
    for i, (cid, img) in enumerate(loader):
        img = img.numpy() if torch.is_tensor(img) else img
        for r, d in dets.items():
            results[r][int(cid)] = d.detect(img)
        if i % 25 == 0:
            n = {r: len(results[r][int(cid)]["xy"]) for r in list(runs)[:3]}
            print(f"[{i + 1}/{len(ids)}] case {cid} {time.time() - t0:.0f}s cands={n}", flush=True)
    for r in runs:
        with open(os.path.join(out_dir, r + ".pkl"), "wb") as f:
            pickle.dump(results[r], f)


if __name__ == "__main__":
    main()
