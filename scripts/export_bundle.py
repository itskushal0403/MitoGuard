"""Package a trained pipeline (Stage 1 weights + Stage 2 filter + thresholds) for the demo / inference.

The demo uses the in-domain fold (trained on the official MIDOG++ training split of all 7 domains).
Detection threshold: tuned on that fold's validation cases (from results/folds.csv).
Flag threshold: the u_var cut-off that maximises Youden's J for ">20% count error" on validation cases.
"""
import argparse
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT  # noqa: E402
from mitoguard.metrics import large_error_label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="indomain_all")
    ap.add_argument("--backbone", default="phikon2-z112-ctr", help="feature spec of the Stage 2 filter")
    ap.add_argument("--pool", default="m05i10")
    ap.add_argument("--mode", default="stack", choices=["filter", "stack"])
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints"))
    args = ap.parse_args()
    run = f"{args.fold}_stain"
    protocol, name = args.fold.split("_", 1)
    from mitoguard.stage2 import s2_name
    tag_ = args.backbone if args.pool == "m05" else f"{args.backbone}@{args.pool}"
    system = f"yolo+{tag_}" + ("+stack" if args.mode == "stack" else "")
    folds = pd.read_csv(os.path.join(ROOT, "results", "folds.csv"))
    row = folds[(folds.protocol == protocol) & (folds.fold == name) & (folds.system == system)].iloc[0]
    cases = pd.read_csv(os.path.join(ROOT, "results", "cases.csv.gz"))
    val = cases[(cases.protocol == protocol) & (cases.fold == name) & (cases.system == system) & (cases.split == "val")]
    y = large_error_label(val["count"], val["n_gt"], 0.2)
    fpr, tpr, thr = roc_curve(y, val["u_var"])
    flag_thr = float(thr[np.argmax(tpr - fpr)])
    os.makedirs(args.out, exist_ok=True)
    yolo_dst = os.path.join(args.out, f"yolo_{run}.pt")
    shutil.copy(os.path.join(WORK_ROOT, "yolo", run, "weights", "last.pt"), yolo_dst)
    s2 = pickle.load(open(os.path.join(WORK_ROOT, "stage2", s2_name(run, args.backbone, args.pool) + ".pkl"), "rb"))
    stacker = None
    if args.mode == "stack":
        stacker = pickle.load(open(os.path.join(WORK_ROOT, "stackers", f"{args.fold}__{system}.pkl"), "rb"))
    bundle = {"yolo_weights": os.path.basename(yolo_dst),  # resolved relative to the bundle file
              "backbone": args.backbone, "filter": s2["filter"], "pool": args.pool, "stacker": stacker,
              "det_thr": float(row.threshold), "flag_thr": flag_thr,
              "provenance": {"fold": args.fold, "system": system, "test_f1": float(row.f1), "val_f1": float(row.val_f1)}}
    torch.save(bundle, os.path.join(args.out, "mitoguard_bundle.pt"))
    print(f"saved bundle: det_thr={row.threshold:.2f} flag_thr={flag_thr:.3f} test F1={row.f1:.3f}")


if __name__ == "__main__":
    main()
