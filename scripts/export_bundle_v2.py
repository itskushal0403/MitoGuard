"""Package MitoGuard v2 for inference / the demo (in-domain fold: trained on the official training split).

Contents: both Stage 1 YOLO11 detectors, the LoRA + head weights of the fine-tuned classifier, the Platt
calibrator and stacker fitted on validation candidates, the detection threshold (validation), the
reliability-flag threshold (Youden on validation regions) and the conformal quantile for 90% count
intervals (validation regions).
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
from mitoguard.stage2 import s2_name  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="indomain_all")
    ap.add_argument("--backbone", default="phikon2")
    ap.add_argument("--pool", default="a05")
    ap.add_argument("--variant", default="union")
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints"))
    args = ap.parse_args()
    protocol, name = args.fold.split("_", 1)
    feat = f"ft-{args.backbone}"
    sfx = "" if args.variant == "stain" else f"_{args.variant}"
    system = f"yolo+{feat}@{args.pool}" + ("+stack" if args.stack else "") + sfx
    folds = pd.read_csv(os.path.join(ROOT, "results", "folds.csv"))
    row = folds[(folds.protocol == protocol) & (folds.fold == name) & (folds.system == system)].iloc[0]
    cases = pd.read_csv(os.path.join(ROOT, "results", "cases.csv.gz"))
    val = cases[(cases.protocol == protocol) & (cases.fold == name) & (cases.system == system) & (cases.split == "val")]
    fpr, tpr, thr = roc_curve(large_error_label(val["count"], val["n_gt"], 0.2), val["u_var"])
    flag_thr = float(thr[np.argmax(tpr - fpr)])
    sigma = np.sqrt(val.u_var * val["count"].clip(lower=1))
    scores = np.sort(((val["count"] - val.n_gt).abs() / np.sqrt(sigma ** 2 + 1)).to_numpy())
    k = int(np.ceil((len(scores) + 1) * (1 - args.alpha)))
    conf_q = float(scores[min(k, len(scores)) - 1])
    os.makedirs(args.out, exist_ok=True)
    yolos = []
    for v in ("stain", "nostain"):
        dst = f"yolo_{args.fold}_{v}.pt"
        shutil.copy(os.path.join(WORK_ROOT, "yolo", f"{args.fold}_{v}", "weights", "last.pt"), os.path.join(args.out, dst))
        yolos.append(dst)
    s2 = pickle.load(open(os.path.join(WORK_ROOT, "stage2", s2_name(f"{args.fold}_{args.variant}", feat, args.pool) + ".pkl"), "rb"))
    stacker = None
    if args.stack:
        stacker = pickle.load(open(os.path.join(WORK_ROOT, "stackers", f"{args.fold}__{system}.pkl"), "rb"))
    state = torch.load(os.path.join(WORK_ROOT, "ft", f"{args.fold}__{args.backbone}.pt"), map_location="cpu")
    bundle = {"version": 2, "yolo_weights": yolos, "backbone": args.backbone, "classifier_state": state,
              "lora_r": 16, "zoom": 112, "tta": 8, "platt": s2["platt"], "stacker": stacker, "pool": args.pool,
              "det_thr": float(row.threshold), "flag_thr": flag_thr, "conformal_q": conf_q,
              "provenance": {"fold": args.fold, "system": system, "test_f1": float(row.f1), "val_f1": float(row.val_f1)}}
    torch.save(bundle, os.path.join(args.out, "mitoguard_v2_bundle.pt"))
    print(f"saved v2 bundle: {system} det_thr={row.threshold:.2f} flag_thr={flag_thr:.3f} conformal_q={conf_q:.2f} "
          f"in-domain test F1={row.f1:.3f}")


if __name__ == "__main__":
    main()
