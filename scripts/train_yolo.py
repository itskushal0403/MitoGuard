"""Stage 1: fine-tune a YOLO candidate detector for one cross-validation fold.

Two classes: mitotic figure (0) and imposter (1).  Training the imposter class explicitly
forces the detector to look at the look-alikes instead of treating them as background.

Example:
    python scripts/train_yolo.py --fold loto_HBC --model yolo11s.pt --epochs 40
    python scripts/train_yolo.py --fold loto_HBC --no-stain-aug   # ablation
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, get_fold  # noqa: E402
from mitoguard.stain import HEDJitterTransform  # noqa: E402


def write_list(case_ids, path, empty_frac, seed=0):
    tiles = pd.read_csv(os.path.join(WORK_ROOT, "tiles", "tiles.csv"))
    tiles = tiles[tiles.case_id.isin(case_ids)]
    labelled = tiles[(tiles.n_mitosis + tiles.n_imposter) > 0]
    empty = tiles[((tiles.n_mitosis + tiles.n_imposter) == 0) & (tiles.tissue > 0.25)]
    empty = empty.sample(frac=empty_frac, random_state=seed)
    sel = pd.concat([labelled, empty])
    img_dir = os.path.join(WORK_ROOT, "tiles", "images")
    with open(path, "w") as f:
        f.write("\n".join(os.path.join(img_dir, n + ".jpg") for n in sel.name))
    return len(labelled), len(empty)


def patch_stain_aug():
    from ultralytics.data import dataset as uds

    orig = uds.YOLODataset.build_transforms

    def build_transforms(self, hyp):
        t = orig(self, hyp)
        if self.augment:
            t.insert(-1, HEDJitterTransform(p=0.8, sigma=0.05, bias=0.05))
        return t

    uds.YOLODataset.build_transforms = build_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--empty-frac", type=float, default=0.3)
    ap.add_argument("--no-stain-aug", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0, help="subset of training tiles (smoke tests)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    fold = get_fold(args.fold)
    variant = "nostain" if args.no_stain_aug else "stain"
    name = args.name or f"{fold.tag}_{variant}"
    run_dir = os.path.join(WORK_ROOT, "yolo")
    os.makedirs(os.path.join(run_dir, "lists"), exist_ok=True)
    tr_list = os.path.join(run_dir, "lists", f"{name}_train.txt")
    va_list = os.path.join(run_dir, "lists", f"{name}_val.txt")
    print("train tiles (labelled, empty):", write_list(fold.train, tr_list, args.empty_frac))
    print("val tiles (labelled, empty):", write_list(fold.val, va_list, args.empty_frac))
    data_yaml = os.path.join(run_dir, "lists", f"{name}.yaml")
    yaml.safe_dump({"train": tr_list, "val": va_list, "names": {0: "mitotic_figure", 1: "imposter"}}, open(data_yaml, "w"))

    if args.no_stain_aug:
        colour = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    else:
        patch_stain_aug()
        colour = dict(hsv_h=0.015, hsv_s=0.3, hsv_v=0.2)

    from ultralytics import YOLO

    weights_dir = os.path.join(WORK_ROOT, "weights")  # pretrained checkpoints are downloaded here, not into the repo
    os.makedirs(weights_dir, exist_ok=True)
    os.chdir(weights_dir)
    model = YOLO(args.model)
    model.train(
        data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        project=run_dir, name=name, exist_ok=True, seed=0, deterministic=False, cos_lr=True,
        optimizer="AdamW", lr0=1e-3, weight_decay=5e-4, warmup_epochs=2, close_mosaic=5,
        degrees=0.0, translate=0.1, scale=0.2, shear=0.0, perspective=0.0, flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.0, val=False, plots=True, amp=True, fraction=args.fraction, cache=False,
        **colour,
    )


if __name__ == "__main__":
    main()
