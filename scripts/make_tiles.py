"""Extract 512x512 training tiles + YOLO labels for every MIDOG++ case, and record pixel spacing.

Classes: 0 = mitotic figure, 1 = imposter (hard-negative look-alike).
Tiles are written once; each fold selects its tiles through text lists (see make_yolo_lists.py).
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import DATA_ROOT, WORK_ROOT, image_path, load_annotations, load_cases  # noqa: E402
from mitoguard.tiling import grid, read_mpp, read_region  # noqa: E402

TILE, STRIDE, BOX = 512, 448, 50


def process(case_id):
    out_img = os.path.join(WORK_ROOT, "tiles", "images")
    out_lbl = os.path.join(WORK_ROOT, "tiles", "labels")
    path = image_path(case_id)
    img = read_region(path)
    mpp = read_mpp(path)
    ann = load_annotations()
    cells = ann[ann.case_id == case_id][["x", "y", "label"]].to_numpy()
    h, w = img.shape[:2]
    rows = []
    for x0, y0 in grid(h, w, TILE, STRIDE):
        name = f"{case_id:03d}_{x0:05d}_{y0:05d}"
        inside = cells[(cells[:, 0] >= x0) & (cells[:, 0] < x0 + TILE) & (cells[:, 1] >= y0) & (cells[:, 1] < y0 + TILE)]
        lines = []
        for cx, cy, lab in inside:
            bx0, by0 = max(cx - BOX / 2, x0), max(cy - BOX / 2, y0)
            bx1, by1 = min(cx + BOX / 2, x0 + TILE), min(cy + BOX / 2, y0 + TILE)
            lines.append(f"{int(lab) - 1} {((bx0 + bx1) / 2 - x0) / TILE:.6f} {((by0 + by1) / 2 - y0) / TILE:.6f} "
                         f"{(bx1 - bx0) / TILE:.6f} {(by1 - by0) / TILE:.6f}")
        tile = img[y0:y0 + TILE, x0:x0 + TILE]
        # skip near-empty background (white glass)
        tissue = float((tile.mean(axis=2) < 220).mean())
        img_file, lbl_file = os.path.join(out_img, name + ".jpg"), os.path.join(out_lbl, name + ".txt")
        if not os.path.exists(lbl_file):  # resumable: label is written after the image
            cv2.imwrite(img_file, tile[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_file, "w") as f:
                f.write("\n".join(lines))
        rows.append((case_id, name, x0, y0, int((inside[:, 2] == 1).sum()) if len(inside) else 0,
                     int((inside[:, 2] == 2).sum()) if len(inside) else 0, tissue))
    return case_id, mpp, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cases", type=int, nargs="*", default=None, help="subset of case ids (smoke tests)")
    args = ap.parse_args()
    os.makedirs(os.path.join(WORK_ROOT, "tiles", "images"), exist_ok=True)
    os.makedirs(os.path.join(WORK_ROOT, "tiles", "labels"), exist_ok=True)
    os.makedirs(os.path.join(DATA_ROOT, "meta"), exist_ok=True)
    case_ids = sorted(load_cases().case_id.tolist())  # the 503 MIDOG++ cases (JSON also lists 50 unlabelled ids)
    if args.cases:
        case_ids = [c for c in case_ids if c in set(args.cases)]
    spacing, tiles = [], []
    with Pool(args.workers) as pool:
        for i, (cid, mpp, rows) in enumerate(pool.imap_unordered(process, case_ids)):
            spacing.append((cid, mpp))
            tiles += rows
            print(f"[{i + 1}/{len(case_ids)}] case {cid} mpp={mpp:.4f} tiles={len(rows)}", flush=True)
    pd.DataFrame(spacing, columns=["case_id", "mpp"]).sort_values("case_id").to_csv(
        os.path.join(DATA_ROOT, "meta", "spacing.csv"), index=False)
    pd.DataFrame(tiles, columns=["case_id", "name", "x0", "y0", "n_mitosis", "n_imposter", "tissue"]).to_csv(
        os.path.join(WORK_ROOT, "tiles", "tiles.csv"), index=False)


if __name__ == "__main__":
    main()
