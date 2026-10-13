"""External validation on TUPAC16 (auxiliary mitosis set, 73 breast cancer cases; never seen in training).

Step 1 (GPU, --predict): run YOLO11, MitoGuard v1 and MitoGuard v2 (all trained on the official MIDOG++
  training split only; thresholds fixed on MIDOG++ validation, no TUPAC16 tuning) on every TUPAC16 image
  and cache the outputs.
Step 2 (CPU, --score): match against three label sets with the official MIDOG criterion (7.5 um radius,
  30 px at 0.25 um/px, grand-challenge-metrics score_detection), pooled F1, case-level bootstrap 95% CIs:
    original     original TUPAC16 ground truth (1,552 mitoses)
    AL-manual    Bertram et al. 2020 alternative labels, manual consensus (1,898)
    AL-augmented Bertram et al. 2020 alternative labels incl. model-assisted, expert-confirmed (1,999)
"""
import argparse
import glob
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUPAC = os.environ.get("TUPAC_ROOT", "/scratch/m25csa016/tupac16")
RADIUS = 30.0  # 7.5 um at 0.25 um/px


def list_images():
    imgs = sorted(glob.glob(os.path.join(TUPAC, "images", "**", "*.tif"), recursive=True))
    return [(os.path.basename(os.path.dirname(p)), os.path.splitext(os.path.basename(p))[0], p) for p in imgs]


def read_rgb(path):
    import tifffile
    a = tifffile.imread(path)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    return np.ascontiguousarray(a[..., :3])


def predict(out, shard, nshards):
    import torch
    from mitoguard.detector import CandidateDetector
    from mitoguard.pipeline import MitoGuard
    from mitoguard.pipeline_v2 import MitoGuardV2

    ck = os.path.join(ROOT, "checkpoints")
    v2 = MitoGuardV2(os.path.join(ck, "mitoguard_v2_bundle.pt"))
    v1 = MitoGuard(os.path.join(ck, "mitoguard_bundle.pt"))
    yolo = CandidateDetector(os.path.join(ck, "yolo_indomain_all_stain.pt"))
    res = {}
    images = list_images()[shard::nshards]
    for k, (case, name, path) in enumerate(images):
        img = read_rgb(path)
        r2 = v2(img)
        r1 = v1(img)
        y = yolo.detect(img)
        res[(case, name)] = {"shape": img.shape[:2],
                             "v2": {"xy": r2.xy, "p": r2.p_final, "count": r2.count, "interval": r2.interval},
                             "v1": {"xy": r1.xy, "p": r1.p_filter},
                             "yolo": {"xy": y["xy"], "p": y["p_mit"]}}
        print(f"[{k + 1}/{len(images)}] {case}/{name} {img.shape} v2 count={r2.count}", flush=True)
        torch.cuda.empty_cache()
    pickle.dump(res, open(out, "wb"))


def load_labels(images):
    """{label_set: {(case, name): (N,2) xy}} in per-image (x, y) pixel coordinates."""
    sets = {"original": {}, "AL-augmented": {}, "AL-manual": {}}
    al = os.path.join(TUPAC, "labels", "TUPAC16_AlternativeLabels")
    for case, name, path in images:
        for key, base in (("original", os.path.join(TUPAC, "labels", "original", "mitoses_ground_truth")),
                          ("AL-augmented", os.path.join(al, "TUPAC_AL", "mitoses_ground_truth"))):
            f = os.path.join(base, case, name + ".csv")
            rows = [l.strip().split(",") for l in open(f)] if os.path.exists(f) else []
            sets[key][(case, name)] = np.array([[float(c), float(r)] for r, c in rows if r], np.float32).reshape(-1, 2)
    # manual alternative labels live in stitched coordinates: map back to tiles with the stitching offsets
    stitch = pickle.load(open(os.path.join(al, "TUPACStitchInformation.p"), "rb"))
    db = sqlite3.connect(os.path.join(al, "TUPAC_alternativeLabels_training.sqlite"))
    mit = [u for u, n in db.execute("select uid, name from Classes") if n.lower().startswith("mitos")][0]
    slides = dict(db.execute("select uid, filename from Slides"))
    sizes = {(c, n): None for c, n, _ in images}
    import tifffile
    for c, n, p in images:
        with tifffile.TiffFile(p) as t:
            s = t.pages[0].shape
            sizes[(c, n)] = (s[1], s[0]) if s[-1] in (3, 4) else (s[2], s[1])
    for (c, n) in sizes:
        sets["AL-manual"][(c, n)] = []
    q = ("select a.slide, c.coordinateX, c.coordinateY from Annotations a join Annotations_coordinates c "
         "on a.uid = c.annoId where a.deleted = 0 and a.agreedClass = ?")
    for slide, x, y in db.execute(q, (mit,)):
        case = f"{int(os.path.basename(slides[slide]).split('_')[0].split('.')[0]):02d}"
        # cases 1-23 were stitched from 2000 px patches; cases 24-73 are single regions (offset 0)
        tiles = stitch.get(int(case), {"01.tif": [0, 0]})
        for tif, (ox, oy) in tiles.items():
            n = os.path.splitext(tif)[0]
            if (case, n) not in sizes:
                continue
            w, h = sizes[(case, n)]
            if ox <= x < ox + w and oy <= y < oy + h:
                sets["AL-manual"][(case, n)].append((x - ox, y - oy))
                break
    sets["AL-manual"] = {k: np.array(v, np.float32).reshape(-1, 2) for k, v in sets["AL-manual"].items()}
    return sets


def score(pred_files, out_md, thresholds):
    from grand_challenge_metrics.scorers import score_detection
    res = {}
    for f in pred_files:
        res.update(pickle.load(open(f, "rb")))
    images = [(c, n, None) for c, n in sorted(res)]
    labels = load_labels([(c, n, glob.glob(os.path.join(TUPAC, "images", "**", c, n + ".tif"), recursive=True)[0])
                          for c, n, _ in images])
    rows, per_case = [], []
    for lab, gt in labels.items():
        total = sum(len(v) for v in gt.values())
        for system, thr in thresholds.items():
            tp = fp = fn = 0
            cases = {}
            for (c, n), r in res.items():
                d = r[system]
                xy = d["xy"][d["p"] >= thr]
                g = gt[(c, n)]
                sc = score_detection(ground_truth=[tuple(p) for p in g], predictions=[tuple(p) for p in xy], radius=RADIUS)
                t, f_, m = sc.true_positives, sc.false_positives, sc.false_negatives
                tp, fp, fn = tp + t, fp + f_, fn + m
                a = cases.setdefault(c, np.zeros(3))
                a += (t, f_, m)
            arr = np.stack(list(cases.values()))
            rng = np.random.default_rng(0)
            boot = []
            for _ in range(2000):
                s = arr[rng.integers(0, len(arr), len(arr))].sum(0)
                boot.append(2 * s[0] / max(2 * s[0] + s[1] + s[2], 1))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            rows.append({"labels": lab, "label count": total, "system": system, "F1": f1, "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "TP": tp, "FP": fp, "FN": fn})
            for c, a in cases.items():
                per_case.append({"labels": lab, "system": system, "case": c, "tp": a[0], "fp": a[1], "fn": a[2]})
    df = pd.DataFrame(rows)
    md = ["# External validation on TUPAC16 (73 breast cancer cases, never used for training or tuning)", "",
          __doc__.strip(), "", f"Images: {len(res)}; cases: {len({c for c, _ in res})}.", "",
          df.round(4).to_markdown(index=False), ""]
    open(out_md, "w").write("\n".join(md))
    pd.DataFrame(per_case).to_csv(out_md.replace(".md", "_cases.csv"), index=False)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    pred_dir = os.path.join(TUPAC, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    if args.predict:
        predict(os.path.join(pred_dir, f"pred_{args.shard}_of_{args.nshards}.pkl"), args.shard, args.nshards)
    if args.score:
        import torch
        b2 = torch.load(os.path.join(ROOT, "checkpoints", "mitoguard_v2_bundle.pt"), map_location="cpu", weights_only=False)
        b1 = torch.load(os.path.join(ROOT, "checkpoints", "mitoguard_bundle.pt"), map_location="cpu", weights_only=False)
        folds = pd.read_csv(os.path.join(ROOT, "results", "v2", "folds.csv"))
        t_yolo = float(folds[(folds.protocol == "indomain") & (folds.system == "yolo")].threshold.iloc[0])
        thresholds = {"yolo": t_yolo, "v1": b1["det_thr"], "v2": b2["det_thr"]}
        print("thresholds (from MIDOG++ validation):", thresholds)
        os.makedirs(os.path.join(ROOT, "results", "external"), exist_ok=True)
        score(sorted(glob.glob(os.path.join(pred_dir, "pred_*.pkl"))), os.path.join(ROOT, "results", "external", "tupac16.md"), thresholds)


if __name__ == "__main__":
    main()
