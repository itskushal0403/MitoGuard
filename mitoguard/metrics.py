"""Evaluation following the official MIDOG protocol.

Detection matching uses `score_detection` from grand-challenge-metrics (the renamed `evalutils`
package that the official MIDOG / MIDOG++ evaluation calls), with the official 7.5 um radius
converted to pixels per image.  F1 is computed from TP/FP/FN pooled over all cases of a set,
exactly like the MIDOG challenge aggregate.
"""
from __future__ import annotations

import numpy as np
from grand_challenge_metrics.scorers import score_detection
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from .data import IMPOSTER, MITOSIS, cells_of, radius_px


def match_case(case_id: int, pred_xy: np.ndarray) -> dict:
    gt = cells_of(case_id, MITOSIS)
    pred = [tuple(p) for p in np.asarray(pred_xy).reshape(-1, 2)]
    sc = score_detection(ground_truth=[tuple(g) for g in gt], predictions=pred, radius=radius_px(case_id))
    return {"tp": sc.true_positives, "fp": sc.false_positives, "fn": sc.false_negatives, "n_gt": len(gt), "n_pred": len(pred)}


def prf(tp, fp, fn, eps=1e-6):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return p, r, f1


def imposter_hits(case_id: int, pred_xy: np.ndarray) -> tuple[int, int]:
    """Number of labelled imposters with a kept detection within the matching radius that is not
    itself within the radius of a true mitosis (i.e. the imposter is counted as a mitosis)."""
    imp = cells_of(case_id, IMPOSTER)
    if len(imp) == 0:
        return 0, 0
    pred = np.asarray(pred_xy).reshape(-1, 2)
    if len(pred) == 0:
        return 0, len(imp)
    r = radius_px(case_id)
    mit = cells_of(case_id, MITOSIS)
    if len(mit):
        d_mit, _ = cKDTree(mit).query(pred)
        pred = pred[d_mit > r]
    if len(pred) == 0:
        return 0, len(imp)
    d, _ = cKDTree(pred).query(imp)
    return int((d <= r).sum()), len(imp)


def label_candidates(case_id: int, xy: np.ndarray) -> np.ndarray:
    """1 if candidate lies within the matching radius of a true mitosis, else 0;
    2 if it is not a mitosis but within radius of a labelled imposter."""
    xy = np.asarray(xy).reshape(-1, 2)
    out = np.zeros(len(xy), dtype=np.int64)
    if len(xy) == 0:
        return out
    r = radius_px(case_id)
    imp = cells_of(case_id, IMPOSTER)
    if len(imp):
        d, _ = cKDTree(imp).query(xy)
        out[d <= r] = 2
    mit = cells_of(case_id, MITOSIS)
    if len(mit):
        d, _ = cKDTree(mit).query(xy)
        out[d <= r] = 1
    return out


def evaluate_set(dets: dict, case_ids, thr: float, extra_cols=None) -> dict:
    """dets: case_id -> dict(xy=Nx2, score=N). Keeps detections with score >= thr."""
    tp = fp = fn = imp_hit = imp_tot = 0
    per_case = []
    for cid in case_ids:
        d = dets[cid]
        keep = d["score"] >= thr
        xy = d["xy"][keep]
        m = match_case(cid, xy)
        h, n = imposter_hits(cid, xy)
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        imp_hit, imp_tot = imp_hit + h, imp_tot + n
        per_case.append({"case_id": cid, **m, "imp_hit": h, "imp_total": n})
    p, r, f1 = prf(tp, fp, fn)
    counts = np.array([c["n_pred"] for c in per_case], dtype=float)
    gts = np.array([c["n_gt"] for c in per_case], dtype=float)
    return {
        "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
        "imposter_fp_rate": imp_hit / max(imp_tot, 1), "imposters_hit": imp_hit, "imposters_total": imp_tot,
        "count_mae": float(np.mean(np.abs(counts - gts))),
        "count_rel_err": float(np.mean(np.abs(counts - gts) / np.maximum(gts, 1))),
        "per_case": per_case,
    }


def best_threshold(dets: dict, case_ids, grid=None) -> tuple[float, float]:
    """Threshold maximising pooled F1 on `case_ids` (used on validation cases only)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    # pre-match once per case at the lowest threshold for speed: sort by score and sweep
    best = (0.5, -1.0)
    for t in grid:
        tp = fp = fn = 0
        for cid in case_ids:
            d = dets[cid]
            m = match_case(cid, d["xy"][d["score"] >= t])
            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        f1 = prf(tp, fp, fn)[2]
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def large_error_label(pred_count, gt_count, rel=0.2, min_abs=0):
    """Region is 'unreliable' if |pred-gt| > rel*max(gt,1) (and >= min_abs)."""
    pred_count, gt_count = np.asarray(pred_count, float), np.asarray(gt_count, float)
    err = np.abs(pred_count - gt_count)
    return ((err > rel * np.maximum(gt_count, 1)) & (err >= min_abs)).astype(int)


def safe_auroc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))
