"""Merge each fold's two Stage 1 detectors (with / without stain augmentation) into one candidate set.

Detections within 8 px are merged; the merged box keeps the max of each class score over its members.
Output: WORK/cands/<fold>_union.pkl, same format as detect_all.py.
"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, all_fold_tags  # noqa: E402
from mitoguard.detector import distance_nms  # noqa: E402


def merge(a, b, radius=8.0):
    xy = np.concatenate([a["xy"], b["xy"]])
    pm, pi = np.concatenate([a["p_mit"], b["p_mit"]]), np.concatenate([a["p_imp"], b["p_imp"]])
    if len(xy) == 0:
        return a
    keep = distance_nms(xy, np.maximum(pm, pi), radius)
    tree = cKDTree(xy)
    out_m, out_i = np.empty(len(keep), np.float32), np.empty(len(keep), np.float32)
    for j, k in enumerate(keep):
        nb = tree.query_ball_point(xy[k], radius)
        out_m[j], out_i[j] = pm[nb].max(), pi[nb].max()
    return {"xy": xy[keep].astype(np.float32), "p_mit": out_m, "p_imp": out_i}


def main():
    d = os.path.join(WORK_ROOT, "cands")
    for tag in all_fold_tags():
        a, b = (pickle.load(open(os.path.join(d, f"{tag}_{v}.pkl"), "rb")) for v in ("stain", "nostain"))
        u = {cid: merge(a[cid], b[cid]) for cid in a}
        pickle.dump(u, open(os.path.join(d, f"{tag}_union.pkl"), "wb"))
        print(tag, sum(len(v["xy"]) for v in a.values()), sum(len(v["xy"]) for v in b.values()), "->", sum(len(v["xy"]) for v in u.values()), flush=True)


if __name__ == "__main__":
    main()
