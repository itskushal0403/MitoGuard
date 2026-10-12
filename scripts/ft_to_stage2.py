"""Convert fine-tuned Stage 2 scores (WORK/ft/*.npz, keyed by crop-bank row) into Stage 2 result files
for any Stage 1 run and candidate pool, so evaluate.py treats them like any other filter ("ft-<backbone>").
Raw probabilities are Platt-calibrated on the fold's validation candidates (training domains only).
"""
import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.bank import Bank  # noqa: E402
from mitoguard.data import WORK_ROOT, get_fold  # noqa: E402
from mitoguard.metrics import label_candidates  # noqa: E402
from mitoguard.stage2 import in_pool, s2_name  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--backbone", required=True, help="name used in WORK/ft/<fold>__<backbone>.npz (e.g. phikon2)")
    ap.add_argument("--variants", nargs="+", default=["union"])
    ap.add_argument("--pools", nargs="+", default=["m05i10", "a05"])
    args = ap.parse_args()
    bank = Bank("phikon")  # only the point positions / offsets are used
    feat = f"ft-{args.backbone}"
    for tag in args.folds:
        f = np.load(os.path.join(WORK_ROOT, "ft", f"{tag}__{args.backbone}.npz"))
        lut = dict(zip(f["rows"].tolist(), f["prob"].tolist()))
        fold = get_fold(tag)
        for var in args.variants:
            run = f"{tag}_{var}"
            cands = pickle.load(open(os.path.join(WORK_ROOT, "cands", run + ".pkl"), "rb"))
            for pool in args.pools:
                raw = {}
                for cid in fold.val + fold.test:
                    c = cands[cid]
                    m = in_pool(c, pool)
                    rows = bank.rows(cid, c["xy"][m])
                    p = np.full(len(c["xy"]), 0.0, np.float32)
                    p[m] = [lut.get(int(r), np.nan) for r in rows]
                    raw[cid] = (p, m)
                missing = sum(np.isnan(p[m]).sum() for p, m in raw.values())
                xv = np.concatenate([raw[c][0][raw[c][1]] for c in fold.val])
                yv = np.concatenate([label_candidates(c, cands[c]["xy"][raw[c][1]]) == 1 for c in fold.val])
                ok = ~np.isnan(xv)
                platt = LogisticRegression(C=1e3).fit(logit(xv[ok])[:, None], yv[ok])
                scores = {}
                for cid, (p, m) in raw.items():
                    q = np.zeros(len(p), np.float32)
                    pm = np.nan_to_num(p[m], nan=0.0)
                    if m.any():
                        q[m] = platt.predict_proba(logit(pm)[:, None])[:, 1]
                    scores[cid] = {"p2": q, "p2_members": q[:, None]}
                out = os.path.join(WORK_ROOT, "stage2", s2_name(run, feat, pool) + ".pkl")
                pickle.dump({"scores": scores, "pool": pool, "filter": None, "platt": platt}, open(out, "wb"))
                print(f"{run} {feat}@{pool}: missing={missing}", flush=True)


if __name__ == "__main__":
    main()
