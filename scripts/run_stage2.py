"""Train the Stage 2 imposter filter for every (fold, backbone) and score the fold's candidates.

Output: WORK/stage2/<yolo_run>__<backbone>.pkl with
    {"scores": {case_id: {"p2": (N,), "p2_members": (N, k)}}  for val + test cases (aligned with cands)
     "filter": ImposterFilter state, "n_train": ..., "val_ap": ...}
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.bank import Bank  # noqa: E402
from mitoguard.data import IMPOSTER, MITOSIS, WORK_ROOT, cells_of, get_fold  # noqa: E402
from mitoguard.metrics import label_candidates  # noqa: E402
from mitoguard.stage2 import ImposterFilter, in_pool, s2_name  # noqa: E402



def assemble(fold, cands, bank, split, pool):
    """Feature rows + labels for a split. Training adds labelled cells; val/test use candidates only."""
    rows, ys = [], []
    for cid in getattr(fold, split):
        c = cands[cid]
        m = in_pool(c, pool)
        r = bank.rows(cid, c["xy"][m])
        y = (label_candidates(cid, c["xy"][m]) == 1).astype(np.int64)
        if split == "train":
            mit, imp = cells_of(cid, MITOSIS), cells_of(cid, IMPOSTER)
            r = np.concatenate([r, bank.rows(cid, mit), bank.rows(cid, imp)])
            y = np.concatenate([y, np.ones(len(mit), np.int64), np.zeros(len(imp), np.int64)])
            r, first = np.unique(r, return_index=True)  # one sample per bank point
            y = y[first]
        rows.append(r)
        ys.append(y)
    rows, ys = np.concatenate(rows), np.concatenate(ys)
    return bank.get(rows), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="YOLO run names, e.g. loto_HBC_stain")
    ap.add_argument("--backbones", nargs="+", default=["phikon", "phikon2", "resnet50"])
    ap.add_argument("--pool", default="m05", help="candidate pool, see mitoguard.stage2.POOLS")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(WORK_ROOT, "stage2")
    os.makedirs(out_dir, exist_ok=True)
    for bb in args.backbones:
        bank = Bank(bb)
        for run in args.runs:
            out = os.path.join(out_dir, s2_name(run, bb, args.pool) + ".pkl")
            if os.path.exists(out) and not args.overwrite:
                continue
            t0 = time.time()
            fold = get_fold(run.rsplit("_", 1)[0])
            cands = pickle.load(open(os.path.join(WORK_ROOT, "cands", run + ".pkl"), "rb"))
            xtr, ytr = assemble(fold, cands, bank, "train", args.pool)
            xva, yva = assemble(fold, cands, bank, "val", args.pool)
            filt = ImposterFilter().fit(xtr, ytr, xva, yva)
            scores = {}
            for cid in fold.val + fold.test:
                c = cands[cid]
                p2 = np.zeros(len(c["xy"]), np.float32)
                mem = np.zeros((len(c["xy"]), filt.n_models), np.float32)
                m = in_pool(c, args.pool)
                if m.any():
                    p2[m], mem[m] = filt.predict(bank.get(bank.rows(cid, c["xy"][m])))
                scores[cid] = {"p2": p2, "p2_members": mem}
            pickle.dump({"scores": scores, "pool": args.pool, "filter": filt.state(), "n_train": len(ytr), "pos_train": int(ytr.sum()),
                         "val_ap": filt.val_ap}, open(out, "wb"))
            print(f"{run} {bb}@{args.pool}: train={len(ytr)} (pos {ytr.sum()}), val={len(yva)} (pos {yva.sum()}), "
                  f"val AP={np.mean(filt.val_ap):.3f}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
