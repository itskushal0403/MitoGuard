"""Average the fine-tuned classifiers' raw probabilities per crop-bank row (WORK/ft/<fold>__<a>.npz ...)
into WORK/ft/<fold>__<name>.npz, so the ensemble is evaluated exactly like a single backbone."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, all_fold_tags  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    d = os.path.join(WORK_ROOT, "ft")
    for tag in all_fold_tags():
        files = [os.path.join(d, f"{tag}__{m}.npz") for m in args.members]
        if not all(os.path.exists(f) for f in files):
            print("skip", tag)
            continue
        maps = [dict(zip(np.load(f)["rows"].tolist(), np.load(f)["prob"].tolist())) for f in files]
        rows = sorted(set.intersection(*(set(m) for m in maps)))
        # average in logit space (geometric mean of odds) - robust to differently scaled members
        lg = np.mean([[np.log(np.clip(m[r], 1e-5, 1 - 1e-5) / (1 - np.clip(m[r], 1e-5, 1 - 1e-5))) for r in rows] for m in maps], 0)
        np.savez(os.path.join(d, f"{tag}__{args.name}.npz"), rows=np.array(rows), prob=(1 / (1 + np.exp(-lg))).astype(np.float32))
        print(tag, len(rows), flush=True)


if __name__ == "__main__":
    main()
