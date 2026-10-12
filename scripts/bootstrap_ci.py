"""Paired bootstrap confidence intervals over held-out regions (pooled TP/FP/FN F1, as in MIDOG).

Regions are resampled with replacement (2,000 draws) within each protocol; every system is evaluated on
the same resampled regions, so differences are paired. Reports F1 [95% CI] and the F1 gain over the
YOLO11 baseline [95% CI] with a one-sided bootstrap p-value P(gain <= 0).
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--systems", nargs="+", required=True, help="systems from cases.csv.gz (and v2 labels)")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cases = pd.read_csv(os.path.join(args.results, "cases.csv.gz"))
    sel = os.path.join(args.results, "v2_selected_cases.csv.gz")
    if os.path.exists(sel):
        cases = pd.concat([cases, pd.read_csv(sel)])
    cases = cases[cases.split == "test"]
    rng = np.random.default_rng(0)
    out = []
    for protocol in ("loto", "loso"):
        c = cases[cases.protocol == protocol]
        ids = np.sort(c[c.system == "yolo"].case_id.unique())
        mats = {}
        for s in ["yolo"] + args.systems:
            g = c[c.system == s].set_index("case_id").reindex(ids)
            if g.tp.isna().any():
                continue
            mats[s] = g[["tp", "fp", "fn"]].to_numpy(float)
        draws = rng.integers(0, len(ids), size=(args.n, len(ids)))
        boot = {s: f1(*(m[draws].sum(1).T)) for s, m in mats.items()}
        for s, m in mats.items():
            point = f1(*m.sum(0))
            lo, hi = np.percentile(boot[s], [2.5, 97.5])
            row = {"protocol": protocol, "system": s, "regions": len(ids), "F1": point, "95% CI": f"[{lo:.3f}, {hi:.3f}]"}
            if s != "yolo":
                d = boot[s] - boot["yolo"]
                dl, dh = np.percentile(d, [2.5, 97.5])
                row.update({"gain vs YOLO11": point - f1(*mats["yolo"].sum(0)), "gain 95% CI": f"[{dl:.3f}, {dh:.3f}]",
                            "p (gain <= 0)": float((d <= 0).mean())})
            out.append(row)
    df = pd.DataFrame(out)
    md = "# Paired bootstrap 95% confidence intervals (held-out regions)\n\n" + __doc__.strip() + "\n\n" + df.round(4).to_markdown(index=False)
    open(os.path.join(args.results, "bootstrap.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
