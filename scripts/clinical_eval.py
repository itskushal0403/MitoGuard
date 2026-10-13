"""Clinically oriented evaluation of region-level mitotic counts (from results/cases.csv.gz).

1. Count agreement on held-out regions: MAE, Lin's concordance correlation (CCC), Spearman rho,
   Bland-Altman bias and 95% limits of agreement.
2. Mitotic-count category agreement: each MIDOG++ region is ~2 mm^2 (about 10 high-power fields), so the
   counts are binned with the Nottingham-style cut-offs <=7 / 8-14 / >=15; quadratic-weighted Cohen's kappa.
3. Conformal count intervals (novel for mitotic counting): the count's Poisson-binomial spread
   sigma^2 = sum p(1-p) (from calibrated candidate probabilities) is turned into a calibrated interval by
   normalised split-conformal prediction (Lei et al., JASA 2018):
       score_i = |count_i - true_i| / sqrt(sigma_i^2 + 1),   q = conformal (1 - alpha) quantile on the fold's
       validation regions (training domains only),   interval = count +- q sqrt(sigma^2 + 1).
   Coverage is then measured on the held-out domain, against the naive interval count +- 1.645 sigma.
4. Grade-ambiguity flag: a region is flagged when its 90% interval straddles a category cut-off; we test
   whether flagged regions are the ones whose category is wrong.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("MITOGUARD_RESULTS", os.path.join(ROOT, "results"))
CUTS = (7.5, 14.5)  # category boundaries: <=7 | 8-14 | >=15


def category(x):
    return np.digitize(np.asarray(x, float), CUTS)


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def conformal_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def evaluate(cases, system, protocol, alpha=0.1):
    c = cases[(cases.protocol == protocol) & (cases.system == system)].copy()
    if c.empty:
        return None, None
    c["sigma"] = np.sqrt(c.u_var * c["count"].clip(lower=1))
    c["score"] = (c["count"] - c.n_gt).abs() / np.sqrt(c.sigma ** 2 + 1)
    rows = []
    for fold, g in c.groupby("fold"):
        q = conformal_q(g[g.split == "val"].score.to_numpy(), alpha)
        t = g[g.split == "test"].copy()
        t["lo"] = (t["count"] - q * np.sqrt(t.sigma ** 2 + 1)).clip(lower=0)
        t["hi"] = t["count"] + q * np.sqrt(t.sigma ** 2 + 1)
        t["naive_lo"], t["naive_hi"] = t["count"] - 1.645 * t.sigma, t["count"] + 1.645 * t.sigma
        t["q"] = q
        rows.append(t)
    t = pd.concat(rows)
    pred, gt = t["count"].to_numpy(), t.n_gt.to_numpy()
    cov = ((gt >= t.lo) & (gt <= t.hi)).mean()
    naive_cov = ((gt >= t.naive_lo) & (gt <= t.naive_hi)).mean()
    cat_p, cat_g = category(pred), category(gt)
    ambiguous = np.zeros(len(t), bool)
    for cut in CUTS:
        ambiguous |= (t.lo.to_numpy() < cut) & (t.hi.to_numpy() > cut)
    wrong = cat_p != cat_g
    diff = pred - gt
    summary = {
        "system": system, "protocol": protocol, "regions": len(t),
        "count MAE": np.abs(diff).mean(), "CCC": ccc(pred, gt), "Spearman": spearmanr(pred, gt).correlation,
        "Bland-Altman bias": diff.mean(), "LoA low": diff.mean() - 1.96 * diff.std(), "LoA high": diff.mean() + 1.96 * diff.std(),
        "category accuracy": (~wrong).mean(), "category kappa (quadratic)": cohen_kappa_score(cat_g, cat_p, weights="quadratic"),
        f"conformal {int((1 - alpha) * 100)}% coverage": cov, "naive +-1.645 sigma coverage": naive_cov,
        "median interval width": float(np.median(t.hi - t.lo)),
        "flagged (interval straddles a cut-off)": ambiguous.mean(),
        "category accuracy | flagged": (~wrong[ambiguous]).mean() if ambiguous.any() else np.nan,
        "category accuracy | not flagged": (~wrong[~ambiguous]).mean() if (~ambiguous).any() else np.nan,
        "share of category errors caught by flag": (ambiguous & wrong).sum() / max(wrong.sum(), 1),
    }
    per_fold = t.groupby("fold").apply(lambda g: pd.Series({
        "coverage": ((g.n_gt >= g.lo) & (g.n_gt <= g.hi)).mean(), "q": g.q.iloc[0],
        "kappa": cohen_kappa_score(category(g.n_gt), category(g["count"]), weights="quadratic") if len(set(category(g.n_gt))) > 1 else np.nan,
        "CCC": ccc(g["count"], g.n_gt)}), include_groups=False).reset_index()
    per_fold.insert(0, "system", system)
    return summary, per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    cases = pd.read_csv(os.path.join(RES, "cases.csv.gz"))
    sel = os.path.join(RES, "v2_selected_cases.csv.gz")
    if os.path.exists(sel):
        cases = pd.concat([cases, pd.read_csv(sel)])
    out = ["# Clinical evaluation of region-level mitotic counts", "",
           "Held-out regions only. Conformal intervals are calibrated on each fold's validation regions",
           "(training domains); categories use the cut-offs <=7 / 8-14 / >=15 per ~2 mm^2 region.", ""]
    for protocol in ("loto", "loso"):
        summ, folds = [], []
        for s in args.systems:
            a, b = evaluate(cases, s, protocol, args.alpha)
            if a:
                summ.append(a)
                folds.append(b)
        if not summ:
            continue
        out += [f"## {protocol.upper()}", "", pd.DataFrame(summ).set_index("system").T.round(3).to_markdown(), "",
                "Per fold:", "", pd.concat(folds).round(3).to_markdown(index=False), ""]
    open(os.path.join(RES, "clinical.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
