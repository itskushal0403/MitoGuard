"""Count-reliability scores for a region.

Given calibrated per-candidate mitosis probabilities p_i (candidates in the Stage 1 pool), the
predicted count is the number of candidates with p_i >= t.  Treating each candidate as an
independent Bernoulli(p_i) the count has variance  sum_i p_i (1 - p_i).
  - u_var     : sum p(1-p) / max(count, 1)           (the score proposed in the project brief)
  - u_relstd  : sqrt(sum p(1-p)) / max(count, 1)     (relative standard deviation of the count)
  - u_ens     : std over ensemble members of their hard counts / max(count, 1)  (epistemic)
  - u_gap     : |sum p - count| / max(count, 1)      (disagreement between soft and hard count)
High values mean many borderline candidates relative to the count -> flag for review.
"""
from __future__ import annotations

import numpy as np


def region_scores(p: np.ndarray, thr: float, members: np.ndarray | None = None) -> dict:
    p = np.asarray(p, dtype=np.float64)
    count = float((p >= thr).sum())
    denom = max(count, 1.0)
    var = float((p * (1 - p)).sum())
    out = {
        "count": count,
        "soft_count": float(p.sum()),
        "u_var": var / denom,
        "u_relstd": np.sqrt(var) / denom,
        "u_gap": abs(p.sum() - count) / denom,
    }
    if members is not None and len(members):
        member_counts = (np.asarray(members) >= thr).sum(0)
        out["u_ens"] = float(np.std(member_counts)) / denom
    else:
        out["u_ens"] = np.nan
    return out
