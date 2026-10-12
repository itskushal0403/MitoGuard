"""Label-free adaptation of candidate posteriors to a new region / site (label shift).

The fraction of true mitoses among detector candidates varies by an order of magnitude between regions
(a lymphoma hotspot vs a quiet breast region) and between labs. A classifier calibrated on the training
domains bakes in the training prior pi_s. Saerens, Latinne & Decaestecker (Neural Computation 2002)
re-estimate the target prior pi_t by EM on unlabelled target posteriors and re-weight them:

    p'_i = (pi_t/pi_s) p_i / [ (pi_t/pi_s) p_i + ((1-pi_t)/(1-pi_s)) (1-p_i) ],    pi_t <- mean_i p'_i
"""
from __future__ import annotations

import numpy as np


def em_prior_shift(p: np.ndarray, pi_s: float, iters: int = 100, tol: float = 1e-6, min_n: int = 5) -> np.ndarray:
    """Adapt calibrated posteriors `p` (one region or one site) from source prior `pi_s`."""
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    if len(p) < min_n:
        return p
    pi_t = p.mean()
    for _ in range(iters):
        w1, w0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
        q = w1 * p / (w1 * p + w0 * (1 - p))
        new = q.mean()
        if abs(new - pi_t) < tol:
            break
        pi_t = new
    w1, w0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    return w1 * p / (w1 * p + w0 * (1 - p))
