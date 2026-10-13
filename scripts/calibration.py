"""Candidate-level probability calibration on held-out domains (LOTO): expected calibration error (ECE,
15 equal-width bins) and a reliability diagram for YOLO's raw p_mitosis, the v1 frozen-FM filter and the
v2 fine-tuned classifier (all Platt-calibrated on training-domain validation candidates only)."""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, make_folds  # noqa: E402
from mitoguard.metrics import label_candidates  # noqa: E402
from mitoguard.stage2 import in_pool, s2_name  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("MITOGUARD_RESULTS", os.path.join(ROOT, "results"))
SYSTEMS = [("YOLO11 p_mitosis (raw)", "stain", None, "m05i10", "#2a78d6"),
           ("v1 frozen Phikon-v2 filter", "stain", "phikon2-z112-ctr", "m05i10", "#eb6834"),
           ("v2 fine-tuned Phikon-v2", "union", "ft-phikon2", "a05", "#1baf7a"),
           ("v2 fine-tuned + site adaptation", "union", "ft-phikon2+st", "a05", "#eda100")]


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append((p[m].mean(), y[m].mean(), m.sum()))
    return e, np.array(rows)


def collect(variant, feat, pool):
    ps, ys = [], []
    for fold in make_folds("loto"):
        run = f"{fold.tag}_{variant}"
        cands = pickle.load(open(os.path.join(WORK_ROOT, "cands", run + ".pkl"), "rb"))
        s2 = None
        if feat:
            f = os.path.join(WORK_ROOT, "stage2", s2_name(run, feat, pool) + ".pkl")
            if not os.path.exists(f):
                return None, None
            s2 = pickle.load(open(f, "rb"))["scores"]
        for cid in fold.test:
            c = cands[cid]
            m = in_pool(c, pool)
            ps.append(s2[cid]["p2"][m] if s2 else c["p_mit"][m])
            ys.append(label_candidates(cid, c["xy"][m]) == 1)
    return np.concatenate(ps), np.concatenate(ys).astype(float)


def main():
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1)
    rows = []
    for label, variant, feat, pool, col in SYSTEMS:
        p, y = collect(variant, feat, pool)
        if p is None:
            continue
        e, r = ece(p, y)
        rows.append({"system": label, "held-out candidates": len(p), "ECE": e, "Brier": float(np.mean((p - y) ** 2))})
        ax.plot(r[:, 0], r[:, 1], "-o", color=col, lw=2, ms=4, label=f"{label} (ECE {e:.3f})")
    ax.set_xlabel("Predicted probability of mitosis")
    ax.set_ylabel("Observed frequency (held-out tumour types)")
    ax.set_title("Calibration on unseen tumour types", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(color="#e1e0d9", lw=0.8)
    fig.tight_layout()
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RES, "figures", "calibration.png"), dpi=160)
    md = "# Candidate-level calibration on held-out tumour types (LOTO)\n\n" + pd.DataFrame(rows).round(4).to_markdown(index=False)
    open(os.path.join(RES, "calibration.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
