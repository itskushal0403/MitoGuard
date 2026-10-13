"""MitoGuard v2 figures + summary (results/summary_v2.md, results/figures/v2_*.png)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.metrics import prf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("MITOGUARD_RESULTS", os.path.join(ROOT, "results"))
FIG = os.path.join(RES, "figures")
SURFACE, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
V1 = "yolo+phikon2-z112-ctr@m05i10+stack"
SYSTEMS = [("yolo", "YOLO11 only", "#2a78d6"), (V1, "MitoGuard v1 (frozen FM)", "#eb6834"),
           ("MitoGuard-v2", "MitoGuard v2 (fine-tuned FM)", "#1baf7a"),
           ("MitoGuard-v2-SA", "v2 + site adaptation (fixed budget)", "#eda100"),
           ("MitoGuard-v2-SApa", "v2 + site adaptation (prevalence-aware, post-hoc)", "#e87ba4")]
NAMES = {"CCMCT": "Canine mast\ncell tumour", "CL": "Canine\nlymphoma", "CLC": "Canine lung\ncarcinoma",
         "CSTS": "Canine soft-\ntissue sarcoma", "HBC": "Human breast\ncarcinoma", "HMEL": "Human\nmelanoma",
         "HNET": "Human neuro-\nendocrine", "3DH": "3D Histech", "CS2": "Aperio CS2", "S360": "Hamamatsu\nS360",
         "XR": "Hamamatsu\nXR"}


def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True,
                         "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold"})


def per_fold_f1(cases, protocol, system):
    c = cases[(cases.protocol == protocol) & (cases.system == system) & (cases.split == "test")]
    return {f: prf(g.tp.sum(), g.fp.sum(), g.fn.sum())[2] for f, g in c.groupby("fold")}


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    cases = pd.read_csv(os.path.join(RES, "cases.csv.gz"))
    sel = pd.read_csv(os.path.join(RES, "v2_selected_cases.csv.gz"))
    allc = pd.concat([cases, sel])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [7, 4]})
    table = []
    for ax, protocol, title in zip(axes, ("loto", "loso"), ("Unseen tumour type (leave-one-tumour-out)",
                                                              "Unseen scanner (leave-one-scanner-out)")):
        f1s = {s: per_fold_f1(allc, protocol, s) for s, _, _ in SYSTEMS}
        folds = sorted(f1s["yolo"])
        w = 0.8 / len(SYSTEMS)
        for k, (s, label, col) in enumerate(SYSTEMS):
            vals = [f1s[s].get(f, np.nan) for f in folds]
            ax.bar(np.arange(len(folds)) - 0.4 + w * (k + 0.5), vals, width=w * 0.9, color=col, label=label)
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels([NAMES.get(f, f) for f in folds], fontsize=7.5)
        ax.set_ylim(0.3 if protocol == "loso" else 0.5, 0.9)
        ax.set_title(title)
        for f in folds:
            table.append({"protocol": protocol, "held-out": f.replace("\n", " "),
                          **{label: f1s[s].get(f, np.nan) for s, label, _ in SYSTEMS}})
    axes[0].set_ylabel("Held-out F1")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(FIG, "v2_f1.png"), dpi=160)
    plt.close(fig)

    # counts with conformal intervals (site-adapted v2, unseen tumour)
    c = sel[(sel.protocol == "loto") & (sel.system == "MitoGuard-v2-SA") & (sel.split == "test")]
    if c.empty:
        c = sel[(sel.protocol == "loto") & (sel.system == "MitoGuard-v2") & (sel.split == "test")]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(c.n_gt, c["count"], s=12, color="#1baf7a", edgecolor=SURFACE, linewidth=0.6, alpha=0.85)
    m = max(c.n_gt.max(), c["count"].max()) + 5
    ax.plot([0, m], [0, m], color=AXIS, lw=1)
    for cut in (7.5, 14.5):
        ax.axvline(cut, color=GRID, lw=1)
        ax.axhline(cut, color=GRID, lw=1)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("True mitotic count per region")
    ax.set_ylabel("MitoGuard v2 count")
    ax.set_title("Unseen tumour types: counts vs ground truth")
    ax.grid(axis="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_counts.png"), dpi=160)
    plt.close(fig)

    md = ["# MitoGuard v2 results", "", "## Held-out F1 per domain", "",
          pd.DataFrame(table).round(3).to_markdown(index=False), ""]
    for f in ("selection_v2.md", "bootstrap.md", "clinical.md"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            md += [open(p).read().replace("\n# ", "\n## ").replace("# ", "## ", 1), ""]
    open(os.path.join(RES, "summary_v2.md"), "w").write("\n".join(md))
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
