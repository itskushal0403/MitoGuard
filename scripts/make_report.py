"""Results tables (results/summary.md) and figures (results/figures/*.png) from evaluate.py outputs."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.metrics import large_error_label, prf, safe_auroc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

# chart tokens (reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Main comparison: every filter uses the same candidate pool (m05i10) and learned stacker, so differences
# come from the features. Colour follows the system, fixed order.
PROPOSED = "yolo+phikon2-z112-ctr@m05i10+stack"
SYSTEMS = [
    ("yolo", "YOLO11 only (baseline)", "#2a78d6"),
    ("yolo+resnet50@m05i10+stack", "+ ImageNet ResNet-50 filter", "#eb6834"),
    ("yolo+phikon-z112-ctr@m05i10+stack", "+ Phikon filter", "#1baf7a"),
    (PROPOSED, "+ Phikon-v2 filter (proposed)", "#eda100"),
]
# Stage 2 design ablation (all on the stain-augmented Stage 1 models)
ABLATION = [
    ("yolo", "YOLO11 only"),
    ("yolo+resnet50", "ImageNet ResNet-50, global pool, filter alone (brief)"),
    ("yolo+phikon", "Phikon CLS token, filter alone (brief)"),
    ("yolo+phikon-z112-ctr", "Phikon cell-centred (zoom + centre tokens), filter alone"),
    ("yolo+phikon-z112-ctr+fused", "  + fixed fusion sqrt(p_yolo * p_filter)"),
    ("yolo+phikon-z112-ctr+stack", "  + learned stacker"),
    ("yolo+phikon-z112-ctr@m05i10+stack", "  + learned stacker + wide pool"),
    ("yolo+resnet50-z112+stack", "ImageNet ResNet-50 zoomed + learned stacker"),
    ("yolo+resnet50@m05i10+stack", "ImageNet ResNet-50 + learned stacker + wide pool"),
    ("yolo+phikon2-z112-ctr+stack", "Phikon-v2 cell-centred + learned stacker"),
    (PROPOSED, "Phikon-v2 cell-centred + learned stacker + wide pool (proposed)"),
]
LABEL = {k: v for k, v, _ in SYSTEMS}
COLOR = {k: c for k, _, c in SYSTEMS}
TUMOR_NAME = {"CCMCT": "Canine mast cell tumour", "CL": "Canine lymphoma", "CLC": "Canine lung carcinoma",
              "CSTS": "Canine soft-tissue sarcoma", "HBC": "Human breast carcinoma", "HMEL": "Human melanoma",
              "HNET": "Human neuroendocrine tumour"}
SCANNER_NAME = {"3DH": "3D Histech", "CS2": "Aperio CS2", "S360": "Hamamatsu S360", "XR": "Hamamatsu XR"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    })


def pooled(cases, keys):
    g = cases.groupby(keys)[["tp", "fp", "fn", "imp_hit", "imp_total"]].sum().reset_index()
    g["f1"] = [prf(r.tp, r.fp, r.fn)[2] for r in g.itertuples()]
    g["precision"] = [prf(r.tp, r.fp, r.fn)[0] for r in g.itertuples()]
    g["recall"] = [prf(r.tp, r.fp, r.fn)[1] for r in g.itertuples()]
    g["imposter_fp_rate"] = g.imp_hit / g.imp_total.clip(lower=1)
    return g


def grouped_bars(ax, df, xcol, systems, value="f1", names=None):
    xs = list(dict.fromkeys(df[xcol]))
    n = len(systems)
    slot = 0.8 / n
    width = min(slot - 0.03, 0.2)
    for k, s in enumerate(systems):
        sub = df[df.system == s].set_index(xcol).reindex(xs)
        pos = np.arange(len(xs)) - 0.4 + slot * (k + 0.5)
        ax.bar(pos, sub[value], width=width, color=COLOR[s], label=LABEL[s], edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([names.get(x, x) if names else x for x in xs], rotation=0)


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    folds = pd.read_csv(os.path.join(RES, "folds.csv"))
    cases = pd.read_csv(os.path.join(RES, "cases.csv"))
    cases = cases[cases.split == "test"]  # validation rows are only used to pick the demo's flag threshold
    systems = [s for s, _, _ in SYSTEMS if s in set(folds.system)]
    out = ["# MitoGuard results", "",
           "All numbers are on held-out domains only. Thresholds and calibration were tuned on validation cases",
           "drawn from the training domains. F1 uses the official MIDOG matching (7.5 um radius) via",
           "`grand_challenge_metrics.score_detection` (the renamed `evalutils` used by the MIDOG evaluation).", ""]

    for protocol, title, names in (("loto", "Leave-one-tumour-type-out (primary)", TUMOR_NAME),
                                   ("loso", "Leave-one-scanner-out (secondary)", SCANNER_NAME)):
        f = folds[folds.protocol == protocol]
        if f.empty:
            continue
        c = cases[cases.protocol == protocol]
        piv = f.pivot_table(index="fold", columns="system", values="f1")[[s for s in systems if s in f.system.unique()]]
        out += [f"## {title}", "", "### Held-out F1 per fold", "", piv.round(3).to_markdown(), ""]
        summ = []
        for s in piv.columns:
            fs = f[f.system == s]
            pl = pooled(c[c.system == s], ["system"]).iloc[0]
            summ.append({"system": LABEL.get(s, s), "mean F1 (folds)": fs.f1.mean(), "sd": fs.f1.std(),
                         "pooled F1": pl.f1, "pooled precision": pl.precision, "pooled recall": pl.recall,
                         "imposter FP rate": pl.imposter_fp_rate,
                         "count MAE": c[c.system == s].eval("abs(count - n_gt)").mean(),
                         "folds beating baseline": int((fs.set_index("fold").f1 > f[f.system == "yolo"].set_index("fold").f1).sum()) if s != "yolo" else None})
        out += ["### Summary", "", pd.DataFrame(summ).round(3).to_markdown(index=False), ""]
        imp = f.pivot_table(index="fold", columns="system", values="imposter_fp_rate")[piv.columns]
        out += ["### Imposter false-positive rate (share of labelled imposters counted as mitoses)", "",
                imp.round(3).to_markdown(), ""]

        # figure: F1 per held-out domain
        fig, ax = plt.subplots(figsize=(10, 4.2))
        main_sys = [s for s in systems if not s.endswith("_nostain")]
        grouped_bars(ax, f[f.system.isin(main_sys)], "fold", main_sys)
        ax.set_ylabel("Held-out F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}: detection F1 on the unseen domain")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(main_sys), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{protocol}_f1.png"), dpi=160)
        plt.close(fig)

        # improvement over the baseline per held-out domain
        base = f[f.system == "yolo"].set_index("fold").f1
        d = f[f.system.isin(main_sys[1:])].copy()
        d["delta"] = d.f1 - d.fold.map(base)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        wrapped = {k: v.replace(" ", "\n", 1) for k, v in names.items()}
        grouped_bars(ax, d, "fold", main_sys[1:], value="delta", names=wrapped)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylabel("F1 change vs YOLO11 only")
        ax.set_title(f"{title}: gain from the Stage 2 filter on each unseen domain")
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{protocol}_delta_f1.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.2))
        grouped_bars(ax, f[f.system.isin(main_sys)], "fold", main_sys, value="imposter_fp_rate")
        ax.set_ylabel("Imposters counted as mitoses")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{title}: imposter false-positive rate")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(main_sys), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{protocol}_imposter_fp.png"), dpi=160)
        plt.close(fig)

    # Stage 2 design ablation (LOTO + LOSO)
    rows = []
    for key, desc in ABLATION:
        r = {"Stage 2 design": desc}
        for protocol in ("loto", "loso"):
            fs = folds[(folds.protocol == protocol) & (folds.system == key)]
            if fs.empty:
                continue
            pl = pooled(cases[(cases.protocol == protocol) & (cases.system == key)], ["system"]).iloc[0]
            r[f"{protocol.upper()} mean F1"] = fs.f1.mean()
            r[f"{protocol.upper()} val F1"] = fs.val_f1.mean()
            r[f"{protocol.upper()} imposter FP"] = pl.imposter_fp_rate
        rows.append(r)
    out += ["## Stage 2 design ablation", "",
            "Mean held-out F1 over folds. `val F1` is the training-domain validation score used for every",
            "design decision; held-out numbers were not used for selection.", "",
            pd.DataFrame(rows).round(3).to_markdown(index=False), ""]

    # stain augmentation ablation
    ab = folds[folds.system.str.endswith("_nostain")]
    if not ab.empty:
        rows = []
        for (p, fo), g in folds.groupby(["protocol", "fold"]):
            for base in ("yolo", "yolo+phikon-z112-ctr@m05i10+stack", PROPOSED):
                a, b = g[g.system == base], g[g.system == base + "_nostain"]
                if len(a) and len(b):
                    rows.append({"protocol": p, "fold": fo, "system": base, "F1 with stain aug": a.f1.iloc[0],
                                 "F1 without": b.f1.iloc[0], "gain": a.f1.iloc[0] - b.f1.iloc[0]})
        if rows:
            t = pd.DataFrame(rows)
            out += ["## Ablation: H&E stain augmentation in Stage 1", "", t.round(3).to_markdown(index=False), "",
                    t.groupby(["protocol", "system"]).gain.agg(["mean", "std"]).round(3).to_markdown(), ""]

    # domain gap: in-domain model vs LOTO model, evaluated on the same official test cases
    ind = cases[(cases.protocol == "indomain")]
    if not ind.empty:
        lo = cases[(cases.protocol == "loto") & (cases.official_split == "test")]
        a = pooled(ind, ["system", "tumor"]).assign(setting="in-domain (tumour seen in training)")
        b = pooled(lo, ["system", "tumor"]).assign(setting="LOTO (tumour unseen)")
        gap = pd.concat([a, b])
        piv = gap.pivot_table(index=["tumor"], columns=["setting", "system"], values="f1")
        out += ["## Domain gap on the official test cases", "",
                "Same 111 test regions, scored by a model that saw the tumour type in training (in-domain) and by",
                "the LOTO model that never saw it.", "", piv.round(3).to_markdown(), ""]
        tot = pooled(pd.concat([ind.assign(setting="in-domain"), lo.assign(setting="LOTO")]), ["setting", "system"])
        out += [tot[["setting", "system", "f1", "precision", "recall", "imposter_fp_rate"]].round(3).to_markdown(index=False), ""]
        # figure
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, setting in zip(axes, ["in-domain (tumour seen in training)", "LOTO (tumour unseen)"]):
            d = gap[(gap.setting == setting) & gap.system.isin(main_sys)]
            grouped_bars(ax, d, "tumor", [s for s in main_sys if s in set(d.system)])
            ax.set_title(setting)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("F1 on official test regions")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=len(main_sys), fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(os.path.join(FIG, "domain_gap.png"), dpi=160)
        plt.close(fig)

    # reliability flag (LOTO: every case is held out exactly once -> pool all 503)
    c = cases[cases.protocol == "loto"].copy()
    if not c.empty:
        c["large_err"] = large_error_label(c["count"], c["n_gt"], 0.2)
        c["large_err3"] = large_error_label(c["count"], c["n_gt"], 0.2, min_abs=3)
        rows = []
        for s in systems:
            cs = c[c.system == s]
            r = {"system": LABEL.get(s, s), "regions": len(cs), "share with >20% error": cs.large_err.mean()}
            for u in ("u_var", "u_relstd", "u_gap"):
                r[f"AUROC {u}"] = safe_auroc(cs.large_err, cs[u].fillna(0))
            r["AUROC u_var (err>20% & >=3)"] = safe_auroc(cs.large_err3, cs.u_var.fillna(0))
            rows.append(r)
        out += ["## Count-reliability flag (LOTO, all 503 regions pooled)", "",
                "Target: region whose predicted count is off by more than 20% of the true count",
                "(denominator max(true count, 1)). `u_var` = sum p(1-p) / count is the score from the brief.", "",
                pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
        out += ["Per-fold AUROC of `u_var`:", "",
                folds[folds.protocol == "loto"].pivot_table(index="fold", columns="system", values="auroc_u_var")[systems].round(3).to_markdown(), ""]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        ax = axes[0]
        for s in main_sys:
            cs = c[c.system == s]
            fpr, tpr, _ = roc_curve(cs.large_err, cs.u_var.fillna(0))
            ax.plot(fpr, tpr, color=COLOR[s], lw=2, label=f"{LABEL[s]} (AUROC {safe_auroc(cs.large_err, cs.u_var.fillna(0)):.2f})")
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        ax.set_xlabel("False-flag rate (reliable regions flagged)")
        ax.set_ylabel("Unreliable regions caught")
        ax.set_title("Reliability flag u = sum p(1-p) / count")
        ax.grid(axis="both")
        ax.legend(loc="lower right", fontsize=8)
        ax = axes[1]
        prop = PROPOSED if PROPOSED in systems else systems[-1]
        cs = c[c.system == prop]
        rel = (cs["count"] - cs.n_gt).abs() / cs.n_gt.clip(lower=1)
        ax.scatter(cs.u_var, rel, s=18, color=COLOR[prop], edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
        ax.axhline(0.2, color=AXIS, lw=1)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlabel("Uncertainty score u")
        ax.set_ylabel("Relative count error")
        ax.set_title(f"{LABEL[prop]}: score vs actual error")
        ax.grid(axis="both")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "reliability.png"), dpi=160)
        plt.close(fig)

        # predicted vs true count
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
        for ax, s in zip(axes, ["yolo", prop]):
            cs = c[c.system == s]
            ax.scatter(cs.n_gt, cs["count"], s=16, color=COLOR[s], edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
            m = max(c.n_gt.max(), c["count"].max()) + 5
            ax.plot([0, m], [0, m], color=AXIS, lw=1)
            ax.set_xscale("symlog", linthresh=5)
            ax.set_yscale("symlog", linthresh=5)
            mae = (cs["count"] - cs.n_gt).abs().mean()
            ax.set_title(f"{LABEL[s]}  (MAE {mae:.1f})")
            ax.set_xlabel("True mitotic count")
            ax.grid(axis="both")
        axes[0].set_ylabel("Predicted mitotic count")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "counts.png"), dpi=160)
        plt.close(fig)

    open(os.path.join(RES, "summary.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
