"""Why do pathology foundation-model features generalise better? Detector-independent analysis on the
26,286 labelled cells (11,937 mitoses, 14,349 imposters).

A. Leave-one-tumour-out linear probe: logistic regression on frozen features, mitosis vs imposter,
   trained on 6 tumour types and tested on the 7th.  Pure test of how well each feature space separates
   the classes in a domain it has never seen (no YOLO involved).
B. Scanner leakage: within human breast cancer only (same tumour, same lab, three scanners), how well can
   the scanner be predicted from a cell's features?  Tumour type is held constant, so above-chance
   accuracy is pure scanner signal - information a domain-robust feature should not carry.
C. UMAP of each feature space coloured by class and by domain.

Outputs results/feature_probe.csv, results/feature_scanner_leakage.csv, results/figures/umap_*.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.bank import Bank  # noqa: E402
from mitoguard.data import WORK_ROOT, load_annotations, load_cases  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SURFACE, INK2, MUTED, AXIS = "#fcfcfb", "#52514e", "#898781", "#c3c2b7"
NAMES = {"resnet50": "ImageNet ResNet-50", "resnet50-z112": "ImageNet ResNet-50, zoom",
         "phikon": "Phikon CLS", "phikon-z112-ctr": "Phikon cell-centred", "phikon2": "Phikon-v2 CLS",
         "phikon2-z112-ctr": "Phikon-v2 cell-centred", "uni": "UNI (ViT-L)"}


def cell_table(bank):
    ann = load_annotations().copy()
    cases = load_cases()
    rows = np.zeros(len(ann), np.int64)
    for cid, idx in ann.groupby("case_id").groups.items():
        rows[ann.index.get_indexer(idx)] = bank.rows(cid, ann.loc[idx, ["x", "y"]].to_numpy(np.float32))
    ann["row"] = rows
    ann["y"] = (ann.label == 1).astype(int)
    ann["tumor"] = ann.case_id.map(cases.tumor_short)
    ann["scanner"] = ann.case_id.map(cases.scanner_short)
    return ann


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "phikon", "phikon2"])
    ap.add_argument("--umap", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    probe_rows, leak_rows, embeds = [], [], {}
    for bb in args.backbones:
        if not os.path.exists(os.path.join(WORK_ROOT, "bank", f"{bb}.npy")):
            continue
        bank = Bank(bb)
        cells = cell_table(bank)
        X = bank.get(cells.row.to_numpy())
        # A. LOTO linear probe
        for t in sorted(cells.tumor.unique()):
            tr, te = (cells.tumor != t).to_numpy(), (cells.tumor == t).to_numpy()
            clf = probe().fit(X[tr], cells.y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            probe_rows.append({"backbone": bb, "held_out_tumor": t, "auroc": roc_auc_score(cells.y[te], p),
                               "balanced_acc": balanced_accuracy_score(cells.y[te], p >= 0.5)})
        # B. scanner leakage inside human breast cancer (case-grouped CV so no case is in train and test)
        hbc = (cells.tumor == "HBC").to_numpy()
        Xh, sh, gh = X[hbc], cells.scanner[hbc].to_numpy(), cells.case_id[hbc].to_numpy()
        pred = np.empty(len(sh), dtype=object)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xh, sh, gh):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(Xh[tr], sh[tr])
            pred[te] = clf.predict(Xh[te])
        leak_rows.append({"backbone": bb, "scanner_balanced_acc": balanced_accuracy_score(sh, pred.astype(str)),
                          "chance": 1 / len(np.unique(sh))})
        print(bb, pd.DataFrame(probe_rows[-7:]).auroc.mean().round(3), leak_rows[-1], flush=True)
        if args.umap:
            rng = np.random.default_rng(0)
            sub = rng.choice(len(cells), size=min(6000, len(cells)), replace=False)
            import umap
            emb = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=0).fit_transform(
                StandardScaler().fit_transform(X[sub]))
            embeds[bb] = (emb, cells.iloc[sub])
    pr = pd.DataFrame(probe_rows)
    pr.to_csv(os.path.join(RES, "feature_probe.csv"), index=False)
    pd.DataFrame(leak_rows).to_csv(os.path.join(RES, "feature_scanner_leakage.csv"), index=False)
    print(pr.pivot(index="held_out_tumor", columns="backbone", values="auroc").round(3))
    print(pd.DataFrame(leak_rows))
    if embeds:
        plot_umaps(embeds)


def plot_umaps(embeds):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold"})
    n = len(embeds)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.2), squeeze=False)
    cls_col = {0: "#2a78d6", 1: "#e34948"}  # imposters first so mitoses are drawn on top
    scan_col = {"XR": "#2a78d6", "S360": "#eb6834", "CS2": "#1baf7a", "3DH": "#6250d6"}
    for j, (bb, (emb, cells)) in enumerate(embeds.items()):
        for i, (key, cmap, title) in enumerate([("y", cls_col, "coloured by class"), ("scanner", scan_col, "coloured by scanner")]):
            ax = axes[i, j]
            for k, col in cmap.items():
                m = (cells[key] == k).to_numpy()
                lab = {1: "mitotic figure", 0: "imposter"}.get(k, k)
                ax.scatter(emb[m, 0], emb[m, 1], s=3, color=col, alpha=0.6, linewidths=0, label=lab, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(AXIS)
            ax.set_title(f"{NAMES.get(bb, bb)}: {title}")
            if j == 0:
                ax.legend(loc="lower left", markerscale=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "umap_features.png"), dpi=150)


if __name__ == "__main__":
    main()
