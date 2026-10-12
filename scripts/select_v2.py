"""Model selection for MitoGuard v2 without touching a fold's own held-out domain.

Rule A (primary): nested leave-one-domain-out selection (DomainBed-style, Gulrajani & Lopez-Paz, ICLR 2021).
  For fold k of a protocol, pick the configuration with the best mean held-out F1 on the *other* folds of
  that protocol, then report its F1 on fold k. Fold k's held-out labels never influence its own choice.
Rule B (reported for transparency): in-domain validation selection, i.e. the configuration with the best
  mean validation F1 (training-domain cases). Known to favour configurations that lean on in-domain cues.

Two families are selected separately: "no adaptation" (pure domain generalisation) and "site-adapted"
(self-training on the unlabelled target site, backbone names ending in '+st').
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("MITOGUARD_RESULTS", os.path.join(ROOT, "results"))
PATTERN = re.compile(r"^yolo\+ft-(?P<bb>[a-z0-9]+)(?P<st>\+st)?@(?P<pool>a05|m05i10)(?P<mode>\+fused|\+stack)?(?P<sfx>_union)?$")


def family(system):
    m = PATTERN.match(system)
    if not m:
        return None
    return "site-adapted" if m.group("st") else "no adaptation"


def nested(folds, protocol, systems):
    f = folds[(folds.protocol == protocol) & folds.system.isin(systems)]
    piv = f.pivot_table(index="fold", columns="system", values="f1")
    piv = piv.dropna(axis=1)  # only configurations available on every fold
    out = []
    for k in piv.index:
        others = piv.drop(index=k).mean()
        best = others.idxmax()
        out.append({"fold": k, "selected": best, "f1": piv.loc[k, best]})
    return pd.DataFrame(out)


def main():
    folds = pd.read_csv(os.path.join(RES, "folds.csv"))
    lines = ["# MitoGuard v2 model selection", "", __doc__.strip(), ""]
    summary = []
    for protocol in ("loto", "loso"):
        base = folds[(folds.protocol == protocol) & (folds.system == "yolo")].set_index("fold").f1
        for fam in ("no adaptation", "site-adapted"):
            systems = [s for s in folds.system.unique() if family(s) == fam]
            if not systems:
                continue
            sel = nested(folds, protocol, systems)
            if sel.empty:
                continue
            sel["yolo"] = sel.fold.map(base)
            sel["gain"] = sel.f1 - sel.yolo
            f = folds[(folds.protocol == protocol) & folds.system.isin(systems)]
            vbest = f.groupby("system").val_f1.mean().idxmax()
            vf1 = f[f.system == vbest].set_index("fold").f1
            summary.append({"protocol": protocol, "family": fam, "rule": "nested LODO (primary)",
                            "mean F1": sel.f1.mean(), "YOLO11": sel.yolo.mean(), "gain": sel.gain.mean(),
                            "folds beating YOLO": f"{int((sel.gain > 0).sum())}/{len(sel)}"})
            summary.append({"protocol": protocol, "family": fam, "rule": f"in-domain val ({vbest})",
                            "mean F1": vf1.mean(), "YOLO11": base.reindex(vf1.index).mean(),
                            "gain": (vf1 - base.reindex(vf1.index)).mean(),
                            "folds beating YOLO": f"{int(((vf1 - base.reindex(vf1.index)) > 0).sum())}/{len(vf1)}"})
            lines += [f"## {protocol.upper()} / {fam}: nested selection per fold", "", sel.round(3).to_markdown(index=False), ""]
    lines = lines[:4] + ["## Summary", "", pd.DataFrame(summary).round(3).to_markdown(index=False), ""] + lines[4:]
    # materialise the nested-selected systems per region, for bootstrap CIs and clinical metrics
    cases = pd.read_csv(os.path.join(RES, "cases.csv.gz"))
    picked = []
    for protocol in ("loto", "loso"):
        for fam, label in (("no adaptation", "MitoGuard-v2"), ("site-adapted", "MitoGuard-v2-SA")):
            systems = [s for s in folds.system.unique() if family(s) == fam]
            sel = nested(folds, protocol, systems) if systems else pd.DataFrame()
            for r in sel.itertuples():
                c = cases[(cases.protocol == protocol) & (cases.fold == r.fold) & (cases.system == r.selected)].copy()
                c["selected_system"], c["system"] = r.selected, label
                picked.append(c)
    if picked:
        pd.concat(picked).to_csv(os.path.join(RES, "v2_selected_cases.csv.gz"), index=False)
    open(os.path.join(RES, "selection_v2.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
