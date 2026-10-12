"""MIDOG++ metadata, annotations and cross-validation folds.

Scanner/tumor confounding in MIDOG++ (see docs/data_audit.md):
  - "Hamammatsu XR" in the official CSV is a typo of "Hamamatsu XR" (same lab, same device),
    so there are 4 physical scanner models, not 5.
  - 3D Histech only ever scans canine lung / lymphoma / soft-tissue sarcoma, Hamamatsu S360
    only human breast, etc.  Scanner and tumor type are therefore strongly linked and, as the
    project brief prescribes, leave-one-tumor-out (LOTO) is the primary protocol while
    leave-one-scanner-out (LOSO) is reported as secondary.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("MIDOG_ROOT", "/scratch/m25csa016/midogpp")
WORK_ROOT = os.environ.get("MITOGUARD_WORK", "/scratch/m25csa016/mitoguard_work")

MITOSIS, IMPOSTER = 1, 2  # category ids in MIDOG++.json
MATCH_RADIUS_UM = 7.5  # official MIDOG matching radius (7.5 micrometres)

TUMOR_SHORT = {
    "human breast cancer": "HBC",
    "canine lung cancer": "CLC",
    "canine lymphoma": "CL",
    "canine cutaneous mast cell tumor": "CCMCT",
    "human neuroendocrine tumor": "HNET",
    "human melanoma": "HMEL",
    "canine soft tissue sarcoma": "CSTS",
}
SCANNER_FIX = {"Hamammatsu XR": "Hamamatsu XR"}
SCANNER_SHORT = {"Hamamatsu XR": "XR", "Hamamatsu S360": "S360", "Aperio CS2": "CS2", "3D Histech": "3DH"}


@lru_cache(maxsize=1)
def load_cases(root: str = DATA_ROOT) -> pd.DataFrame:
    """One row per case: slide id, file, tumor, scanner, lab, species, official split, size, spacing."""
    df = pd.read_csv(os.path.join(root, "datasets_xvalidation.csv"), sep=";")
    df = df.rename(columns=str.lower).rename(columns={"slide": "case_id", "dataset": "official_split"})
    df["scanner"] = df["scanner"].replace(SCANNER_FIX)
    df["tumor_short"] = df["tumor"].map(TUMOR_SHORT)
    df["scanner_short"] = df["scanner"].map(SCANNER_SHORT)
    df["domain"] = df["tumor_short"] + "@" + df["scanner_short"]
    meta = json.load(open(os.path.join(root, "MIDOG++.json")))
    imgs = pd.DataFrame(meta["images"]).rename(columns={"id": "case_id"})
    df = df.merge(imgs[["case_id", "file_name", "width", "height"]], on="case_id", how="left")
    spacing_file = os.path.join(root, "meta", "spacing.csv")
    if os.path.exists(spacing_file):
        df = df.merge(pd.read_csv(spacing_file), on="case_id", how="left")
    else:
        df["mpp"] = 0.25
    return df.set_index("case_id", drop=False)


@lru_cache(maxsize=1)
def load_annotations(root: str = DATA_ROOT) -> pd.DataFrame:
    """One row per labelled cell: case_id, x, y (centre, level-0 pixels), label (1 mitosis / 2 imposter)."""
    meta = json.load(open(os.path.join(root, "MIDOG++.json")))
    rows = []
    for a in meta["annotations"]:
        x1, y1, x2, y2 = a["bbox"]
        rows.append((a["image_id"], (x1 + x2) / 2, (y1 + y2) / 2, a["category_id"]))
    return pd.DataFrame(rows, columns=["case_id", "x", "y", "label"])


def cells_of(case_id: int, label: int | None = None) -> np.ndarray:
    ann = load_annotations()
    sel = ann[ann.case_id == case_id]
    if label is not None:
        sel = sel[sel.label == label]
    return sel[["x", "y"]].to_numpy(dtype=np.float32)


def radius_px(case_id: int) -> float:
    return MATCH_RADIUS_UM / float(load_cases().loc[case_id, "mpp"])


def image_path(case_id: int, root: str = DATA_ROOT) -> str:
    return os.path.join(root, "images", f"{case_id:03d}.tiff")


# --------------------------------------------------------------------------- folds
@dataclass
class Fold:
    protocol: str  # loto | loso | indomain
    name: str  # held-out domain name
    train: list  # case ids used to fit models
    val: list  # case ids from training domains: model selection + threshold tuning
    test: list  # held-out case ids

    @property
    def tag(self) -> str:
        return f"{self.protocol}_{self.name}"


def _inner_split(case_ids, strat, frac=0.2, seed=0):
    """Case-level train/val split stratified by `strat` (domain)."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for _, grp in pd.Series(case_ids, index=case_ids).groupby(strat):
        ids = rng.permutation(grp.to_numpy())
        n_val = max(1, int(round(frac * len(ids))))
        val += ids[:n_val].tolist()
        train += ids[n_val:].tolist()
    return sorted(train), sorted(val)


def make_folds(protocol: str) -> list[Fold]:
    cases = load_cases()
    folds = []
    if protocol == "indomain":
        tr_ids = cases[cases.official_split == "train"].case_id.to_numpy()
        te_ids = cases[cases.official_split == "test"].case_id.tolist()
        tr, va = _inner_split(tr_ids, cases.loc[tr_ids, "domain"].to_numpy())
        return [Fold("indomain", "all", tr, va, sorted(te_ids))]
    key = {"loto": "tumor_short", "loso": "scanner_short"}[protocol]
    for held in sorted(cases[key].unique()):
        te_ids = cases[cases[key] == held].case_id.tolist()
        rest = cases[cases[key] != held]
        tr, va = _inner_split(rest.case_id.to_numpy(), rest["domain"].to_numpy())
        folds.append(Fold(protocol, held, tr, va, sorted(te_ids)))
    return folds


def get_fold(tag: str) -> Fold:
    protocol, name = tag.split("_", 1)
    for f in make_folds(protocol):
        if f.name == name:
            return f
    raise KeyError(tag)


def all_fold_tags() -> list[str]:
    return [f.tag for p in ("loto", "loso", "indomain") for f in make_folds(p)]
