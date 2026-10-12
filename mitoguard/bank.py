"""Access to the shared crop bank: every labelled cell and Stage 1 candidate embedded once per backbone."""
from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.spatial import cKDTree

from .data import WORK_ROOT


class Bank:
    def __init__(self, backbone: str, root: str = WORK_ROOT):
        d = os.path.join(root, "bank")
        self.xy = pickle.load(open(os.path.join(d, "bank.pkl"), "rb"))
        self.off = pickle.load(open(os.path.join(d, "offsets.pkl"), "rb"))
        self.feats = np.load(os.path.join(d, f"{backbone}.npy"), mmap_mode="r")
        self.trees = {}

    def rows(self, cid: int, xy: np.ndarray) -> np.ndarray:
        """Global feature-row indices of the bank points nearest to `xy` (always within 4 px)."""
        if len(xy) == 0:
            return np.zeros(0, np.int64)
        if cid not in self.trees:
            self.trees[cid] = cKDTree(self.xy[cid])
        d, i = self.trees[cid].query(xy)
        assert d.max() <= 4.01, (cid, d.max())
        return self.off[cid][0] + i

    def get(self, rows: np.ndarray) -> np.ndarray:
        """float32 features for arbitrary row order (sorted reads from the memmap)."""
        rows = np.asarray(rows)
        if len(rows) == 0:
            return np.zeros((0, self.feats.shape[1]), np.float32)
        order = np.argsort(rows)
        out = np.empty((len(rows), self.feats.shape[1]), np.float32)
        out[order] = self.feats[rows[order]]
        return out
