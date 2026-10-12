"""End-to-end MitoGuard inference: region -> mitoses, rejected imposters, count, reliability flag."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .detector import CandidateDetector
from .features import Backbone
from .reliability import region_scores
from .stage2 import ImposterFilter
from .tiling import crop_centered


@dataclass
class Result:
    xy: np.ndarray  # (N, 2) candidate centres
    p_yolo: np.ndarray  # Stage 1 mitosis probability
    p_filter: np.ndarray  # Stage 2 calibrated probability
    kept: np.ndarray  # bool, counted as mitotic figure
    count: int
    reliability: dict  # u_var etc.
    flagged: bool


class MitoGuard:
    """bundle: dict saved by scripts/export_bundle.py."""

    def __init__(self, bundle_path: str, device: str = "cuda"):
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.cfg = b
        dev = device if torch.cuda.is_available() else "cpu"
        yolo = b["yolo_weights"]
        if not os.path.isabs(yolo):
            yolo = os.path.join(os.path.dirname(os.path.abspath(bundle_path)), yolo)
        self.detector = CandidateDetector(yolo, device=dev)
        self.backbone = Backbone(b["backbone"], device=dev)
        self.filter = ImposterFilter.from_state(b["filter"], device=dev)
        self.pool_thr, self.det_thr, self.flag_thr = b["pool_thr"], b["det_thr"], b["flag_thr"]

    def __call__(self, img: np.ndarray, batch: int = 128) -> Result:
        c = self.detector.detect(img)
        pool = c["p_mit"] >= self.pool_thr
        xy, p_yolo = c["xy"][pool], c["p_mit"][pool]
        p2 = np.zeros(len(xy), np.float32)
        members = np.zeros((len(xy), self.filter.n_models), np.float32)
        if len(xy):
            feats = []
            for k in range(0, len(xy), batch):
                crops = np.stack([crop_centered(img, x, y, 224) for x, y in xy[k:k + batch]])
                feats.append(self.backbone(crops))
            p2, members = self.filter.predict(np.concatenate(feats))
        rel = region_scores(p2, self.det_thr, members)
        kept = p2 >= self.det_thr
        return Result(xy, p_yolo, p2, kept, int(kept.sum()), rel, bool(rel["u_var"] >= self.flag_thr))

