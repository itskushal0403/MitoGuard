"""MitoGuard v2 end-to-end inference.

region -> two YOLO11 detectors (with / without stain augmentation), candidates merged (union, any class
score >= 0.05) -> LoRA-fine-tuned pathology foundation model on cell-centred crops with 8-view TTA ->
Platt calibration -> learned stacker with YOLO's class scores -> count, conformal 90% count interval,
reliability score and mitotic-count category with a grade-ambiguity flag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .detector import CandidateDetector
from .ft import CellClassifier, predict
from .reliability import region_scores
from .stage2 import apply_stacker, in_pool
from .tiling import crop_centered

CUTS = (7.5, 14.5)
CATEGORY = ["low (<=7)", "intermediate (8-14)", "high (>=15)"]


@dataclass
class Result:
    xy: np.ndarray
    p_yolo: np.ndarray
    p_final: np.ndarray
    kept: np.ndarray
    count: int
    interval: tuple
    category: str
    ambiguous: bool
    reliability: dict


def merge_candidates(dets, radius=8.0):
    from scipy.spatial import cKDTree
    from .detector import distance_nms
    xy = np.concatenate([d["xy"] for d in dets])
    pm = np.concatenate([d["p_mit"] for d in dets])
    pi = np.concatenate([d["p_imp"] for d in dets])
    if len(xy) == 0:
        return {"xy": xy, "p_mit": pm, "p_imp": pi}
    keep = distance_nms(xy, np.maximum(pm, pi), radius)
    tree = cKDTree(xy)
    m = np.array([pm[tree.query_ball_point(xy[k], radius)].max() for k in keep], np.float32)
    i = np.array([pi[tree.query_ball_point(xy[k], radius)].max() for k in keep], np.float32)
    return {"xy": xy[keep], "p_mit": m, "p_imp": i}


class MitoGuardV2:
    def __init__(self, bundle_path: str, device: str = "cuda"):
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        here = os.path.dirname(os.path.abspath(bundle_path))
        dev = device if torch.cuda.is_available() else "cpu"
        self.dev = dev
        self.detectors = [CandidateDetector(os.path.join(here, w), device=dev) for w in b["yolo_weights"]]
        self.clf = CellClassifier(b["backbone"], lora_r=b.get("lora_r", 16), zoom=b.get("zoom", 112))
        self.clf.load_state_dict(b["classifier_state"], strict=False)
        self.clf = self.clf.to(dev).eval()
        self.platt, self.stacker = b["platt"], b.get("stacker")
        self.pool, self.det_thr, self.flag_thr, self.conf_q = b["pool"], b["det_thr"], b["flag_thr"], b["conformal_q"]
        self.tta = b.get("tta", 8)
        self.cfg = b

    def __call__(self, img: np.ndarray, batch: int = 64) -> Result:
        c = merge_candidates([d.detect(img) for d in self.detectors])
        m = in_pool(c, self.pool)
        xy, pm, pi = c["xy"][m], c["p_mit"][m], c["p_imp"][m]
        p = np.zeros(len(xy), np.float32)
        if len(xy):
            crops = torch.from_numpy(np.stack([crop_centered(img, x, y, 224) for x, y in xy]))
            raw = predict(self.clf, (crops[k:k + batch] for k in range(0, len(crops), batch)), self.clf.zoom,
                          tta=self.tta, device=self.dev)
            raw = np.clip(raw, 1e-5, 1 - 1e-5)
            p = self.platt.predict_proba(np.log(raw / (1 - raw))[:, None])[:, 1].astype(np.float32)
            if self.stacker is not None:
                p = apply_stacker(self.stacker, p, pm, pi)
        kept = p >= self.det_thr
        rel = region_scores(p, self.det_thr)
        count = int(kept.sum())
        sigma = float(np.sqrt(rel["u_var"] * max(count, 1)))
        half = self.conf_q * np.sqrt(sigma ** 2 + 1)
        lo, hi = max(0.0, count - half), count + half
        cat = int(np.digitize(count, CUTS))
        ambiguous = any(lo < cut < hi for cut in CUTS)
        return Result(xy, pm, p, kept, count, (lo, hi), CATEGORY[cat], ambiguous, rel)
