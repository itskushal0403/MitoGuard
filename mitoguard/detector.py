"""Stage 1: sliding-window YOLO candidate detection over a full region of interest.

Unlike the default Ultralytics predictor (which keeps only each box's arg-max class), every
candidate keeps both class probabilities (mitosis, imposter).  Stage 2 can therefore rescue a true
mitosis that YOLO mislabelled as an imposter, and the YOLO-only baseline simply thresholds p_mitosis.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision
from scipy.spatial import cKDTree

from .tiling import grid


def distance_nms(xy: np.ndarray, score: np.ndarray, radius: float) -> np.ndarray:
    """Greedy NMS on point detections: keep the highest score, drop others within `radius`."""
    if len(xy) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-score)
    tree = cKDTree(xy)
    suppressed = np.zeros(len(xy), dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed[tree.query_ball_point(xy[i], radius)] = True
    return np.array(keep, dtype=np.int64)


class CandidateDetector:
    def __init__(self, weights: str, device: str = "cuda", tile: int = 512, stride: int = 448,
                 min_score: float = 0.02, batch: int = 32):
        from ultralytics import YOLO

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        net = YOLO(weights).model.fuse().to(self.device).eval()
        self.net = net.half() if self.half else net.float()
        self.tile, self.stride, self.min_score, self.batch = tile, stride, min_score, batch

    @torch.no_grad()
    def _forward(self, tiles: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(tiles).to(self.device).permute(0, 3, 1, 2)
        x = (x.half() if self.half else x.float()) / 255.0  # RGB in [0, 1], as Ultralytics expects
        out = self.net(x)
        return (out[0] if isinstance(out, (tuple, list)) else out).float()  # (B, 4 + nc, N)

    @torch.no_grad()
    def detect(self, img: np.ndarray) -> dict:
        """img: HxWx3 RGB uint8. Returns dict(xy (N,2), p_mit (N,), p_imp (N,)) in level-0 pixels."""
        h, w = img.shape[:2]
        origins = grid(h, w, self.tile, self.stride)
        xy_all, pm_all, pi_all = [], [], []
        for b in range(0, len(origins), self.batch):
            org = origins[b:b + self.batch]
            tiles = np.stack([img[y:y + self.tile, x:x + self.tile] for x, y in org])
            pred = self._forward(tiles)
            for k, (x0, y0) in enumerate(org):
                p = pred[k]  # (4 + nc, N)
                cls = p[4:6]
                score = cls.max(0).values
                m = score >= self.min_score
                if not m.any():
                    continue
                boxes, cls, score = p[:4, m].T, cls[:, m].T, score[m]
                xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)
                keep = torchvision.ops.nms(xyxy, score, 0.5)
                c = boxes[keep, :2].cpu().numpy()
                # drop detections in the outer margin of a tile; the overlapping neighbour sees them centred
                margin = (self.tile - self.stride) / 2
                inner = np.ones(len(c), dtype=bool)
                if x0 > 0:
                    inner &= c[:, 0] >= margin
                if y0 > 0:
                    inner &= c[:, 1] >= margin
                if x0 + self.tile < w:
                    inner &= c[:, 0] < self.tile - margin
                if y0 + self.tile < h:
                    inner &= c[:, 1] < self.tile - margin
                cl = cls[keep].cpu().numpy()[inner]
                xy_all.append(c[inner] + [x0, y0])
                pm_all.append(cl[:, 0])
                pi_all.append(cl[:, 1])
        if not xy_all:
            z = np.zeros(0, dtype=np.float32)
            return {"xy": np.zeros((0, 2), np.float32), "p_mit": z, "p_imp": z}
        xy, pm, pi = np.concatenate(xy_all), np.concatenate(pm_all), np.concatenate(pi_all)
        keep = distance_nms(xy, np.maximum(pm, pi), radius=20.0)
        return {"xy": xy[keep].astype(np.float32), "p_mit": pm[keep].astype(np.float32), "p_imp": pi[keep].astype(np.float32)}
