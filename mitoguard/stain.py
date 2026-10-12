"""H&E stain augmentation in the haematoxylin-eosin-DAB optical-density space.

Each pixel's optical density is unmixed into H, E and residual channels (Ruifrok & Johnston),
each channel is scaled by alpha ~ U(1-s, 1+s) and shifted by beta ~ U(-b, b), then re-mixed.
This mimics differences in staining protocol and scanner colour response far more faithfully
than generic HSV jitter.
"""
from __future__ import annotations

import os

import numpy as np

# Ruifrok & Johnston stain vectors (rows: H, E, DAB), same as skimage.color.hed_from_rgb inverse
_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
_RGB_FROM_HED /= np.linalg.norm(_RGB_FROM_HED, axis=1, keepdims=True)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


_OD_LUT = (-np.log((np.arange(256, dtype=np.float32) + 1.0) / 256.0)).astype(np.float32)


def hed_jitter(img: np.ndarray, sigma: float = 0.05, bias: float = 0.05, rng=None) -> np.ndarray:
    """img: HxWx3 uint8 RGB. Returns augmented uint8 RGB image.

    The unmix -> scale/shift -> remix chain is linear in optical density, so it collapses into a
    single 3x3 matrix A and offset b:  od' = od @ A + b.  Applied channel-wise without BLAS so it
    stays fast inside many dataloader workers.
    """
    rng = rng or np.random.default_rng()
    alpha = rng.uniform(1 - sigma, 1 + sigma, size=3).astype(np.float32)
    beta = rng.uniform(-bias, bias, size=3).astype(np.float32)
    A = (_HED_FROM_RGB * alpha[None, :]) @ _RGB_FROM_HED
    b = beta @ _RGB_FROM_HED
    od = _OD_LUT[img]
    out = np.empty(img.shape, dtype=np.uint8)
    for c in range(3):
        odc = od[..., 0] * A[0, c] + od[..., 1] * A[1, c] + od[..., 2] * A[2, c] + b[c]
        out[..., c] = np.clip(256.0 * np.exp(-odc) - 1.0, 0, 255)
    return out


class HEDJitterTransform:
    """Ultralytics-compatible transform: operates on labels dict with 'img' (BGR uint8)."""

    def __init__(self, p: float = 0.8, sigma: float = 0.05, bias: float = 0.05):
        self.p, self.sigma, self.bias = p, sigma, bias
        self._pid, self.rng = None, None

    def __call__(self, labels):
        if self._pid != os.getpid():  # fresh entropy in every dataloader worker after fork
            self._pid, self.rng = os.getpid(), np.random.default_rng()
        if self.rng.random() < self.p:
            img = labels["img"]
            rgb = img[..., ::-1]
            labels["img"] = np.ascontiguousarray(hed_jitter(rgb, self.sigma, self.bias, self.rng)[..., ::-1])
        return labels
