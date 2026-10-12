"""Reading MIDOG++ regions and tiling them into overlapping patches."""
from __future__ import annotations

import numpy as np
import tifffile


def read_region(path: str) -> np.ndarray:
    """Full-resolution RGB uint8 array (MIDOG++ TIFFs are RGBA, level 0 is page 0)."""
    with tifffile.TiffFile(path) as t:
        img = t.pages[0].asarray()
    return np.ascontiguousarray(img[..., :3])


def read_mpp(path: str) -> float:
    """Microns per pixel from the TIFF resolution tags."""
    with tifffile.TiffFile(path) as t:
        tags = t.pages[0].tags
        num, den = tags["XResolution"].value
        unit = tags["ResolutionUnit"].value
    px_per_unit = num / den
    unit_um = {2: 25400.0, 3: 10000.0}.get(int(unit), 25400.0)  # inch / centimetre
    return unit_um / px_per_unit


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, length) with the last tile flush to the border."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def grid(h: int, w: int, tile: int = 512, stride: int = 448) -> list[tuple[int, int]]:
    return [(x, y) for y in tile_origins(h, tile, stride) for x in tile_origins(w, tile, stride)]


def crop_centered(img: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """size x size crop centred on (x, y) with reflect padding at borders."""
    h, w = img.shape[:2]
    half = size // 2
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    x1, y1 = x0 + size, y0 + size
    px0, py0, px1, py1 = max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if px0 or py0 or px1 or py1:
        crop = np.pad(crop, ((py0, py1), (px0, px1), (0, 0)), mode="reflect")
    return crop
