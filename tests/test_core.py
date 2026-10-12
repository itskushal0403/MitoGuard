"""Unit tests for dataset-independent building blocks."""
import numpy as np
import pytest

from mitoguard.detector import distance_nms
from mitoguard.metrics import large_error_label, prf
from mitoguard.reliability import region_scores
from mitoguard.stain import hed_jitter
from mitoguard.tiling import crop_centered, grid, tile_origins


def test_tile_origins_cover_and_flush():
    starts = tile_origins(1000, 512, 448)
    assert starts[0] == 0 and starts[-1] == 1000 - 512
    covered = np.zeros(1000, bool)
    for s in starts:
        covered[s:s + 512] = True
    assert covered.all()
    assert tile_origins(300, 512, 448) == [0]


def test_grid_count():
    assert len(grid(5412, 7215)) == len(tile_origins(5412, 512, 448)) * len(tile_origins(7215, 512, 448))


def test_crop_centered_shape_and_border():
    img = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    assert crop_centered(img, 60, 50, 64).shape == (64, 64, 3)
    assert crop_centered(img, 2, 3, 64).shape == (64, 64, 3)  # reflect-padded at the corner
    c = crop_centered(img, 60, 50, 4)
    assert np.array_equal(c, img[48:52, 58:62])


def test_hed_jitter_identity_and_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    same = hed_jitter(img, sigma=0.0, bias=0.0)
    assert np.abs(same.astype(int) - img).max() <= 1  # float round-trip
    out = hed_jitter(img, 0.2, 0.2, rng)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.abs(out.astype(int) - img).mean() > 0


def test_distance_nms_keeps_best_per_cluster():
    xy = np.array([[0, 0], [5, 0], [100, 100], [103, 100]], np.float32)
    s = np.array([0.4, 0.9, 0.7, 0.2])
    keep = set(distance_nms(xy, s, 10).tolist())
    assert keep == {1, 2}


def test_prf():
    p, r, f1 = prf(8, 2, 8)
    assert p == pytest.approx(0.8, abs=1e-5) and r == pytest.approx(0.5, abs=1e-5)
    assert f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3, abs=1e-5)


def test_large_error_label():
    y = large_error_label([10, 13, 0, 1, 5], [10, 10, 0, 0, 20])
    assert y.tolist() == [0, 1, 0, 1, 1]
    assert large_error_label([1], [0], min_abs=3).tolist() == [0]


def test_region_scores():
    p = np.array([0.95, 0.9, 0.5, 0.1, 0.05])
    r = region_scores(p, 0.5)
    assert r["count"] == 3
    assert r["u_var"] == pytest.approx((p * (1 - p)).sum() / 3)
    # confident predictions give lower uncertainty than borderline ones
    assert region_scores(np.array([0.99, 0.98, 0.01]), 0.5)["u_var"] < region_scores(np.array([0.6, 0.55, 0.45]), 0.5)["u_var"]
    members = np.stack([p, p, p], 1)
    assert region_scores(p, 0.5, members)["u_ens"] == 0.0


def test_official_scorer_radius():
    from grand_challenge_metrics.scorers import score_detection
    gt = [(0.0, 0.0), (100.0, 100.0)]
    sc = score_detection(ground_truth=gt, predictions=[(3.0, 4.0), (200.0, 200.0)], radius=30)
    assert (sc.true_positives, sc.false_positives, sc.false_negatives) == (1, 1, 1)
