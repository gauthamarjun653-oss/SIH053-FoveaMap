"""Module A: T-01 (bad .bin size), T-02 (round-trip), T-03 (collision),
AC-A2 (accounting), AC-A3 (determinism)."""

import numpy as np
import pytest

from src.common.config import ProjectionConfig
from src.common.errors import DatasetError
from src.io.kitti_reader import read_scan
from src.projection.range_projection import back_project, project

CFG = ProjectionConfig()


def _synthetic_sweep(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(3.0, 90.0, n)
    az = rng.uniform(-np.pi, np.pi, n)
    el = rng.uniform(np.deg2rad(-24.5), np.deg2rad(2.5), n)
    pts = np.stack([r * np.cos(el) * np.cos(az),
                    r * np.cos(el) * np.sin(az),
                    r * np.sin(el),
                    rng.uniform(0, 1, n)], axis=1).astype(np.float32)
    return pts


def test_t01_bad_bin_size_rejected(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"\x00" * 33)  # not divisible by 16
    with pytest.raises(DatasetError) as e:
        read_scan(p)
    assert "33" in str(e.value)


def test_good_bin_roundtrip(tmp_path):
    pts = _synthetic_sweep(100)
    p = tmp_path / "ok.bin"
    pts.astype(np.float32).tofile(p)
    assert read_scan(p).shape == (100, 4)


def test_t02_project_backproject_median_error_below_1cm():
    pts = _synthetic_sweep()
    res = project(pts, CFG)
    valid = res.pixel_map[:, 0] >= 0
    px = res.pixel_map[valid]
    stored = res.range_image[px[:, 1], px[:, 0], 1:4]
    err = np.linalg.norm(stored - pts[valid, :3], axis=1)
    assert np.median(err) < 0.01


def test_ac_a2_full_accounting():
    pts = _synthetic_sweep(5000)
    pts[:10, :3] = 0.1  # inside range_min -> discarded
    res = project(pts, CFG)
    assert res.n_assigned + res.n_discarded == res.n_points == 5000
    assert res.n_discarded >= 10


def test_t03_collision_nearest_wins_loser_still_labelled():
    d = np.array([1.0, 0.0, 0.0])
    pts = np.array([np.r_[d * 6.0, 0.5], np.r_[d * 5.0, 0.5]], dtype=np.float32)
    res = project(pts, CFG)
    assert res.n_collisions == 1
    u, v = res.pixel_map[0]
    assert (res.pixel_map[1] == (u, v)).all()          # same pixel
    assert res.range_image[v, u, 0] == pytest.approx(5.0, abs=1e-4)  # nearest wins
    label_img = np.zeros((CFG.height, CFG.width), dtype=np.int8)
    label_img[v, u] = 3
    labels = back_project(label_img, res.pixel_map)
    assert labels[0] == 3 and labels[1] == 3           # loser labelled too


def test_ac_a3_deterministic():
    pts = _synthetic_sweep(3000, seed=7)
    a, b = project(pts, CFG), project(pts, CFG)
    assert np.array_equal(a.range_image, b.range_image)
    assert np.array_equal(a.pixel_map, b.pixel_map)
    assert np.array_equal(a.mask, b.mask)
