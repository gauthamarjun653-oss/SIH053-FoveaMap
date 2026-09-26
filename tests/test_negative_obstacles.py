"""T-15 (occlusion excludes flagging) and T-16 (synthetic depression detected,
confidence in [0,1]); plus the offline ground-truth deriver on known geometry."""

import numpy as np

from src.analytics.negative_obs import (derive_ground_truth_depressions,
                                        negative_obstacle_candidates)
from src.common.config import load_rings
from src.grid.foveated_grid import build_grid
from src.segmentation.class_map import ROAD, DYNAMIC

RINGS = load_rings("configs/rings.yaml")
HOLE = np.array([8.0, 0.0])
HOLE_R = 0.3


def _road_scene(with_hole=True, with_vehicle=False):
    """Dense flat road out to 12 m, optional 0.3 m-radius hole at (8, 0),
    optional tall vehicle at 6 m in front of the hole."""
    xs = np.arange(2.6, 12.0, 0.04)
    ys = np.arange(-6.0, 6.0, 0.04)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.7)], axis=1)
    if with_hole:
        d = np.hypot(pts[:, 0] - HOLE[0], pts[:, 1] - HOLE[1])
        pts = pts[d > HOLE_R]
    labels = np.full(pts.shape[0], ROAD, dtype=np.int8)
    if with_vehicle:
        rng = np.random.default_rng(0)
        n = 600
        veh = np.stack([rng.uniform(5.8, 6.2, n), rng.uniform(-0.6, 0.6, n),
                        rng.uniform(-1.5, 0.6, n)], axis=1)
        pts = np.concatenate([pts, veh])
        labels = np.concatenate([labels, np.full(n, DYNAMIC, dtype=np.int8)])
    return pts, labels


def _missing_return_candidates(cands):
    return [c for c in cands if c.kind == "missing_return"]


def test_t16_synthetic_depression_detected_with_valid_confidence():
    pts, labels = _road_scene(with_hole=True)
    grid = build_grid(pts, labels, RINGS)
    cands = negative_obstacle_candidates(grid, RINGS, max_range=12.0)
    assert all(0.0 <= c.confidence <= 1.0 for c in cands)  # never hard classifications
    near_hole = [c for c in _missing_return_candidates(cands)
                 if np.hypot(c.x - HOLE[0], c.y - HOLE[1]) <= HOLE_R + 0.1]
    assert near_hole, "the known depression was not flagged"


def test_t15_cell_behind_classified_vehicle_not_flagged():
    pts, labels = _road_scene(with_hole=True, with_vehicle=True)
    grid = build_grid(pts, labels, RINGS)
    cands = negative_obstacle_candidates(grid, RINGS, max_range=12.0)
    behind = [c for c in _missing_return_candidates(cands)
              if np.hypot(c.x - HOLE[0], c.y - HOLE[1]) <= HOLE_R + 0.1]
    assert not behind, "occlusion by a classified vehicle must be ruled out first"


def test_no_planes_no_guessing():
    # Obstacle-only scene: no ROAD cells -> no plane -> no candidates, no crash.
    rng = np.random.default_rng(2)
    pts = np.stack([rng.uniform(3, 9, 200), rng.uniform(-2, 2, 200),
                    rng.uniform(-1, 1, 200)], axis=1)
    grid = build_grid(pts, np.full(200, DYNAMIC, dtype=np.int8), RINGS)
    assert negative_obstacle_candidates(grid, RINGS, max_range=12.0) == []


def test_ground_truth_deriver_known_geometry():
    rng = np.random.default_rng(1)
    n = 200_000
    pts = np.stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n),
                    np.zeros(n)], axis=1)
    # A 0.6 x 0.6 m, 25 cm-deep depression centred at (10, 10):
    in_dep = (np.abs(pts[:, 0] - 10) < 0.3) & (np.abs(pts[:, 1] - 10) < 0.3)
    pts[in_dep, 2] = -0.25
    instances = derive_ground_truth_depressions(pts, depth_threshold=0.15,
                                                max_extent_m=1.0)
    assert len(instances) == 1
    inst = instances[0]
    assert np.hypot(inst.centroid[0] - 10, inst.centroid[1] - 10) < 0.3
    assert inst.depth >= 0.2
    assert 0.0 <= inst.confidence <= 1.0


def test_phantom_points_below_ground_are_not_depressions():
    # Mirror-reflection phantoms sit tens of meters below ground (observed on
    # seq 00: down to -29 m). They must be excluded, not reported as absurdly
    # deep depressions — while a real 25 cm depression is still found.
    rng = np.random.default_rng(3)
    n = 200_000
    pts = np.stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n),
                    np.zeros(n)], axis=1)
    in_dep = (np.abs(pts[:, 0] - 10) < 0.3) & (np.abs(pts[:, 1] - 10) < 0.3)
    pts[in_dep, 2] = -0.25
    phantoms = np.array([[3.0, 3.0, -28.0], [15.0, 5.0, -22.0], [7.0, 17.0, -12.0]])
    instances = derive_ground_truth_depressions(
        np.concatenate([pts, phantoms]), depth_threshold=0.15, max_extent_m=1.0)
    assert len(instances) == 1
    assert instances[0].depth < 1.0
