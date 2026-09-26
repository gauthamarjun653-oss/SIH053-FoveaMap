"""T-14 / AC-D4 (safety): 99 ROAD + 1 DYNAMIC surfaces DYNAMIC in both payload
and render. Plus AC-D1: no NaN in any stats field."""

import numpy as np

from src.analytics.cell_stats import compute_stats, render_class
from src.common.config import load_rings
from src.grid.foveated_grid import build_grid
from src.segmentation.class_map import ROAD, DYNAMIC

RINGS = load_rings("configs/rings.yaml")


def _one_cell_scene():
    """100 points inside one 5 cm R1 cell at ~5 m: 99 ROAD + 1 DYNAMIC."""
    rng = np.random.default_rng(0)
    pts = np.stack([5.0 + rng.uniform(0, 0.049, 100),
                    rng.uniform(0, 0.049, 100),
                    np.full(100, -1.7)], axis=1)
    pts[0, 2] = -1.0  # the pedestrian return sits higher
    labels = np.full(100, ROAD, dtype=np.int8)
    labels[0] = DYNAMIC
    return pts, labels


def test_t14_minority_dynamic_survives_payload_and_render():
    pts, labels = _one_cell_scene()
    grid = build_grid(pts, labels, RINGS)
    assert grid.n_cells == 1
    block = grid.blocks[1]
    # Payload: counts, never a majority vote — the pedestrian exists.
    assert block.hist[0, ROAD] == 99
    assert block.hist[0, DYNAMIC] == 1
    # Render: highest-severity class present, not majority.
    assert render_class(block.hist, min_count=1)[0] == DYNAMIC


def test_ac_d1_stats_no_nan_and_bounded():
    rng = np.random.default_rng(3)
    n = 5000
    r = rng.uniform(2.5, 99, n)
    az = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([r * np.cos(az), r * np.sin(az), rng.normal(-1.5, 0.5, n)], axis=1)
    labels = rng.integers(0, 4, n).astype(np.int8)
    grid = build_grid(pts, labels, RINGS)
    compute_stats(grid)  # asserts no NaN internally (AC-D1)
    for block in grid.blocks.values():
        assert (block.z_var >= 0).all()
        assert ((block.occupancy_conf >= 0) & (block.occupancy_conf <= 1)).all()


def test_ac_d2_variance_separates_flat_road_from_kerb():
    """Two cells with the same mean height: flat road vs a kerb step. The
    variance is the load-bearing separator."""
    flat = np.stack([np.full(40, 5.01), np.linspace(0, 0.049, 40), np.full(40, -1.7)], axis=1)
    kerb_z = np.r_[np.full(20, -1.8), np.full(20, -1.6)]  # same mean -1.7, 20 cm step
    kerb = np.stack([np.full(40, 6.01), np.linspace(0, 0.049, 40), kerb_z], axis=1)
    labels = np.ones(40, dtype=np.int8)
    g_flat = build_grid(flat, labels, RINGS)
    g_kerb = build_grid(kerb, labels, RINGS)
    compute_stats(g_flat)
    compute_stats(g_kerb)
    v_flat = g_flat.blocks[1].z_var[0]
    v_kerb = g_kerb.blocks[1].z_var[0]
    assert abs(g_flat.blocks[1].mean[0] - g_kerb.blocks[1].mean[0]) < 1e-9
    assert v_kerb > 100 * max(v_flat, 1e-12)


def test_ac_d2_kerb_variance_separates_from_road():
    # Synthetic street: flat road at z=0 (raw 40), sidewalk 12 cm up (raw 48).
    # Cells straddling the boundary must show higher z-variance than
    # road-interior cells.
    from src.analytics.cell_stats import kerb_road_variance_cells
    rng = np.random.default_rng(0)
    n = 120_000
    x = rng.uniform(3.0, 9.0, n)
    y = rng.uniform(-6.0, 6.0, n)
    # Boundary deliberately OFF the 5 cm lattice: a boundary on a cell edge
    # produces zero straddling cells (no cell sees both classes).
    sidewalk = y > 0.017
    z = np.where(sidewalk, 0.12, 0.0) + rng.normal(0, 0.008, n)
    raw = np.where(sidewalk, 48, 40).astype(np.int32)
    road_var, kerb_var, cells = kerb_road_variance_cells(
        x, y, z, raw, cell_size=0.05, r_min=2.5, r_max=10.0)
    assert road_var.size > 100 and kerb_var.size > 10
    assert np.median(kerb_var) > 5 * np.median(road_var)
    assert all(c >= 4 for _, _, _, c in cells)


def test_negative_candidates_stats_out_counters():
    from src.analytics.negative_obs import negative_obstacle_candidates
    rng = np.random.default_rng(1)
    n = 60_000
    r = rng.uniform(2.6, 12.0, n)
    az = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([r * np.cos(az), r * np.sin(az),
                    rng.normal(-1.7, 0.01, n)], axis=1)
    labels = np.full(n, ROAD, dtype=np.int8)
    grid = build_grid(pts, labels, RINGS)
    stats = {}
    negative_obstacle_candidates(grid, RINGS, stats_out=stats)
    assert set(stats) == {"below_plane", "missing_flagged",
                          "occlusion_suppressed", "no_plane"}
    assert all(isinstance(v, int) and v >= 0 for v in stats.values())
