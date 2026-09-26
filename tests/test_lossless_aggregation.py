"""AC-C1 / T-08: lossless aggregation, every ring pair, BIT-IDENTICAL.

This is the primary evidence for the project's strongest claim; it runs in CI
and blocks merge. Also covers T-11 (boundary ring assignment), T-12 (empty
scene), T-13 (single point), T-24 (memory ceiling abort).
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.common.config import RingConfig, RingSpec, load_rings
from src.common.errors import MemoryCeilingExceeded
from src.grid.foveated_grid import (build_grid, block_to_size,
                                    cells_bit_identical, N_CLASSES)
from src.grid.rings import assign_rings, assign_ring_scalar

RINGS = load_rings("configs/rings.yaml")
SIZES = [r.cell_size for r in RINGS.rings]
ALL_RING_PAIRS = [(a, b) for a in SIZES for b in SIZES if a < b]


def single_ring_config(cell_size: float) -> RingConfig:
    return RingConfig(base_cell_size=RINGS.base_cell_size,
                      rings=(RingSpec(id=1, radius_min=0.0, radius_max=200.0,
                                      cell_size=cell_size),))


def _grid(points: np.ndarray, cell_size: float):
    labels = (np.arange(points.shape[0]) % N_CLASSES).astype(np.int8)
    return build_grid(points, labels, single_ring_config(cell_size))


def _assert_lossless(points: np.ndarray, fine: float, coarse: float):
    fine_grid = _grid(points, fine)
    native = _grid(points, coarse)
    if 1 not in fine_grid.blocks:
        assert 1 not in native.blocks or len(native.blocks[1]) == 0
        return
    aggregated = block_to_size(fine_grid.blocks[1], coarse)
    assert cells_bit_identical(aggregated, native.blocks[1])


point_clouds = st.lists(
    st.tuples(st.floats(-99, 99, allow_nan=False, allow_infinity=False),
              st.floats(-99, 99, allow_nan=False, allow_infinity=False),
              st.floats(-10, 30, allow_nan=False, allow_infinity=False)),
    min_size=0, max_size=200,
)


@settings(max_examples=60, deadline=None)
@given(points=point_clouds, ring_pair=st.sampled_from(ALL_RING_PAIRS))
def test_lossless_aggregation(points, ring_pair):
    fine, coarse = ring_pair
    pts = np.array(points, dtype=np.float64).reshape(-1, 3)
    _assert_lossless(pts, fine, coarse)


@pytest.mark.parametrize("fine,coarse", ALL_RING_PAIRS)
def test_lossless_degenerate_cases(fine, coarse):
    cases = [
        np.zeros((0, 3)),                                    # empty
        np.array([[1.234, -5.678, 0.3]]),                    # single point
        np.tile([[3.001, 3.002, -1.7]], (50, 1)),            # all in one cell
        np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 1.0],        # exactly on ring boundary
                  [2.5, 0.0, -1.0], [25.0, 0.0, 2.0]]),
        np.array([[0.0, 0.0, 0.0]]),                         # rho = 0
        np.array([[-0.025, -0.025, 0.5], [0.025, 0.025, 0.5]]),  # negative cells
    ]
    for pts in cases:
        _assert_lossless(pts, fine, coarse)


def test_t11_boundary_point_goes_to_inner_ring():
    ids = assign_rings(np.array([10.0, 25.0, 50.0]), np.zeros(3), RINGS)
    assert list(ids) == [1, 2, 3]
    assert assign_ring_scalar(10.0, RINGS) == 1


def test_t12_empty_scene_memory_approx_zero():
    grid = build_grid(np.zeros((0, 3)), np.zeros(0, dtype=np.int8), RINGS)
    assert grid.n_cells == 0
    assert grid.memory_bytes() == 0


def test_t13_single_point_one_cell_no_nan():
    pts = np.array([[5.0, 1.0, -1.7]])
    grid = build_grid(pts, np.array([1], dtype=np.int8), RINGS)
    assert grid.n_cells == 1
    block = grid.blocks[1]
    assert block.count[0] == 1
    assert block.m2[0] == 0.0
    assert not np.isnan(block.mean).any() and not np.isnan(block.m2).any()


def test_ac_c3_discards_counted():
    pts = np.array([[5.0, 0.0, 0.0], [150.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    grid = build_grid(pts, np.zeros(3, dtype=np.int8), RINGS)
    assert grid.assigned_points == 1
    assert grid.discarded_points == 2
    assert grid.assigned_points + grid.discarded_points == 3


def test_t24_memory_ceiling_aborts_loudly():
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(3, 90, 5000), rng.uniform(-5, 5, 5000),
                    rng.uniform(-2, 2, 5000)], axis=1)
    with pytest.raises(MemoryCeilingExceeded) as e:
        build_grid(pts, np.zeros(5000, dtype=np.int8), RINGS, memory_ceiling_bytes=10)
    assert e.value.occupied_cells > 0
    assert "cells" in str(e.value)
