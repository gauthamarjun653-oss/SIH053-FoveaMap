"""T-10 / AC-C6: Welford combine against hand-computed expected values."""

import numpy as np
import pytest

from src.grid.foveated_grid import Cell, combine, update, variance, zero_cell


def _fill(values, cls=0):
    c = Cell()
    for v in values:
        update(c, v, cls)
    return c


def test_t10_combine_two_known_groups_matches_union_variance():
    a = _fill([1.0, 2.0, 3.0])
    b = _fill([10.0, 20.0])
    c = combine(a, b)
    # Union {1,2,3,10,20}: mean 7.2, sum of squared deviations 254.8 (by hand):
    # 6.2^2 + 5.2^2 + 4.2^2 + 2.8^2 + 12.8^2 = 38.44+27.04+17.64+7.84+163.84
    assert c.count == 5
    assert c.mean == pytest.approx(7.2, abs=1e-12)
    assert c.M2 == pytest.approx(254.8, abs=1e-9)
    assert variance(c) == pytest.approx(50.96, abs=1e-9)
    assert c.z_min == 1.0 and c.z_max == 20.0


def test_combine_with_empty_is_identity():
    a = _fill([4.0, 6.0], cls=2)
    c = combine(a, zero_cell())
    assert c.count == 2 and c.mean == a.mean and c.M2 == a.M2
    c2 = combine(zero_cell(), zero_cell())
    assert c2.count == 0


def test_variance_never_nan():
    assert variance(zero_cell()) == 0.0
    assert variance(_fill([3.3])) == 0.0  # single point: variance 0, not NaN


def test_histogram_counts_are_summed_never_voted():
    a = _fill([0.0] * 3, cls=1)
    b = _fill([0.0], cls=3)
    c = combine(a, b)
    assert list(c.hist) == [0, 3, 0, 1]  # the single DYNAMIC point survives
