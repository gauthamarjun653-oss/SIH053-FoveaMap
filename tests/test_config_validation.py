"""T-09 / AC-C4 / AC-C5: ring table validation at startup."""

import pytest

from src.common.config import RingSpec, load_rings, validate_rings
from src.common.errors import ConfigError


def _ring(id, lo, hi, size):
    return RingSpec(id=id, radius_min=lo, radius_max=hi, cell_size=size)


def test_spec_table_passes(tmp_path):
    cfg = load_rings("configs/rings.yaml")
    assert [r.cell_size for r in cfg.rings] == [0.05, 0.10, 0.20, 0.40]
    assert cfg.base_cell_size == 0.05


def test_t09_5_10_25_50_rejected_naming_r2_r3():
    rings = [_ring(1, 2.5, 10, 0.05), _ring(2, 10, 25, 0.10),
             _ring(3, 25, 50, 0.25), _ring(4, 50, 100, 0.50)]
    with pytest.raises(ConfigError) as e:
        validate_rings(rings, 0.05)
    # Must name the offending ring (0.25 is not a power-of-two multiple of base).
    assert "3" in str(e.value)


def test_non_power_of_two_pair_rejected():
    # Sizes individually pow2 of base, but adjacent pair 0.1 -> 0.3 is not:
    rings = [_ring(1, 0, 10, 0.10), _ring(2, 10, 50, 0.30)]
    with pytest.raises(ConfigError):
        validate_rings(rings, 0.05)


def test_non_contiguous_rejected():
    rings = [_ring(1, 2.5, 10, 0.05), _ring(2, 12, 25, 0.10)]
    with pytest.raises(ConfigError) as e:
        validate_rings(rings, 0.05)
    assert "contiguous" in str(e.value)


def test_empty_table_rejected():
    with pytest.raises(ConfigError):
        validate_rings([], 0.05)
