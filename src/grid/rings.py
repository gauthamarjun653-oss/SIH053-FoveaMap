"""Ring assignment by planar radius.

Boundary points go to the INNER (finer) ring: r == 10.0 lands in R1, not R2
(spec section 3 / T-11).
"""

from __future__ import annotations

import numpy as np

from src.common.config import RingConfig

OUT_OF_RANGE = -1


def assign_rings(x: np.ndarray, y: np.ndarray, ring_config: RingConfig) -> np.ndarray:
    """Return per-point ring ids; OUT_OF_RANGE (-1) for points outside all rings.

    Uses planar radius r = sqrt(x^2 + y^2). searchsorted with side='left' on the
    radius_max table implements `r <= radius_max`, so exact boundary radii land
    in the inner ring.
    """
    r = np.hypot(x, y)
    maxima = np.array([rg.radius_max for rg in ring_config.rings])
    ids = np.array([rg.id for rg in ring_config.rings], dtype=np.int64)
    idx = np.searchsorted(maxima, r, side="left")
    out = np.full(r.shape, OUT_OF_RANGE, dtype=np.int64)
    in_range = (idx < len(maxima)) & (r >= ring_config.radius_min)
    out[in_range] = ids[idx[in_range]]
    return out


def assign_ring_scalar(r: float, ring_config: RingConfig) -> int:
    """Scalar reference implementation from the spec."""
    if r < ring_config.radius_min:
        return OUT_OF_RANGE
    ring = next((rg for rg in ring_config.rings if r <= rg.radius_max), None)
    return ring.id if ring is not None else OUT_OF_RANGE
