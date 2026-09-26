"""Shared lattice indexing.

All rings index into ONE lattice anchored at the sensor origin. Per-ring
independent division would produce misaligned boundaries — the alignment error
Requirement 3 prohibits. Because every ring is anchored at 0 and sizes are
power-of-two multiples, a 40 cm cell at index k contains exactly the 5 cm cells
8k .. 8k+7; nesting is structural, not checked at runtime.
"""

from __future__ import annotations

import numpy as np


def cell_index(x: float, y: float, cell_size: float) -> tuple[int, int]:
    """Scalar cell index (reference definition from the spec)."""
    return (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))


def cell_indices(x: np.ndarray, y: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised cell indices, int64."""
    cx = np.floor(x / cell_size).astype(np.int64)
    cy = np.floor(y / cell_size).astype(np.int64)
    return cx, cy
