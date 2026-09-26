"""Module D — cell statistics.

Payload per cell (spec section 9): z_min, z_max, z_mean, z_var (load-bearing:
separates flat road from kerb when means match), class_hist (COUNTS, never a
majority vote), return_count, occupancy_conf, neg_obstacle_conf.

Why the histogram and not a vote: a cell with 99 road points and 1 pedestrian
point votes ROAD and the pedestrian ceases to exist. In far-field rings that is
the case that matters — a safety property, not a precision refinement.
"""

from __future__ import annotations

import numpy as np

from src.grid.foveated_grid import FoveatedGrid, CellBlock
from src.segmentation.class_map import ROAD, STATIC, DYNAMIC

# Severity order for rendering: DYNAMIC > STATIC > ROAD > IGNORE.
SEVERITY_ASCENDING = (ROAD, STATIC, DYNAMIC)


def expected_returns(r: np.ndarray, cell_size: float, azimuth_res_rad: float) -> np.ndarray:
    """Heuristic expected return count for a cell at planar range r: number of
    azimuth beams crossing the cell footprint, at least 1."""
    with np.errstate(divide="ignore"):
        e = cell_size / (np.maximum(r, 1e-6) * azimuth_res_rad)
    return np.maximum(e, 1.0)


def compute_stats(grid: FoveatedGrid, image_width: int = 2048) -> FoveatedGrid:
    """Fill z_var, occupancy_conf, neg_obstacle_conf in place. No NaN in any
    field across a full sequence (AC-D1)."""
    az_res = 2.0 * np.pi / image_width
    for block in grid.blocks.values():
        n = len(block)
        counts = block.count.astype(np.float64)
        z_var = np.zeros(n, dtype=np.float32)
        multi = block.count > 1
        z_var[multi] = (block.m2[multi] / counts[multi]).astype(np.float32)  # never NaN
        block.z_var = z_var

        cx_m = (block.cx + 0.5) * block.cell_size
        cy_m = (block.cy + 0.5) * block.cell_size
        r = np.hypot(cx_m, cy_m)
        exp = expected_returns(r, block.cell_size, az_res)
        block.occupancy_conf = np.clip(counts / exp, 0.0, 1.0).astype(np.float32)

        if block.neg_obstacle_conf is None:
            block.neg_obstacle_conf = np.zeros(n, dtype=np.float32)

        for arr in (block.z_var, block.occupancy_conf, block.neg_obstacle_conf,
                    block.mean, block.m2, block.z_min, block.z_max):
            assert not np.isnan(arr).any(), "AC-D1 violated: NaN in cell payload"
    return grid


def render_class(hist: np.ndarray, min_count: int = 1) -> np.ndarray:
    """Render by highest-severity class present with >= min_count returns, not
    by majority — otherwise the display discards the property the storage
    preserves (AC-D4 / T-14). hist: uint [n, 4]. Returns int8 [n]."""
    out = np.zeros(hist.shape[0], dtype=np.int8)
    for cls in SEVERITY_ASCENDING:  # ascending severity; later writes win
        out[hist[:, cls] >= min_count] = cls
    return out


# ---------------------------------------------------------------------------
# AC-D2 measurement: variance separates flat road from kerb.
# ---------------------------------------------------------------------------

# Raw SemanticKITTI IDs (NOT the 3-class map, which folds both into ROAD):
# the road/sidewalk boundary is where the physical kerb step lives, so a cell
# containing both raw classes straddles the kerb.
RAW_ROAD, RAW_SIDEWALK = 40, 48


def kerb_road_variance_cells(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             raw_labels: np.ndarray, cell_size: float,
                             r_min: float, r_max: float,
                             min_points: int = 4):
    """Per-cell z-variance for road-interior cells (all points raw road) vs
    kerb cells (raw road AND raw sidewalk present) within [r_min, r_max].

    min_points defaults to 4: at HDL-64 beam spacing a 5 cm cell in a SINGLE
    sweep rarely collects more returns, and requiring 8 empties both groups.

    Returns (road_var, kerb_var, kerb_cells) where kerb_cells is
    [(cx_m, cy_m, var, count), ...] for the manual verification list AC-D2
    requires (>= 20 manually verified cells).
    """
    r = np.hypot(x, y)
    keep = (r > r_min) & (r <= r_max) & np.isin(raw_labels, (RAW_ROAD, RAW_SIDEWALK))
    if not np.any(keep):
        return np.zeros(0), np.zeros(0), []
    x, y, z, lab = x[keep], y[keep], z[keep], raw_labels[keep]

    cx = np.floor(x / cell_size).astype(np.int64)
    cy = np.floor(y / cell_size).astype(np.int64)
    order = np.lexsort((cy, cx))
    cx_s, cy_s, z_s, lab_s = cx[order], cy[order], z[order], lab[order]
    new = np.empty(z_s.shape[0], dtype=bool)
    new[0] = True
    new[1:] = (cx_s[1:] != cx_s[:-1]) | (cy_s[1:] != cy_s[:-1])
    starts = np.flatnonzero(new)
    cell_of = np.cumsum(new) - 1
    counts = np.diff(np.append(starts, z_s.shape[0]))

    sums = np.add.reduceat(z_s, starts)
    means = sums / counts
    centered = z_s - means[cell_of]
    var = np.add.reduceat(centered * centered, starts) / counts
    n_road = np.add.reduceat((lab_s == RAW_ROAD).astype(np.int64), starts)
    n_side = counts - n_road

    enough = counts >= min_points
    road_interior = enough & (n_side == 0)
    kerb = enough & (n_road > 0) & (n_side > 0)

    kerb_cells = [((float(cx_s[s]) + 0.5) * cell_size, (float(cy_s[s]) + 0.5) * cell_size,
                   float(v), int(c))
                  for s, v, c in zip(starts[kerb], var[kerb], counts[kerb])]
    return var[road_interior], var[kerb], kerb_cells
