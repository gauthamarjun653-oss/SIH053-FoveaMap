"""Ground estimates with per-plane convex-hull support and fit-quality gates.

ROAD is a semantic grouping, not a guarantee of traversability. Mixed
road/obstacle cells are excluded from plane fitting. No extrapolation is
labelled SUPPORTED, including across tile boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from src.grid.foveated_grid import FoveatedGrid
from src.segmentation.class_map import ROAD

UNSUPPORTED, EXTRAPOLATED, SUPPORTED = 0, 1, 2
STATE_NAMES = ("UNSUPPORTED", "EXTRAPOLATED", "SUPPORTED")

@dataclass
class GroundPlane:
    tile: tuple[int, int]
    a: float
    b: float
    c: float
    n_cells: int
    rms_residual: float
    # Convex hull halfspaces: nx*x + ny*y + offset <= 0.
    support_equations: np.ndarray | None = None

class GroundField:
    def __init__(self, planes: dict[tuple[int, int], GroundPlane],
                 tile_size: float, global_plane: tuple[float, float, float] | None):
        self.planes = planes
        self.tile_size = float(tile_size)
        self.global_plane = global_plane
        ordered = sorted(planes.values(), key=lambda p: int(self._encode(*p.tile)))
        self._ordered = ordered
        self._codes = np.array([self._encode(*p.tile) for p in ordered], dtype=np.int64)
        self._coef = np.array([[p.a, p.b, p.c] for p in ordered], dtype=np.float64).reshape(-1, 3)
        self._rms = np.array([p.rms_residual for p in ordered])

    @staticmethod
    def _encode(tx, ty):
        return (np.asarray(tx, dtype=np.int64) + (1 << 20)) * (1 << 21) + (np.asarray(ty, dtype=np.int64) + (1 << 20))

    @property
    def median_rms(self):
        return float(np.median(self._rms)) if self._rms.size else 0.0

    @property
    def sensor_height(self):
        # Bootstrap estimate from near-ground data, NOT sensor calibration.
        if self.global_plane is None or not np.isfinite(self.global_plane[2]):
            return None
        h = -float(self.global_plane[2])
        return h if h > 0 else None

    def tile_of(self, x, y):
        return (np.floor(np.asarray(x) / self.tile_size).astype(np.int64),
                np.floor(np.asarray(y) / self.tile_size).astype(np.int64))

    def _lookup(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        tx, ty = self.tile_of(np.where(finite, x, 0), np.where(finite, y, 0))
        codes = self._encode(tx, ty)
        if not self._codes.size:
            return np.zeros(x.shape, dtype=np.int64), np.zeros(x.shape, dtype=bool)
        idx = np.clip(np.searchsorted(self._codes, codes), 0, self._codes.size - 1)
        return idx, finite & (self._codes[idx] == codes)

    def plane_coeffs_at(self, x, y):
        idx, hit = self._lookup(x, y)
        if not self._codes.size:
            return np.zeros((np.asarray(x).size, 3)), hit
        return self._coef[idx], hit

    def rms_at(self, x, y):
        idx, hit = self._lookup(x, y)
        if not self._codes.size:
            return np.full(np.asarray(x).shape, np.nan)
        return np.where(hit, self._rms[idx], np.nan)

    def height_and_state(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        idx, hit = self._lookup(x, y)
        z = np.full(x.shape, np.nan)
        state = np.where(hit, EXTRAPOLATED, UNSUPPORTED).astype(np.int8)
        if not self._codes.size:
            return z, state
        coef = self._coef[idx]
        z[hit] = (coef[:, 0] * x + coef[:, 1] * y + coef[:, 2])[hit]
        for i in np.unique(idx[hit]):
            eq = self._ordered[i].support_equations
            if eq is None:
                continue  # A manually supplied plane has no support evidence.
            sel = np.flatnonzero(hit & (idx == i))
            inside = np.ones(sel.size, dtype=bool)
            for nx, ny, offset in eq:
                inside &= nx*x[sel] + ny*y[sel] + offset <= 1e-9
            state[sel[inside]] = SUPPORTED
        return z, state


def _fit(x, y, z, min_spread_m, max_condition):
    xy = np.column_stack([x, y])
    centre = xy.mean(axis=0)
    centered = xy - centre
    spread = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(x))
    if spread.size < 2 or spread[1] < min_spread_m or spread[0]/spread[1] > max_condition:
        return None
    A = np.column_stack([centered, np.ones(len(x))])
    coef, _, rank, _ = np.linalg.lstsq(A, z, rcond=None)
    if rank != 3:
        return None
    rms = float(np.sqrt(np.mean((z - A @ coef)**2)))
    coef[2] -= coef[:2] @ centre
    return coef, rms


def build_ground_field(grid: FoveatedGrid, tile_size: float = 5.0,
                       min_cells: int = 8, max_rms: float = 0.15,
                       max_range: float = 100.0, min_spread_m: float = 0.10,
                       max_condition: float = 100.0) -> GroundField:
    if tile_size <= 0 or min_cells < 3 or max_rms < 0 or min_spread_m <= 0:
        raise ValueError("Invalid ground fitting parameters")
    parts = []
    for block in grid.blocks.values():
        # Only cells whose returns all belong to the ground grouping.
        road = (block.hist[:, ROAD] == block.count) & (block.count > 0)
        x = (block.cx.astype(np.float64) + .5)*block.cell_size
        y = (block.cy.astype(np.float64) + .5)*block.cell_size
        road &= np.hypot(x, y) <= max_range
        if np.any(road):
            parts.append(np.column_stack([x[road], y[road], block.z_min[road]]))
    if not parts:
        return GroundField({}, tile_size, None)
    xyz = np.concatenate(parts)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < min_cells:
        return GroundField({}, tile_size, None)
    x, y, z = xyz.T
    # Near-field bootstrap reduces distant hills' influence on sensor height.
    near = np.hypot(x, y) <= 15.0
    # Robust bootstrap only seeds ray iteration; local support is still required.
    # Trim outlying heights, never force an unacceptable residual to pass.
    initial = None
    keep = near.copy()
    for _ in range(4):
        if keep.sum() < min_cells:
            break
        initial = _fit(x[keep], y[keep], z[keep], min_spread_m, max_condition)
        if initial is None:
            break
        coef, rms = initial
        if rms <= max_rms:
            break
        residual = z - (coef[0]*x + coef[1]*y + coef[2])
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep]-median))
        keep &= np.abs(residual-median) <= max(.03, 3*1.4826*mad)
    global_plane = tuple(initial[0]) if initial is not None and initial[1] <= max_rms else None
    tx, ty = np.floor(x/tile_size).astype(np.int64), np.floor(y/tile_size).astype(np.int64)
    order = np.lexsort((ty, tx))
    keys = np.column_stack([tx[order], ty[order]])
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    ends = np.r_[starts[1:], len(order)]
    planes = {}
    for s, e in zip(starts, ends):
        if e-s < min_cells:
            continue
        ii = order[s:e]
        fit = _fit(x[ii], y[ii], z[ii], min_spread_m, max_condition)
        if fit is None or fit[1] > max_rms:
            continue
        try:
            hull = ConvexHull(xyz[ii, :2])
        except QhullError:
            continue
        tile = (int(tx[ii[0]]), int(ty[ii[0]]))
        coef, rms = fit
        planes[tile] = GroundPlane(tile, *map(float, coef), len(ii), rms, hull.equations.copy())
    return GroundField(planes, tile_size, global_plane)
