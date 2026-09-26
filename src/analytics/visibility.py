"""Modelled beam intersections and evidence-aware visibility.

Beam angles default to an approximate uniform model, not calibrated firing
angles. Missing returns are candidates, never proof of a depression. BLIND
requires an independently supplied, validated azimuth mask; the current scan
cannot establish its own self-occlusion mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from scipy.spatial import cKDTree

import numpy as np

from src.analytics.ground import SUPPORTED, GroundField
from src.common.config import RingConfig
from src.grid.foveated_grid import FoveatedGrid
from src.grid.rings import OUT_OF_RANGE, assign_rings

OBSERVED, OCCLUDED, NO_GROUND, BLIND, EXPECTED_EMPTY = 0, 1, 2, 3, 4
STATE_NAMES = ("OBSERVED", "OCCLUDED", "NO_GROUND", "BLIND", "EXPECTED_EMPTY")


def beam_elevations(height: int = 64, fov_up_deg: float = 3.0,
                    fov_down_deg: float = -25.0) -> np.ndarray:
    """Uniform beam model matching src/projection/range_projection.py.

    Kept identical to the projection so ray counts correspond to the image the
    segmentation model actually sees. `derive_beam_elevations` refines this
    from data when the uniform assumption is not good enough.
    """
    return np.radians(np.linspace(fov_up_deg, fov_down_deg, height))


def derive_beam_elevations(points_xyz: np.ndarray, rows: np.ndarray,
                           height: int = 64) -> np.ndarray:
    """Empirical beam elevations: median elevation of the returns assigned to
    each image row. Rows with no returns fall back to the uniform model.

    The HDL-64E's beams are not evenly spaced; row medians remain an approximation: projection rows are not physical
    laser-channel identifiers. Use calibrated angles for physical ray counts.
    """
    r = np.hypot(points_xyz[:, 0], points_xyz[:, 1])
    el = np.arctan2(points_xyz[:, 2], np.maximum(r, 1e-9))
    out = beam_elevations(height)
    valid = rows >= 0
    if not np.any(valid):
        return out
    order = np.argsort(rows[valid], kind="stable")
    rr = rows[valid][order]
    ee = el[valid][order]
    starts = np.flatnonzero(np.r_[True, rr[1:] != rr[:-1]])
    ends = np.r_[starts[1:], rr.size]
    for s, e in zip(starts.tolist(), ends.tolist()):
        row = int(rr[s])
        if 0 <= row < height:
            out[row] = float(np.median(ee[s:e]))
    return out


@dataclass
class VisibilityResult:
    """Landing points of every ray that reached trusted ground."""
    x: np.ndarray
    y: np.ndarray
    z_ground: np.ndarray
    state: np.ndarray            # one of OBSERVED / OCCLUDED / NO_GROUND / EXPECTED_EMPTY
    ring_id: np.ndarray
    cell_size: np.ndarray
    expected_returns: np.ndarray  # modelled rays landing in the same cell
    counts: dict                  # state name -> count, for the §5 diagnostic
    inconsistent: int             # rays whose tile iteration never settled
    radial_tolerance: np.ndarray | None = None  # per-ray occupancy search radius
    reached_cells: int = 0


def _cell_of(x: np.ndarray, y: np.ndarray, ring_config: RingConfig
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ring id and lattice indices for landing points.

    Ownership (definition §3.3): landing points are predicted *returns*, so
    they are assigned exactly as real returns are — by point radius via
    `assign_rings`. Using the same convention for both is what removes the
    ring-boundary false alarms; the old scan walked lattice cells by
    cell-centre radius while points were stored by point radius, so cells
    either side of a transition radius were scanned in a ring that did not
    hold their points.
    """
    ring_id = assign_rings(x, y, ring_config)
    size = np.zeros(x.shape, dtype=np.float64)
    for rg in ring_config.rings:
        size[ring_id == rg.id] = rg.cell_size
    ok = ring_id != OUT_OF_RANGE
    cx = np.zeros(x.shape, dtype=np.int64)
    cy = np.zeros(x.shape, dtype=np.int64)
    cx[ok] = np.floor(x[ok] / size[ok]).astype(np.int64)
    cy[ok] = np.floor(y[ok] / size[ok]).astype(np.int64)
    return ring_id, cx, cy, size


class _OccupancyProximity:
    """Exact XY distance check, with optional ground-height compatibility.

    A KD tree accelerates candidate lookup; its buckets never establish
    occupancy. Raw returns are preferred. Grid-only callers explicitly use
    cell centres as an approximation to return positions.
    """
    def __init__(self, grid, extent, points_xyz=None):
        if points_xyz is None:
            parts = [np.column_stack([(b.cx.astype(float)+.5)*b.cell_size,
                                      (b.cy.astype(float)+.5)*b.cell_size, b.mean])
                     for b in grid.blocks.values() if len(b)]
            points = np.concatenate(parts) if parts else np.empty((0, 3))
        else:
            points = np.asarray(points_xyz, dtype=float)[:, :3]
        keep = np.isfinite(points).all(axis=1) & (np.hypot(points[:, 0], points[:, 1]) <= extent)
        self.points = points[keep]
        self.tree = cKDTree(self.points[:, :2])

    def occupied_near(self, x, y, tol_m, z_ground=None, z_tol_m=.05):
        x, y = np.asarray(x), np.asarray(y)
        tol = np.broadcast_to(tol_m, x.shape)
        if np.any(~np.isfinite(tol)) or np.any(tol < 0):
            raise ValueError("Occupancy tolerance must be finite and nonnegative")
        out = np.zeros(x.size, dtype=bool)
        if not len(self.points):
            return out
        queries = np.column_stack([x, y])
        if z_ground is None:
            distance, _ = self.tree.query(queries, k=1)
            return distance <= tol + 1e-12
        zg = np.broadcast_to(z_ground, x.shape)
        zt = np.broadcast_to(z_tol_m, x.shape)
        # Most ground queries are explained by their nearest XY return.
        # Only height-incompatible nearest returns need a full radius search.
        distance, nearest = self.tree.query(queries, k=1)
        close = distance <= tol + 1e-12
        out = close & (np.abs(self.points[nearest, 2]-zg) <= zt)
        pending = np.flatnonzero(close & ~out)
        for start in range(0, len(pending), 512):
            selection = pending[start:start+512]
            neighbours = self.tree.query_ball_point(queries[selection], tol[selection])
            lengths = np.fromiter(map(len, neighbours), dtype=np.int64, count=len(selection))
            if not lengths.sum():
                continue
            qi = np.repeat(selection, lengths)
            pi = np.concatenate([np.asarray(v, dtype=np.int64) for v in neighbours])
            good = np.abs(self.points[pi, 2] - zg[qi]) <= zt[qi]
            np.logical_or.at(out, qi, good)
        return out


class _BlindZone:
    """Validated minimum visible range per azimuth; absent mask means unknown.

    No returns from the frame are used to infer vehicle geometry. A mask is
    supplied in bins spanning [-pi, pi); entries are nonnegative metres.
    """
    def __init__(self, ranges_m=None):
        self.r_min = None if ranges_m is None else np.asarray(ranges_m, dtype=float)
        if self.r_min is not None and (self.r_min.ndim != 1 or not self.r_min.size
                or not np.isfinite(self.r_min).all() or np.any(self.r_min < 0)):
            raise ValueError("Blind mask must be a nonempty finite nonnegative 1D array")

    def blind(self, x, y):
        if self.r_min is None:
            return np.zeros(np.asarray(x).shape, dtype=bool)
        az = np.arctan2(y, x)
        bins = np.floor((az+np.pi)/(2*np.pi)*len(self.r_min)).astype(np.int64) % len(self.r_min)
        return np.hypot(x, y) < self.r_min[bins]


def compute_visibility(grid: FoveatedGrid, ground: GroundField,
                       ring_config: RingConfig, occlusion,
                       elevations: np.ndarray | None = None,
                       image_width: int = 2048,
                       max_range: float = 25.0,
                       require_supported: bool = True,
                       max_refine_iters: int = 4,
                       points_xyz: np.ndarray | None = None,
                       blind_ranges_m: np.ndarray | None = None) -> VisibilityResult:
    """Cast one ray per (beam, azimuth) and classify where each lands."""
    if max_refine_iters < 1 or image_width < 1:
        raise ValueError("Positive iteration limit and image width required")
    if elevations is None:
        elevations = beam_elevations()
    if ground.global_plane is None:
        empty = np.zeros(0)
        return VisibilityResult(empty, empty, empty,
                                np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int64),
                                empty, empty,
                                {n: 0 for n in STATE_NAMES}, 0)

    az = (np.arange(image_width, dtype=np.float64) + 0.5) / image_width * 2 * np.pi - np.pi
    el = np.asarray(elevations, dtype=np.float64)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    dx = (ce[:, None] * ca[None, :]).ravel()
    dy = (ce[:, None] * sa[None, :]).ravel()
    dz = np.repeat(se, image_width)

    def intersect(a, b, c):
        denom = dz - a * dx - b * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = c / denom
        good = np.isfinite(t) & (t > 0)
        return t, good

    # Bootstrap on the global plane, then iterate: look up the tile the ray is
    # heading for, re-intersect that tile's own plane, repeat. One pass is not
    # enough — at range, the r/h gain means the global-plane estimate can land
    # a tile or more away from the truth, and a single correction then lands in
    # a third tile. The fixed point is normally reached in two or three steps.
    ga, gb, gc = ground.global_plane
    t, good = intersect(ga, gb, gc)
    t = np.where(good, t, np.nan)
    x, y = t * dx, t * dy
    active = good.copy()
    consistent = np.zeros(good.shape, dtype=bool)
    # Compare the source tile (whose plane we intersect) with its destination.
    # Once settled, a ray is frozen. Oscillations exhaust the iteration limit.
    for _ in range(max_refine_iters):
        if not active.any():
            break
        source_tx, source_ty = ground.tile_of(np.nan_to_num(x), np.nan_to_num(y))
        coef, hit = ground.plane_coeffs_at(np.nan_to_num(x), np.nan_to_num(y))
        denom = dz - coef[:, 0]*dx - coef[:, 1]*dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t_new = coef[:, 2]/denom
        step_ok = active & hit & np.isfinite(t_new) & (t_new > 0)
        t = np.where(step_ok, t_new, t)
        x, y = t*dx, t*dy
        dest_tx, dest_ty = ground.tile_of(np.nan_to_num(x), np.nan_to_num(y))
        settled = step_ok & (source_tx == dest_tx) & (source_ty == dest_ty)
        consistent |= settled
        active = step_ok & ~settled
    z = t*dz
    inconsistent = int(np.count_nonzero(active))

    r = np.hypot(x, y)
    keep = (np.isfinite(r) & (r > ring_config.radius_min) & (r <= max_range)
            & consistent)
    x, y, z = x[keep], y[keep], z[keep]
    if x.size == 0:
        return VisibilityResult(x, y, z, np.zeros(0, dtype=np.int8),
                                np.zeros(0, dtype=np.int64), np.zeros(0),
                                np.zeros(0), {n: 0 for n in STATE_NAMES}, inconsistent)

    ring_id, cx, cy, size = _cell_of(x, y, ring_config)
    valid = ring_id != OUT_OF_RANGE
    x, y, z, ring_id, cx, cy, size = (a[valid] for a in (x, y, z, ring_id, cx, cy, size))

    # Count rays in the supplied model, not independently verified firings.
    key = np.stack([ring_id, cx, cy], axis=1)
    order = np.lexsort((cy, cx, ring_id))
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, np.any(ks[1:] != ks[:-1], axis=1)])
    group = np.zeros(x.size, dtype=np.int64)
    group[starts] = 1
    group = np.cumsum(group) - 1
    counts = np.bincount(group)
    expected = np.empty(x.size, dtype=np.float64)
    expected[order] = counts[group]

    z_ground, gstate = ground.height_and_state(x, y)
    # Local fit residual and per-ray range; no ring-wide median tolerance.
    h = ground.sensor_height or 1.73
    rms = np.maximum(ground.rms_at(x, y), 1e-3)
    tol_used = np.maximum(size, rms*np.hypot(x, y)/h)
    prox = _OccupancyProximity(grid, extent=max_range+float(tol_used.max())+1,
                               points_xyz=points_xyz)
    observed = prox.occupied_near(x, y, tol_used, z_ground,
                                  np.maximum(.03, 2*rms))
    state = np.full(x.size, EXPECTED_EMPTY, dtype=np.int8)
    state[~observed & ((gstate < SUPPORTED) if require_supported
                       else (gstate == 0))] = NO_GROUND
    blind = _BlindZone(blind_ranges_m).blind(x, y)
    state[~observed & blind & (state == EXPECTED_EMPTY)] = BLIND

    # Occlusion is checked only where absence would otherwise be evidence —
    # the index is a Python-level call per site, so restricting it matters.
    cand = np.flatnonzero((state == EXPECTED_EMPTY) & ~observed)
    if cand.size and occlusion is not None:
        blocked = np.fromiter(
            (occlusion.occluded(float(x[i]), float(y[i]),
                                float(z_ground[i]) if np.isfinite(z_ground[i]) else float(z[i]))
             for i in cand.tolist()), dtype=bool, count=cand.size)
        state[cand[blocked]] = OCCLUDED
    state[observed] = OBSERVED

    counts_out = {STATE_NAMES[s]: int(np.count_nonzero(state == s))
                  for s in range(len(STATE_NAMES))}
    return VisibilityResult(x, y, z_ground, state, ring_id, size, expected,
                            counts_out, inconsistent, tol_used, int(starts.size))


def unsampled_cell_estimate(ring_config: RingConfig, max_range: float,
                            reached: int) -> int:
    """Approximate annulus cell count minus UNIQUE accepted landing cells.

    This is an unreached-cell estimate, not a measured UNSAMPLED state count:
    unsupported or rejected intersections also reduce accepted coverage.
    """
    total = 0
    for rg in ring_config.rings:
        r_hi = min(rg.radius_max, max_range)
        if rg.radius_min >= r_hi:
            continue
        area = np.pi * (r_hi ** 2 - rg.radius_min ** 2)
        total += int(area / (rg.cell_size ** 2))
    return max(0, total - reached)
