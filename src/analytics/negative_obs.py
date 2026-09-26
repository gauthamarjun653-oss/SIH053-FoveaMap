"""Negative-obstacle inference (Module D) and offline ground-truth derivation
for the section 11.3 pre-check.

Per-sweep inference:
  1. Estimate a local ground plane per region from ROAD cells. On failure the
     region is skipped and logged — never guessed.
  2. Find cells where a ground intersection was expected but no return arrived,
     plus occupied cells whose floor sits below the plane by >= the depth
     threshold (a pothole that DOES return points).
  3. Rule out occlusion by classified static/dynamic obstacles before flagging.
  4. Emit candidates with confidence in [0, 1] — never hard classifications.

A cell with return_count == 0 is NOT automatically a negative obstacle;
occlusion must be excluded first (AC-D5 / T-15).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.common.config import RingConfig
from src.grid.foveated_grid import FoveatedGrid
from src.segmentation.class_map import ROAD, STATIC, DYNAMIC

log = logging.getLogger(__name__)


@dataclass
class GroundPlane:
    tile: tuple[int, int]
    a: float  # z = a*x + b*y + c
    b: float
    c: float
    n_cells: int
    rms_residual: float


@dataclass
class NegativeObstacleCandidate:
    x: float
    y: float
    cell_size: float
    confidence: float           # [0, 1], never a hard classification
    kind: str                   # "missing_return" | "below_plane"
    depth: float = 0.0          # for below_plane candidates


def fit_ground_planes(grid: FoveatedGrid, tile_size: float = 5.0,
                      min_cells: int = 8, max_rms: float = 0.15) -> dict[tuple[int, int], GroundPlane]:
    """Least-squares plane per tile from ROAD-bearing cells' z_min.
    Failed tiles are skipped and logged. Do not guess."""
    xs, ys, zs = [], [], []
    for block in grid.blocks.values():
        road = block.hist[:, ROAD] > 0
        if not np.any(road):
            continue
        xs.append((block.cx[road] + 0.5) * block.cell_size)
        ys.append((block.cy[road] + 0.5) * block.cell_size)
        zs.append(block.z_min[road])
    planes: dict[tuple[int, int], GroundPlane] = {}
    if not xs:
        return planes
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    tx = np.floor(x / tile_size).astype(np.int64)
    ty = np.floor(y / tile_size).astype(np.int64)
    for tile in set(zip(tx.tolist(), ty.tolist())):
        m = (tx == tile[0]) & (ty == tile[1])
        if int(m.sum()) < min_cells:
            log.info("Ground plane skipped for tile %s: only %d road cells", tile, int(m.sum()))
            continue
        A = np.stack([x[m], y[m], np.ones(int(m.sum()))], axis=1)
        coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
        resid = z[m] - A @ coef
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if rms > max_rms:
            log.info("Ground plane skipped for tile %s: rms residual %.3f m", tile, rms)
            continue
        planes[tile] = GroundPlane(tile, float(coef[0]), float(coef[1]), float(coef[2]),
                                   int(m.sum()), rms)
    return planes


class _OcclusionIndex:
    """Azimuth-binned prefix-max of obstacle top slopes for O(log n) ray checks."""

    def __init__(self, grid: FoveatedGrid, n_bins: int = 720, margin: float = 0.05):
        self.n_bins = n_bins
        self.margin = margin
        self.bins: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        ox, oy, oz = [], [], []
        for block in grid.blocks.values():
            obst = (block.hist[:, STATIC] + block.hist[:, DYNAMIC]) > 0
            if not np.any(obst):
                continue
            ox.append((block.cx[obst] + 0.5) * block.cell_size)
            oy.append((block.cy[obst] + 0.5) * block.cell_size)
            oz.append(block.z_max[obst])
        if not ox:
            return
        x = np.concatenate(ox)
        y = np.concatenate(oy)
        z_top = np.concatenate(oz)
        r = np.hypot(x, y)
        valid = r > 1e-6
        x, y, z_top, r = x[valid], y[valid], z_top[valid], r[valid]
        az = np.arctan2(y, x)
        b = ((az + np.pi) / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
        slope = z_top / r  # elevation slope of the obstacle top as seen from origin
        for bin_id in np.unique(b):
            m = b == bin_id
            order = np.argsort(r[m], kind="stable")
            rr = r[m][order]
            self.bins[int(bin_id)] = (rr, np.maximum.accumulate(slope[m][order]))

    def occluded(self, x: float, y: float, z_ground: float) -> bool:
        r_g = float(np.hypot(x, y))
        if r_g < 1e-6:
            return False
        az = float(np.arctan2(y, x))
        bin_id = int((az + np.pi) / (2 * np.pi) * self.n_bins) % self.n_bins
        entry = self.bins.get(bin_id)
        if entry is None:
            return False
        rr, prefix_max_slope = entry
        k = int(np.searchsorted(rr, r_g, side="left"))
        if k == 0:
            return False
        # Line of sight from the sensor origin to the expected ground point has
        # slope z_ground / r_g; any nearer obstacle top at or above it blocks it.
        return bool(prefix_max_slope[k - 1] >= (z_ground / r_g) - self.margin)


def negative_obstacle_candidates(grid: FoveatedGrid, ring_config: RingConfig,
                                 depth_threshold: float = 0.15,
                                 tile_size: float = 5.0,
                                 max_range: float = 30.0,
                                 stats_out: dict | None = None) -> list[NegativeObstacleCandidate]:
    """Emit candidates with confidence in [0, 1]. Also writes neg_obstacle_conf
    onto occupied below-plane cells in place.

    stats_out, when given, is filled with the frame's decision counters
    (below_plane, missing_flagged, occlusion_suppressed, no_plane) so the
    occlusion exclusion is measurable on real data, not just unit-tested."""
    if stats_out is not None:
        stats_out.update(below_plane=0, missing_flagged=0,
                         occlusion_suppressed=0, no_plane=0)
    planes = fit_ground_planes(grid, tile_size=tile_size)
    if not planes:
        log.info("No ground planes fit; no negative-obstacle inference this frame.")
        return []
    occlusion = _OcclusionIndex(grid)
    candidates: list[NegativeObstacleCandidate] = []

    # Occupied index per ring for the missing-return scan.
    occupied: dict[int, set[tuple[int, int]]] = {
        ring_id: set(zip(block.cx.tolist(), block.cy.tolist()))
        for ring_id, block in grid.blocks.items()
    }

    # (a) Occupied cells whose floor is below the local plane by >= threshold.
    for block in grid.blocks.values():
        if block.neg_obstacle_conf is None:
            block.neg_obstacle_conf = np.zeros(len(block), dtype=np.float32)
        cx_m = (block.cx + 0.5) * block.cell_size
        cy_m = (block.cy + 0.5) * block.cell_size
        for i in range(len(block)):
            tile = (int(np.floor(cx_m[i] / tile_size)), int(np.floor(cy_m[i] / tile_size)))
            plane = planes.get(tile)
            if plane is None:
                continue
            z_exp = plane.a * cx_m[i] + plane.b * cy_m[i] + plane.c
            depth = float(z_exp - block.z_min[i])
            if depth >= depth_threshold:
                conf = float(np.clip(depth / (2.0 * depth_threshold), 0.0, 1.0))
                block.neg_obstacle_conf[i] = conf
                candidates.append(NegativeObstacleCandidate(
                    float(cx_m[i]), float(cy_m[i]), block.cell_size, conf,
                    "below_plane", depth))
                if stats_out is not None:
                    stats_out["below_plane"] += 1

    # (b) Missing-return cells: expected ground intersection, no return, and
    # occlusion ruled out.
    for ring in ring_config.rings:
        if ring.radius_min > max_range:
            continue
        occ = occupied.get(ring.id, set())
        r_hi = min(ring.radius_max, max_range)
        lo = int(np.floor(-r_hi / ring.cell_size))
        hi = int(np.ceil(r_hi / ring.cell_size))
        for cx in range(lo, hi):
            x = (cx + 0.5) * ring.cell_size
            for cy in range(lo, hi):
                y = (cy + 0.5) * ring.cell_size
                r = float(np.hypot(x, y))
                if not (ring.radius_min < r <= r_hi):
                    continue
                if (cx, cy) in occ:
                    continue
                tile = (int(np.floor(x / tile_size)), int(np.floor(y / tile_size)))
                plane = planes.get(tile)
                if plane is None:
                    if stats_out is not None:
                        stats_out["no_plane"] += 1
                    continue  # no expectation without a fitted plane — do not guess
                z_exp = plane.a * x + plane.b * y + plane.c
                if occlusion.occluded(x, y, z_exp):
                    if stats_out is not None:
                        stats_out["occlusion_suppressed"] += 1
                    continue  # explained by a classified obstacle (AC-D5)
                # Confidence scales with how much evidence was expected.
                exp_returns = ring.cell_size / (r * (2 * np.pi / 2048))
                conf = float(np.clip(exp_returns / 5.0, 0.05, 1.0))
                candidates.append(NegativeObstacleCandidate(
                    x, y, ring.cell_size, conf, "missing_return"))
                if stats_out is not None:
                    stats_out["missing_flagged"] += 1
    return candidates


# ---------------------------------------------------------------------------
# Offline ground truth (spec 11.3) — run the pre-check FIRST, count instances.
# ---------------------------------------------------------------------------

@dataclass
class DepressionInstance:
    centroid: tuple[float, float]
    size_m: tuple[float, float]   # bounding extent
    depth: float
    n_cells: int
    confidence: float


def derive_ground_truth_depressions(map_points: np.ndarray,
                                    depth_threshold: float = 0.15,
                                    max_extent_m: float | None = 1.0,
                                    cell: float = 0.10,
                                    neighborhood_m: float = 3.0,
                                    phantom_depth_m: float = 5.0) -> list[DepressionInstance]:
    """Detect depressions in the global map: cells whose ground level sits >=
    depth_threshold below the local median. Passing max_extent_m=None drops the
    horizontal constraint (the section 11.3 reframe: road-edge elevation
    discontinuities, NOT potholes — naming discipline matters).

    Spot-check >= 20 instances manually before using this to score anything.
    """
    if map_points.shape[0] == 0:
        return []
    cx = np.floor(map_points[:, 0] / cell).astype(np.int64)
    cy = np.floor(map_points[:, 1] / cell).astype(np.int64)
    key = (cx - cx.min()) * (cy.max() - cy.min() + 1) + (cy - cy.min())
    order = np.argsort(key, kind="stable")
    uniq, starts = np.unique(key[order], return_index=True)
    z_ground = np.minimum.reduceat(map_points[:, 2][order], starts)
    ucx = cx[order][starts]
    ucy = cy[order][starts]

    # Local reference: median ground level within a square neighborhood, via a
    # coarse tile median (robust and cheap).
    tile = max(1, int(round(neighborhood_m / cell)))
    txy = {}
    tkx = ucx // tile
    tky = ucy // tile
    for t in set(zip(tkx.tolist(), tky.tolist())):
        m = (tkx == t[0]) & (tky == t[1])
        txy[t] = float(np.median(z_ground[m]))
    ref = np.array([txy[(int(a), int(b))] for a, b in zip(tkx, tky)])

    depth = ref - z_ground
    # Phantom guard: per-cell ground is a MINIMUM, so one mirror-reflection
    # point tens of meters below ground (observed on seq 00: down to -29 m)
    # would otherwise become an absurdly deep "depression". Real negative
    # obstacles are decimeters deep; cells deeper than phantom_depth_m are
    # sensor artifacts, excluded and counted out loud.
    is_phantom = depth > phantom_depth_m
    if np.any(is_phantom):
        log.warning("Excluding %d cells with apparent depth > %.1f m "
                    "(mirror-reflection phantoms, not depressions).",
                    int(is_phantom.sum()), phantom_depth_m)
    is_dep = (depth >= depth_threshold) & ~is_phantom
    if not np.any(is_dep):
        return []

    # Connected components (8-neighbour — robust to sparsely hit cells) over
    # depressed cells.
    dep_cells = {(int(a), int(b)): float(d)
                 for a, b, d in zip(ucx[is_dep], ucy[is_dep], depth[is_dep])}
    seen: set[tuple[int, int]] = set()
    instances: list[DepressionInstance] = []
    for start in dep_cells:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            comp.append(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in dep_cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        arr = np.array(comp)
        extent = ((arr[:, 0].max() - arr[:, 0].min() + 1) * cell,
                  (arr[:, 1].max() - arr[:, 1].min() + 1) * cell)
        if max_extent_m is not None and max(extent) > max_extent_m:
            continue
        d_max = max(dep_cells[c] for c in comp)
        conf = float(np.clip(len(comp) / 6.0, 0.0, 1.0)) * float(np.clip(d_max / (2 * depth_threshold), 0.0, 1.0))
        instances.append(DepressionInstance(
            centroid=(float((arr[:, 0].mean() + 0.5) * cell), float((arr[:, 1].mean() + 0.5) * cell)),
            size_m=extent, depth=float(d_max), n_cells=len(comp),
            confidence=float(np.clip(conf, 0.0, 1.0))))
    return instances
