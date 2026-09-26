"""Negative-obstacle detection, rebuilt on the ground/visibility definition.

See docs/ground-visibility-definition.md. Differences from `negative_obs.py`:

  * Enumeration is by beam-ground intersection, not by lattice sweep, so
    unsampled ground is never visited (definition §2).
  * Ground must be SUPPORTED, not merely present in a tile (§3.1).
  * Landing points are assigned to rings exactly as returns are, so the
    ring-boundary false alarms disappear (§3.3).
  * Candidates are *clusters*, not cells.

The last point is physical, not cosmetic. A depression does not delete a
return; it displaces or shadows one. A depression of depth d seen at range r
from sensor height h hides a strip of ground roughly

    shadow = d * r / h

metres long behind its near lip — 2.2 m for a 0.15 m dip at 25 m. So the real
signature of a pothole at range is a contiguous run of missing returns whose
length matches that prediction. An isolated missing cell is noise. Requiring
the cluster to reach a fraction of its predicted shadow turns "how many cells
are empty" into "does this look like what a depression would actually do",
and it collapses hundreds of thousands of per-cell candidates into instances
a human can check one at a time.

Occlusion by classified obstacles is unchanged and reused from the original
module: it suppresses 41% of missing-return sites and is not the problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.analytics.ground import (EXTRAPOLATED, SUPPORTED, GroundField,
                                  build_ground_field)
from src.analytics.negative_obs import _OcclusionIndex
from src.analytics.visibility import (EXPECTED_EMPTY, compute_visibility,
                                      unsampled_cell_estimate)
from src.common.config import RingConfig
from src.grid.foveated_grid import FoveatedGrid

log = logging.getLogger(__name__)


@dataclass
class NegativeObstacle:
    """One candidate instance. Confidence in [0, 1]; never a hard class."""
    centroid: tuple[float, float]
    size_m: tuple[float, float]
    n_cells: int
    confidence: float  # heuristic score, not a calibrated probability
    kind: str                 # "missing_return" | "below_plane"
    depth: float = 0.0        # measured below-plane; 0 means unknown for missing
    mean_range: float = 0.0
    expected_returns: float = 0.0
    shadow_predicted_m: float = 0.0

    @property
    def x(self):
        return self.centroid[0]

    @property
    def y(self):
        return self.centroid[1]


@dataclass
class DetectionResult:
    candidates: list[NegativeObstacle] = field(default_factory=list)
    visibility_counts: dict = field(default_factory=dict)
    ground_counts: dict = field(default_factory=dict)
    unsampled_cells: int = 0
    inconsistent_rays: int = 0
    clusters_rejected_small: int = 0
    clusters_rejected_shape: int = 0
    below_plane_out_of_range: int = 0
    sensor_height: float | None = None
    missing_return_status: str = "not evaluated"

    def counts_by_kind(self) -> dict:
        out = {"missing_return": 0, "below_plane": 0}
        for c in self.candidates:
            out[c.kind] += 1
        return out


def _components(codes: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """8-connected components over sparse cells of one ring.

    Flood fill on a hash set. The cell count here is in the thousands after
    the visibility filter, so a Python-level fill is affordable — it was not
    affordable over the 2.3M sites the old lattice sweep produced.
    """
    # Python integer coordinates avoid int32 arithmetic overflow for packed cells.
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(cx, cy))}
    label = np.full(codes.size, -1, dtype=np.int64)
    nxt = 0
    for start in range(codes.size):
        if label[start] != -1:
            continue
        stack = [start]
        label[start] = nxt
        while stack:
            i = stack.pop()
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    j = index.get((int(cx[i]) + ox, int(cy[i]) + oy))
                    if j is not None and label[j] == -1:
                        label[j] = nxt
                        stack.append(j)
        nxt += 1
    return label


def _below_plane_candidates(grid: FoveatedGrid, ground: GroundField,
                            depth_threshold: float,
                            require_supported: bool,
                            max_range: float) -> tuple[list[NegativeObstacle], dict, int]:
    """Occupied cells whose floor sits below the local plane, within `max_range`.

    Vectorised; the original looped `for i in range(len(block))` over every
    occupied cell of every ring. `require_supported` is open decision 5 in the
    definition: the hull rule suppresses some genuine kerbside detections, so
    it is a flag to be measured both ways, not a silent choice.

    The independent range limit applies to measured below-plane evidence.
    Missing-return candidates have a separately configured range limit.
    Cell-centre range is used for this output gate; point-radius assignment
    still defines storage ownership.
    """
    out: list[NegativeObstacle] = []
    gcounts = {"UNSUPPORTED": 0, "EXTRAPOLATED": 0, "SUPPORTED": 0}
    out_of_range = 0
    for block in grid.blocks.values():
        if len(block) == 0:
            continue
        if block.neg_obstacle_conf is None:
            block.neg_obstacle_conf = np.zeros(len(block), dtype=np.float32)
        x = (block.cx + 0.5) * block.cell_size
        y = (block.cy + 0.5) * block.cell_size
        in_range = np.hypot(x, y) <= max_range
        out_of_range += int(np.count_nonzero(~in_range))
        z_exp, gstate = ground.height_and_state(x, y)
        for s, name in ((0, "UNSUPPORTED"), (EXTRAPOLATED, "EXTRAPOLATED"),
                        (SUPPORTED, "SUPPORTED")):
            gcounts[name] += int(np.count_nonzero(gstate == s))

        usable = ((gstate >= SUPPORTED) if require_supported
                  else (gstate >= EXTRAPOLATED)) & in_range
        depth = np.where(usable, z_exp - block.z_min, -np.inf)
        flag = np.isfinite(depth) & (depth >= depth_threshold)
        if not np.any(flag):
            continue
        conf = np.clip(depth[flag] / (2.0 * depth_threshold), 0.0, 1.0)
        block.neg_obstacle_conf[flag] = conf.astype(np.float32)

        idx = np.flatnonzero(flag)
        bx = block.cx[idx].astype(np.int64)
        by = block.cy[idx].astype(np.int64)
        codes = (bx + (1 << 30)) * (1 << 31) + (by + (1 << 30))
        labels = _components(codes, block.cx[idx], block.cy[idx])
        for lab in np.unique(labels):
            m = labels == lab
            xs, ys = x[idx][m], y[idx][m]
            out.append(NegativeObstacle(
                centroid=(float(xs.mean()), float(ys.mean())),
                size_m=(float(xs.max() - xs.min() + block.cell_size),
                        float(ys.max() - ys.min() + block.cell_size)),
                n_cells=int(m.sum()),
                confidence=float(conf[m].max()),
                kind="below_plane",
                depth=float(depth[idx][m].max()),
                mean_range=float(np.hypot(xs, ys).mean()),
            ))
    return out, gcounts, out_of_range


def detect(grid: FoveatedGrid, ring_config: RingConfig,
           depth_threshold: float = 0.15,
           tile_size: float = 5.0,
           max_range_missing: float = 25.0,
           max_range_below: float = 30.0,
                  min_expected_returns: float = 1.0,
           shadow_fraction: float = 0.35,
           min_cluster_cells: int = 3,
           max_extent_m: float = 1.0,
           min_fill_ratio: float = 0.20,
           require_supported: bool = True,
           elevations: np.ndarray | None = None,
           image_width: int = 2048,
           points_xyz: np.ndarray | None = None,
           blind_ranges_m: np.ndarray | None = None) -> DetectionResult:
    """Emit negative-obstacle instances under the §4 emission rule.

    The two kinds carry different evidence and get different ranges.
    `missing_return` rests on absence, and past ~25 m the beam-ring gap
    exceeds the feature size, so absence there says nothing (§2).
    `below_plane` rests on measured returns sitting below the plane — direct
    evidence that degrades more slowly, so it is allowed further out. One
    shared radius was tried first and cost below_plane 5-8% of its candidates
    to remove 1.5-1.9% of missing-return ones: the wrong trade.
    """
    if depth_threshold <= 0 or min_cluster_cells < 1 or max_extent_m <= 0:
        raise ValueError("Positive depth, cluster size and extent required")
    # Repeated analyses must not retain flags from an earlier threshold/run.
    for block in grid.blocks.values():
        block.neg_obstacle_conf = np.zeros(len(block), dtype=np.float32)
    ground = build_ground_field(grid, tile_size=tile_size,
                                max_range=max(max_range_missing, max_range_below))
    res = DetectionResult(sensor_height=ground.sensor_height)

    if not ground.planes:
        res.missing_return_status = "unavailable: no accepted local ground planes"
        log.info("No ground planes fit; no negative-obstacle inference this frame.")
        return res

    below, gcounts, oor = _below_plane_candidates(grid, ground, depth_threshold,
                                                  require_supported, max_range_below)
    res.candidates.extend(below)
    res.ground_counts = gcounts
    res.below_plane_out_of_range = oor

    if ground.global_plane is None:
        res.missing_return_status = "unavailable: no accepted bootstrap plane"
        return res
    res.missing_return_status = "approximate beam model; blind mask supplied" if blind_ranges_m is not None else "approximate beam model; self-occlusion uncalibrated"
    occlusion = _OcclusionIndex(grid)
    vis = compute_visibility(grid, ground, ring_config, occlusion,
                             elevations=elevations, image_width=image_width,
                             max_range=max_range_missing,
                             require_supported=require_supported,
                             points_xyz=points_xyz, blind_ranges_m=blind_ranges_m)
    res.visibility_counts = vis.counts
    res.inconsistent_rays = vis.inconsistent
    res.unsampled_cells = unsampled_cell_estimate(ring_config, max_range_missing,
                                                  vis.reached_cells)

    sel = (vis.state == EXPECTED_EMPTY) & (vis.expected_returns >= min_expected_returns)
    if not np.any(sel):
        return res

    h = ground.sensor_height or 1.73
    x, y = vis.x[sel], vis.y[sel]
    ring_id, size = vis.ring_id[sel], vis.cell_size[sel]
    expected = vis.expected_returns[sel]
    cx = np.floor(x / size).astype(np.int64)
    cy = np.floor(y / size).astype(np.int64)

    for rid in np.unique(ring_id):
        m = ring_id == rid
        # One entry per cell, not per ray.
        codes = (cx[m] + (1 << 30)) * (1 << 31) + (cy[m] + (1 << 30))
        uniq, first = np.unique(codes, return_index=True)
        ux, uy = x[m][first], y[m][first]
        ucx, ucy = cx[m][first], cy[m][first]
        uexp = expected[m][first]
        usize = float(size[m][first][0])

        labels = _components(uniq, ucx, ucy)
        for lab in np.unique(labels):
            k = labels == lab
            n = int(k.sum())
            xs, ys = ux[k], uy[k]
            r_mean = float(np.hypot(xs, ys).mean())
            # Predicted shadow of a depth_threshold depression at this range.
            shadow = depth_threshold * r_mean / max(h, 1e-6)
            need = max(min_cluster_cells, int(shadow_fraction * shadow / usize))
            if n < need:
                res.clusters_rejected_small += 1
                continue
            extent = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + usize)
            # A depression's shadow is a contiguous patch. A sparse scatter of
            # isolated misses linked diagonally across metres is road noise,
            # and 8-connectivity alone will happily chain it into one huge
            # "candidate": the two largest clusters in an untuned frame spanned
            # 7 m at fill ratios of 0.08 and 0.06, against 0.25-0.30 for
            # plausible ones. Extent is also bounded by the configured maximum
            # pothole size - anything larger is a different phenomenon.
            span_x = float(xs.max() - xs.min() + usize)
            span_y = float(ys.max() - ys.min() + usize)
            fill = n / max(1.0, (span_x * span_y) / (usize * usize))
            if extent > max_extent_m or fill < min_fill_ratio:
                res.clusters_rejected_shape += 1
                continue
            # Evidence strength: how much the cluster matches its predicted
            # shadow, and how much return evidence was expected per cell.
            shape_term = float(np.clip(extent / max(shadow, 1e-6), 0.0, 1.0))
            evidence_term = float(np.clip(uexp[k].mean() / 3.0, 0.0, 1.0))
            conf = float(np.clip(0.5 * shape_term + 0.5 * evidence_term, 0.0, 1.0))
            res.candidates.append(NegativeObstacle(
                centroid=(float(xs.mean()), float(ys.mean())),
                size_m=(float(xs.max() - xs.min() + usize),
                        float(ys.max() - ys.min() + usize)),
                n_cells=n,
                confidence=conf,
                kind="missing_return",
                depth=0.0,
                mean_range=r_mean,
                expected_returns=float(uexp[k].mean()),
                shadow_predicted_m=shadow,
            ))
    return res
