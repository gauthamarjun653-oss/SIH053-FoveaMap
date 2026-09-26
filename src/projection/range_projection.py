"""Module A — Ingestion and spherical projection.

project(points, config) -> ProjectionResult with:
  range_image float32[H, W, 5]   channels [range, x, y, z, intensity]
  pixel_map   int32[N, 2]        one (u, v) per input point; (-1, -1) if discarded
  mask        bool[H, W]         pixel occupied

Range-gated to [range_min, range_max]; discards are COUNTED, never silent
(AC-A2: assigned + discarded == N). On pixel collision the nearest return wins;
losers keep their (u, v) in pixel_map so back-projection labels them with the
winning pixel's class. Deterministic: identical input -> bit-identical output
(AC-A3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import ProjectionConfig


@dataclass
class ProjectionResult:
    range_image: np.ndarray   # float32 [H, W, 5]
    pixel_map: np.ndarray     # int32   [N, 2]  (u, v); (-1,-1) for discarded points
    mask: np.ndarray          # bool    [H, W]
    n_points: int
    n_assigned: int
    n_discarded: int
    n_collisions: int         # in-range points that lost their pixel to a nearer return


def project(points: np.ndarray, cfg: ProjectionConfig) -> ProjectionResult:
    """points: float32 [N, 4] = (x, y, z, intensity), sensor frame."""
    n = points.shape[0]
    H, W = cfg.height, cfg.width
    range_image = np.zeros((H, W, 5), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    pixel_map = np.full((n, 2), -1, dtype=np.int32)

    if n == 0:
        return ProjectionResult(range_image, pixel_map, mask, 0, 0, 0, 0)

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)
    intensity = points[:, 3].astype(np.float64)

    r = np.sqrt(x * x + y * y + z * z)
    in_range = (r >= cfg.range_min) & (r <= cfg.range_max)
    n_discarded = int(np.count_nonzero(~in_range))

    fov_up = np.deg2rad(cfg.fov_up_deg)
    fov_down = np.deg2rad(cfg.fov_down_deg)
    fov_total = abs(fov_up) + abs(fov_down)

    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.floor(0.5 * W * (1.0 - np.arctan2(y, x) / np.pi)).astype(np.int64)
        v = np.floor(H * (1.0 - (np.arcsin(np.clip(z / np.where(r > 0, r, 1.0), -1, 1))
                                 + abs(fov_down)) / fov_total)).astype(np.int64)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    idx = np.flatnonzero(in_range)
    pixel_map[idx, 0] = u[idx]
    pixel_map[idx, 1] = v[idx]

    # Nearest return wins each pixel: scatter in order of DECREASING range so
    # the nearest point writes last. Ties broken deterministically by original
    # point index (stable sort), satisfying AC-A3.
    order = idx[np.argsort(-r[idx], kind="stable")]
    uu, vv = u[order], v[order]
    range_image[vv, uu, 0] = r[order]
    range_image[vv, uu, 1] = x[order]
    range_image[vv, uu, 2] = y[order]
    range_image[vv, uu, 3] = z[order]
    range_image[vv, uu, 4] = intensity[order]
    mask[vv, uu] = True

    n_assigned = idx.size
    n_collisions = n_assigned - int(np.count_nonzero(mask))

    assert n_assigned + n_discarded == n, "AC-A2 violated: point accounting broken"
    return ProjectionResult(range_image, pixel_map, mask,
                            n, n_assigned, n_discarded, n_collisions)


def back_project(label_image: np.ndarray, pixel_map: np.ndarray) -> np.ndarray:
    """Assign the winning pixel's label to ALL points that mapped to it,
    winners and losers alike. Discarded points (pixel (-1,-1)) get IGNORE (0),
    and remain accounted for via n_discarded."""
    n = pixel_map.shape[0]
    labels = np.zeros(n, dtype=np.int8)
    valid = pixel_map[:, 0] >= 0
    labels[valid] = label_image[pixel_map[valid, 1], pixel_map[valid, 0]]
    return labels
