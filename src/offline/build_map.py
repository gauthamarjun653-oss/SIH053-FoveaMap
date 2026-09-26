"""Module F — global map construction (offline, run once; NOT part of the
deployed system — keep that separation visible in every diagram).

poses.txt gives the pose of the LEFT CAMERA, not the LiDAR:

    T_i = inv(Tr) @ pose_i @ Tr        # OUTER INVERSE IS MANDATORY

Omitting the outer inv(Tr) yields a map that registers correctly and looks
entirely plausible — but sits in the camera convention with y pointing down,
silently corrupting every elevation statistic downstream. AC-F1 (z-extent
10-40 m for an urban sequence) is what catches it (T-17).
"""

from __future__ import annotations

import logging

import numpy as np

from src.common.errors import DatasetError
from src.io.kitti_reader import SequenceReader
from src.segmentation.class_map import build_raw_to_class3, DYNAMIC

log = logging.getLogger(__name__)


def velo_to_world(Tr: np.ndarray, pose_cam: np.ndarray) -> np.ndarray:
    """Velodyne-frame -> world transform for one frame. The outer inverse is
    mandatory (spec 11.1)."""
    return np.linalg.inv(Tr) @ pose_cam @ Tr


def _voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    ijk = np.floor(points / voxel).astype(np.int64)
    # Pack into one int64; 21 bits per axis covers +/- 1e6 cells.
    return ((ijk[:, 0] + (1 << 20)) << 42) | ((ijk[:, 1] + (1 << 20)) << 21) | (ijk[:, 2] + (1 << 20))


def build_global_map(reader: SequenceReader, voxel: float = 0.10,
                     stride: int = 1, max_frames: int | None = None) -> np.ndarray:
    """Aggregate deduplicated world-frame points for a sequence.

    Dynamic-class points are excluded (moving cars smear into comet trails) and
    deduplication is INCREMENTAL — stacking 4541 x 120k points before
    downsampling is ~6.5 GB and is exactly what the spec says not to do.
    """
    raw_to_3 = build_raw_to_class3()
    seen: set[int] = set()
    kept: list[np.ndarray] = []
    n = len(reader) if max_frames is None else min(len(reader), max_frames)
    for i in range(0, n, stride):
        pts = reader.scan(i)
        labels3 = raw_to_3[reader.labels(i)]
        static_mask = labels3 != DYNAMIC
        xyz = pts[static_mask, :3].astype(np.float64)
        T = velo_to_world(reader.calib, reader.pose(i))
        world = xyz @ T[:3, :3].T + T[:3, 3]

        keys = _voxel_keys(world, voxel)
        uniq, first_idx = np.unique(keys, return_index=True)
        new = np.array([k not in seen for k in uniq.tolist()])
        if np.any(new):
            kept.append(world[first_idx[new]])
            seen.update(uniq[new].tolist())
        if i % 200 == 0:
            log.info("map: frame %d/%d, %d voxels", i, n, len(seen))
    return np.concatenate(kept, axis=0) if kept else np.zeros((0, 3))


def assert_z_extent(map_points: np.ndarray, lo: float = 10.0, hi: float = 40.0) -> float:
    """AC-F1: z-extent within [lo, hi] for an urban sequence. This is the check
    that catches the inv(Tr) bug — run it before anything is derived from the
    map.

    Extent is p99.9 - p0.1, not raw max - min: real sequences carry a few
    mirror-reflection phantoms tens of meters below ground (seq 00 measures
    22.4 m robust vs 53.8 m raw), while the camera-frame bug this gate exists
    for shifts the BULK of the distribution to the driving-direction span
    (hundreds of meters), which no percentile trim hides."""
    if map_points.shape[0] == 0:
        raise DatasetError("Global map is empty; cannot validate z-extent.")
    z_lo, z_hi = np.percentile(map_points[:, 2], [0.1, 99.9])
    extent = float(z_hi - z_lo)
    raw = float(map_points[:, 2].max() - map_points[:, 2].min())
    if not (lo <= extent <= hi):
        raise DatasetError(
            f"Global map z-extent {extent:.1f} m (p99.9-p0.1; raw {raw:.1f} m) "
            f"outside [{lo}, {hi}] m. "
            "Suspect the inv(Tr) conjugation (camera-frame map, y pointing down) "
            "or a corrupted pose file. Do not derive anything from this map."
        )
    if raw > extent + 10.0:
        log.warning("z-extent: raw %.1f m vs robust %.1f m — sparse outlier "
                    "points present (mirror reflections); downstream elevation "
                    "statistics must stay percentile-based.", raw, extent)
    return extent


def wall_thickness_stats(map_points: np.ndarray, cell: float = 0.5,
                         min_points: int = 40, min_z_extent: float = 2.5) -> float:
    """AC-F2: building facade thickness (wall-normal standard deviation) should
    be < 15 cm. Wall-like columns are 0.5 m 2D cells with tall z-extent; the
    thickness is the point spread along the minor horizontal PCA axis. Returns
    the median across facade cells (T-18)."""
    if map_points.shape[0] == 0:
        raise DatasetError("Global map is empty; cannot estimate wall thickness.")
    cx = np.floor(map_points[:, 0] / cell).astype(np.int64)
    cy = np.floor(map_points[:, 1] / cell).astype(np.int64)
    key = cx * 1_000_003 + cy
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    ends = np.r_[starts[1:], sorted_key.shape[0]]
    thicknesses = []
    for s, e in zip(starts, ends):
        if e - s < min_points:
            continue
        pts = map_points[order[s:e]]
        if pts[:, 2].max() - pts[:, 2].min() < min_z_extent:
            continue
        xy = pts[:, :2] - pts[:, :2].mean(axis=0)
        cov = xy.T @ xy / xy.shape[0]
        eigvals = np.linalg.eigvalsh(cov)
        thicknesses.append(float(np.sqrt(max(eigvals[0], 0.0))))
    if not thicknesses:
        raise DatasetError("No facade-like cells found; cannot estimate wall thickness.")
    return float(np.median(thicknesses))
