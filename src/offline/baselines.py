"""Baseline representations for the Pareto comparison (spec 11.4).

Both measure their memory from actual array footprints. Hardcoded memory
figures are on the what-NOT-to-build list: measure, never assert.
"""

from __future__ import annotations

import numpy as np

from src.segmentation.class_map import STATIC, DYNAMIC


class UniformVoxel3D:
    """Uniform 5 cm 3D voxel grid — high-fidelity, high-cost reference.
    Sparse SoA: voxel indices + per-voxel obstacle flag."""

    def __init__(self, points_xyz: np.ndarray, labels3: np.ndarray, voxel: float = 0.05):
        self.voxel = voxel
        if points_xyz.shape[0] == 0:
            self.ijk = np.zeros((0, 3), dtype=np.int32)
            self.obstacle = np.zeros(0, dtype=bool)
            self.count = np.zeros(0, dtype=np.uint16)
            return
        ijk = np.floor(points_xyz / voxel).astype(np.int64)
        key = ((ijk[:, 0] + (1 << 20)) << 42) | ((ijk[:, 1] + (1 << 20)) << 21) | (ijk[:, 2] + (1 << 20))
        order = np.argsort(key, kind="stable")
        k_s = key[order]
        is_obst = np.isin(labels3[order], (STATIC, DYNAMIC))
        starts = np.flatnonzero(np.r_[True, k_s[1:] != k_s[:-1]])
        ends = np.r_[starts[1:], k_s.shape[0]]
        self.ijk = ijk[order][starts].astype(np.int32)
        self.count = np.minimum(ends - starts, np.iinfo(np.uint16).max).astype(np.uint16)
        self.obstacle = np.logical_or.reduceat(is_obst, starts)

    def memory_bytes(self) -> int:
        return self.ijk.nbytes + self.obstacle.nbytes + self.count.nbytes

    def obstacle_columns(self) -> set[tuple[int, int]]:
        """2D (x, y) voxel-column indices containing >= 1 obstacle voxel, at
        the voxel resolution (== the recall reference lattice for 5 cm)."""
        obst = self.ijk[self.obstacle]
        return set(zip(obst[:, 0].tolist(), obst[:, 1].tolist()))


class UniformOccupancy2D:
    """Uniform 2D occupancy grid — cheap, height-free reference.

    Stores binary occupancy only. Since per-cell height statistics are
    discarded, the only obstacle rule available is a single global height cut
    (returns above `obstacle_z_cut` in the sensor frame). Low obstacles, kerbs,
    and anything below the cut are indistinguishable from ground — which is
    precisely why the spec expects this baseline's recall to be poor.
    """

    def __init__(self, points_xyz: np.ndarray, cell: float = 0.05,
                 obstacle_z_cut: float = -1.2):
        self.cell = cell
        if points_xyz.shape[0] == 0:
            self.cxy = np.zeros((0, 2), dtype=np.int32)
            self.obstacle = np.zeros(0, dtype=bool)
            return
        cxy = np.floor(points_xyz[:, :2] / cell).astype(np.int64)
        key = (cxy[:, 0] + (1 << 30)) * (1 << 31) + (cxy[:, 1] + (1 << 30))
        order = np.argsort(key, kind="stable")
        k_s = key[order]
        above = points_xyz[order, 2] > obstacle_z_cut
        starts = np.flatnonzero(np.r_[True, k_s[1:] != k_s[:-1]])
        self.cxy = cxy[order][starts].astype(np.int32)
        self.obstacle = np.logical_or.reduceat(above, starts)

    def memory_bytes(self) -> int:
        return self.cxy.nbytes + self.obstacle.nbytes

    def obstacle_cells(self) -> set[tuple[int, int]]:
        obst = self.cxy[self.obstacle]
        return set(zip(obst[:, 0].tolist(), obst[:, 1].tolist()))
