"""KITTI odometry / SemanticKITTI I/O.

Layout note (spec 11.2): the calib archive ships poses at top-level
poses/XX.txt, NOT at sequences/XX/poses.txt where every module expects them.
Move them, then SequenceReader verifies scan count == pose count == label
count and aborts naming all three on mismatch (T-19) — a truncated extract is
common and fails silently otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.errors import DatasetError


def read_scan(path: str | Path) -> np.ndarray:
    """Read a .bin sweep: flat float32 (x, y, z, intensity), no header.
    Rejects files whose size is not divisible by 16, reporting the byte count."""
    path = Path(path)
    size = path.stat().st_size
    if size % 16 != 0:
        raise DatasetError(
            f"Scan '{path}' has {size} bytes, not divisible by 16 "
            "(expected flat float32 x,y,z,intensity records)."
        )
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_labels(path: str | Path) -> np.ndarray:
    """Read a .label file. Lower 16 bits are the raw semantic ID; upper 16 the
    instance ID (discarded here)."""
    raw = np.fromfile(Path(path), dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.int32)


def load_calib(path: str | Path) -> np.ndarray:
    """Return Tr (velodyne -> left camera) as a 4x4 homogeneous matrix."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Tr:") or line.startswith("Tr "):
                vals = np.array([float(v) for v in line.split(":", 1)[1].split()])
                if vals.size != 12:
                    raise DatasetError(f"Calib '{path}': Tr has {vals.size} values, expected 12.")
                tr = np.eye(4)
                tr[:3, :] = vals.reshape(3, 4)
                return tr
    raise DatasetError(f"Calib '{path}' contains no 'Tr' line.")


def load_poses(path: str | Path) -> np.ndarray:
    """Return [N, 4, 4] left-camera poses from a KITTI poses.txt.
    These are CAMERA poses — Module F must conjugate with Tr (spec 11.1)."""
    data = np.loadtxt(Path(path)).reshape(-1, 12)
    poses = np.tile(np.eye(4), (data.shape[0], 1, 1))
    poses[:, :3, :] = data.reshape(-1, 3, 4)
    return poses


class SequenceReader:
    """One KITTI sequence: calib, poses, scans, labels, with count verification."""

    def __init__(self, data_root: str | Path, sequence: str, require_labels: bool = True):
        self.sequence = sequence
        self.dir = Path(data_root) / "sequences" / sequence
        if not self.dir.is_dir():
            raise DatasetError(f"Sequence directory '{self.dir}' does not exist.")

        self.scan_paths = sorted((self.dir / "velodyne").glob("*.bin"))
        self.label_paths = sorted((self.dir / "labels").glob("*.label"))
        poses_path = self.dir / "poses.txt"
        if not poses_path.is_file():
            raise DatasetError(
                f"'{poses_path}' missing. The calib archive ships poses at top-level "
                f"poses/{sequence}.txt — move it to sequences/{sequence}/poses.txt."
            )
        self.calib = load_calib(self.dir / "calib.txt")
        self.poses = load_poses(poses_path)

        n_scans, n_poses, n_labels = len(self.scan_paths), len(self.poses), len(self.label_paths)
        counts_ok = n_scans == n_poses and (not require_labels or n_scans == n_labels)
        if n_scans == 0 or not counts_ok:
            raise DatasetError(
                f"Sequence {sequence}: scan count ({n_scans}) == pose count ({n_poses}) "
                f"== label count ({n_labels}) does not hold. A truncated extract is "
                "common and fails silently otherwise — aborting."
            )

    def __len__(self) -> int:
        return len(self.scan_paths)

    def scan(self, i: int) -> np.ndarray:
        return read_scan(self.scan_paths[i])

    def labels(self, i: int) -> np.ndarray:
        return read_labels(self.label_paths[i])

    def pose(self, i: int) -> np.ndarray:
        return self.poses[i]
