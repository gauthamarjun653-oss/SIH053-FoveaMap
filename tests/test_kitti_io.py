"""T-19: scan/pose/label count mismatch aborts naming all three."""

import numpy as np
import pytest

from src.common.errors import DatasetError
from src.io.kitti_reader import SequenceReader, load_calib, load_poses


def _make_sequence(root, seq="00", n_scans=2, n_poses=2, n_labels=2):
    d = root / "sequences" / seq
    (d / "velodyne").mkdir(parents=True)
    (d / "labels").mkdir()
    for i in range(n_scans):
        np.zeros((4, 4), dtype=np.float32).tofile(d / "velodyne" / f"{i:06d}.bin")
    for i in range(n_labels):
        np.zeros(4, dtype=np.uint32).tofile(d / "labels" / f"{i:06d}.label")
    tr = "Tr: " + " ".join(str(v) for v in np.eye(3, 4).ravel())
    (d / "calib.txt").write_text(tr + "\n", encoding="utf-8")
    pose_line = " ".join(str(v) for v in np.eye(3, 4).ravel())
    (d / "poses.txt").write_text("\n".join([pose_line] * n_poses) + "\n", encoding="utf-8")
    return d


def test_matching_counts_ok(tmp_path):
    _make_sequence(tmp_path)
    reader = SequenceReader(tmp_path, "00")
    assert len(reader) == 2
    assert reader.scan(0).shape == (4, 4)
    assert reader.labels(1).shape == (4,)
    assert reader.pose(0).shape == (4, 4)


def test_t19_mismatch_aborts_naming_all_three(tmp_path):
    _make_sequence(tmp_path, n_scans=3, n_poses=2, n_labels=1)
    with pytest.raises(DatasetError) as e:
        SequenceReader(tmp_path, "00")
    msg = str(e.value)
    assert "3" in msg and "2" in msg and "1" in msg
    assert "scan" in msg and "pose" in msg and "label" in msg


def test_missing_poses_names_the_layout_fix(tmp_path):
    d = _make_sequence(tmp_path)
    (d / "poses.txt").unlink()
    with pytest.raises(DatasetError) as e:
        SequenceReader(tmp_path, "00")
    assert "poses/00.txt" in str(e.value)


def test_calib_and_pose_shapes(tmp_path):
    d = _make_sequence(tmp_path)
    assert load_calib(d / "calib.txt").shape == (4, 4)
    assert load_poses(d / "poses.txt").shape == (2, 4, 4)
