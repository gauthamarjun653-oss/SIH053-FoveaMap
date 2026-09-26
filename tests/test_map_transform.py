"""T-17 / AC-F1: the inv(Tr) conjugation, and the z-extent assertion that
catches its omission — the highest-consequence low-visibility bug in the
codebase."""

import numpy as np
import pytest

from src.common.errors import DatasetError
from src.offline.build_map import assert_z_extent, velo_to_world

# KITTI-like velodyne -> camera rotation: cam_x = -velo_y, cam_y = -velo_z,
# cam_z = velo_x (camera y points DOWN).
TR = np.eye(4)
TR[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)


def _scene():
    """Urban-ish velodyne-frame scene: ground at z=-1.7, buildings to z=+12."""
    rng = np.random.default_rng(0)
    ground = np.stack([rng.uniform(-40, 40, 4000), rng.uniform(-40, 40, 4000),
                       rng.normal(-1.7, 0.05, 4000)], axis=1)
    walls = np.stack([rng.uniform(-40, 40, 1000), rng.uniform(-40, 40, 1000),
                      rng.uniform(0, 12, 1000)], axis=1)
    return np.concatenate([ground, walls])


def _camera_poses(n=60):
    """Camera poses corresponding to pure forward velodyne motion."""
    poses = []
    for i in range(n):
        t_velo = np.eye(4)
        t_velo[0, 3] = float(i)  # 1 m forward per frame
        poses.append(TR @ t_velo @ np.linalg.inv(TR))
    return poses


def _accumulate(transform_fn):
    pts = _scene()
    out = []
    for pose in _camera_poses():
        T = transform_fn(pose)
        out.append(pts @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(out)


def test_correct_conjugation_passes_z_extent():
    world = _accumulate(lambda pose: velo_to_world(TR, pose))
    extent = assert_z_extent(world, 10.0, 40.0)
    assert 13.0 < extent < 15.0  # ground -1.7 .. walls +12


def test_t17_missing_outer_inverse_is_caught():
    # The bug: applying the camera pose with only the inner Tr. The map still
    # registers, but sits in the camera convention — z-extent explodes because
    # camera z is the driving direction.
    world = _accumulate(lambda pose: pose @ TR)
    with pytest.raises(DatasetError):
        assert_z_extent(world, 10.0, 40.0)


def test_empty_map_rejected():
    with pytest.raises(DatasetError):
        assert_z_extent(np.zeros((0, 3)))


def test_sparse_phantom_outliers_do_not_fail_a_correct_map():
    # Mirror reflections put a handful of points tens of meters below ground
    # (observed on seq 00: raw extent 53.8 m vs robust 22.4 m). The robust
    # extent must ignore them...
    world = _accumulate(lambda pose: velo_to_world(TR, pose))
    phantoms = np.array([[5.0, 5.0, -30.0], [8.0, 2.0, -25.0], [1.0, 9.0, 35.0]])
    extent = assert_z_extent(np.concatenate([world, phantoms]), 10.0, 40.0)
    assert 13.0 < extent < 15.0


def test_camera_frame_bug_still_caught_despite_robust_extent():
    # ...while the camera-frame bug shifts the BULK of the distribution, so
    # the percentile trim must not hide it.
    world = _accumulate(lambda pose: pose @ TR)
    with pytest.raises(DatasetError):
        assert_z_extent(world, 10.0, 40.0)
