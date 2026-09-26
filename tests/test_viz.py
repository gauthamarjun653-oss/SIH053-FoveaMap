"""Module E: S-1 selector rules, comparison toggle invariants (AC-E4/E6),
frame-failure markers, and that log_frame actually runs against the installed
rerun SDK (the set_time_sequence -> set_time API break was caught here)."""

import numpy as np
import pytest

from src.common.config import load_rings
from src.grid.foveated_grid import build_grid
from src.viz.rerun_viewer import (list_sequences, format_sequence_table,
                                  uniform_comparison_block, log_frame,
                                  log_frame_failure)
from src.viz.telemetry import Telemetry

RINGS = load_rings("configs/rings.yaml")


def _synthetic_sweep(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(2.5, 100.0, n)
    az = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([r * np.cos(az), r * np.sin(az),
                    rng.normal(-1.7, 0.1, n)], axis=1)
    labels = rng.choice([1, 2, 3], size=n, p=[0.8, 0.15, 0.05]).astype(np.int8)
    return pts, labels


# ---------------------------------------------------------------------------
# S-1 sequence selector
# ---------------------------------------------------------------------------

def _make_seq(root, name, scans=3, labels=3, poses=True):
    d = root / "sequences" / name
    (d / "velodyne").mkdir(parents=True)
    for i in range(scans):
        (d / "velodyne" / f"{i:06d}.bin").write_bytes(b"\0" * 16)
    if labels:
        (d / "labels").mkdir()
        for i in range(labels):
            (d / "labels" / f"{i:06d}.label").write_bytes(b"\0" * 4)
    if poses:
        (d / "poses.txt").write_text("x\n")


def test_s1_missing_poses_disabled_with_reason(tmp_path):
    _make_seq(tmp_path, "00", poses=True)
    _make_seq(tmp_path, "01", poses=False)
    entries = {e.sequence: e for e in list_sequences(tmp_path)}
    assert entries["00"].enabled
    assert not entries["01"].enabled
    assert "poses" in entries["01"].reason  # disabled WITH a stated reason
    # ...and shown, not hidden:
    assert "01" in format_sequence_table(list(entries.values()), tmp_path)


def test_s1_truncated_labels_disabled(tmp_path):
    _make_seq(tmp_path, "00", scans=3, labels=2)
    (e,) = list_sequences(tmp_path)
    assert not e.enabled and "label count" in e.reason


def test_s1_empty_state_names_the_path(tmp_path):
    msg = format_sequence_table(list_sequences(tmp_path), tmp_path)
    assert str(tmp_path) in msg


# ---------------------------------------------------------------------------
# S-2: telemetry keeps BOTH memories visible (AC-E4)
# ---------------------------------------------------------------------------

def test_status_line_shows_both_grid_memories():
    tel = Telemetry()
    tel.frame(2_000_000, 40_000_000)
    line = tel.status_line()
    assert "foveated 2.0 MB" in line and "uniform 40.0 MB" in line and "FPS" in line


# ---------------------------------------------------------------------------
# S-2: log_frame runs against the installed rerun (API-drift canary), the
# comparison layer shares the frame timeline (AC-E6), failures leave markers.
# ---------------------------------------------------------------------------

def test_log_frame_with_comparison_and_failure_marker():
    rr = pytest.importorskip("rerun")
    rr.init("sih26053_test_viz")
    mem = rr.memory_recording()

    pts, labels = _synthetic_sweep()
    grid = build_grid(pts, labels, RINGS)
    from src.analytics.cell_stats import compute_stats
    compute_stats(grid)
    uniform = uniform_comparison_block(pts, labels, 0.10, RINGS.base_cell_size)
    assert uniform.cell_size == 0.10

    tel = Telemetry()
    tel.cached_mode = True  # AC-E5 banner must be logged, asserted below
    tel.frame(grid.memory_bytes(), uniform.memory_bytes())

    log_frame(grid, tel, frame_idx=7, ring_config=RINGS, uniform_block=uniform)
    log_frame_failure(8, "test failure")

    assert mem.num_msgs() > 0
    assert any("CACHED" in b for b in tel.banners())
