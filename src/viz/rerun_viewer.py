"""Module E — rerun.io viewer (S-1 / S-2 / S-3).

The section-12 spike verified rerun renders a static variable-resolution grid
with ring-boundary overlays, so Module E is built on rerun.

Class colours are fixed and paired with a text legend (never colour alone).
Elevation uses a colour ramp. Comparison is BY TOGGLE, not split-screen (spec
S-2): the uniform grid is logged as a sibling entity on the same frame
timeline, so showing one layer or the other in the viewer cannot change the
frame index (AC-E6 holds structurally). Both grids' memory stays in the status
line regardless of which layer is visible (AC-E4).

rerun has no data-driven slider, so the negative-obstacle confidence threshold
(spec S-2) is exposed as toggleable confidence tiers — one entity per
threshold — which is the same interaction with coarser steps.

Banners from Telemetry are logged as always-visible text (AC-E5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.config import RingConfig
from src.grid.foveated_grid import FoveatedGrid, CellBlock, build_block
from src.analytics.cell_stats import render_class
from src.segmentation.class_map import CLASS_NAMES
from src.viz.telemetry import Telemetry

# Fixed, distinct class colours (RGB), paired with the on-screen legend text.
CLASS_COLORS = np.array([
    [128, 128, 128],   # IGNORE  - grey
    [60, 120, 216],    # ROAD    - blue
    [240, 180, 40],    # STATIC  - amber
    [230, 60, 60],     # DYNAMIC - red
], dtype=np.uint8)

# Toggleable stand-in for a confidence slider (see module docstring).
NEG_CONF_TIERS = (0.3, 0.5, 0.7)


def _require_rerun():
    try:
        import rerun as rr
        return rr
    except ImportError as e:
        raise RuntimeError(
            "rerun-sdk is not installed. Install it (pip install rerun-sdk) or "
            "fall back to Open3D per spec section 12."
        ) from e


# ---------------------------------------------------------------------------
# S-1 — sequence selector
# ---------------------------------------------------------------------------

@dataclass
class SequenceEntry:
    sequence: str
    n_scans: int
    n_labels: int
    has_poses: bool
    enabled: bool
    reason: str  # why disabled; "" when enabled


def list_sequences(data_root: str | Path) -> list[SequenceEntry]:
    """S-1: every sequence directory with frame count and label availability.
    A sequence lacking poses is disabled WITH A STATED REASON, not hidden."""
    root = Path(data_root).expanduser() / "sequences"
    entries: list[SequenceEntry] = []
    if not root.is_dir():
        return entries
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n_scans = len(list((d / "velodyne").glob("*.bin"))) if (d / "velodyne").is_dir() else 0
        n_labels = len(list((d / "labels").glob("*.label"))) if (d / "labels").is_dir() else 0
        has_poses = (d / "poses.txt").is_file()
        if n_scans == 0:
            enabled, reason = False, "no velodyne scans"
        elif not has_poses:
            enabled, reason = False, "poses.txt missing (calib archive ships it at top-level poses/)"
        elif n_labels not in (0, n_scans):
            enabled, reason = False, f"label count {n_labels} != scan count {n_scans} (truncated extract?)"
        else:
            enabled, reason = True, ""
        entries.append(SequenceEntry(d.name, n_scans, n_labels, has_poses, enabled, reason))
    return entries


def format_sequence_table(entries: list[SequenceEntry], data_root: str | Path) -> str:
    """Render S-1 as text (CLI and the rerun selector panel share it).
    Empty state names the configured path."""
    if not entries:
        return (f"No sequences found under '{Path(data_root).expanduser() / 'sequences'}'. "
                "Check data_root in configs/pipeline.yaml.")
    lines = ["seq | scans | labels | poses | status"]
    for e in entries:
        status = "ok" if e.enabled else f"DISABLED: {e.reason}"
        lines.append(f"{e.sequence:>3} | {e.n_scans:>5} | {e.n_labels:>6} | "
                     f"{'yes' if e.has_poses else 'NO':>5} | {status}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# S-2 — main viewer
# ---------------------------------------------------------------------------

def _block_rects(block: CellBlock, min_class_count: int):
    """Cell centres, half-sizes, per-cell render class and mean elevation."""
    n = len(block)
    if n == 0:
        return None
    centers = np.stack([(block.cx + 0.5) * block.cell_size,
                        (block.cy + 0.5) * block.cell_size], axis=1)
    half_sizes = np.full((n, 2), block.cell_size / 2.0)
    return centers, half_sizes, render_class(block.hist, min_class_count), block.mean


def _grid_rects(grid: FoveatedGrid, min_class_count: int):
    parts = [_block_rects(b, min_class_count) for b in grid.blocks.values()]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def _elevation_ramp(elev: np.ndarray) -> np.ndarray:
    z = elev - elev.min()
    t = (z / max(float(z.max()), 1e-6)).astype(np.float32)
    return np.stack([50 + 200 * t, 50 + 100 * t, 200 - 150 * t], axis=1).astype(np.uint8)


def _log_rects(rr, path: str, rects) -> None:
    centers, half_sizes, classes, elev = rects
    rr.log(f"{path}/class", rr.Boxes2D(
        centers=centers, half_sizes=half_sizes, colors=CLASS_COLORS[classes]))
    # Elevation as a colour ramp (clearer than extrusion for this data).
    rr.log(f"{path}/elevation", rr.Boxes2D(
        centers=centers, half_sizes=half_sizes, colors=_elevation_ramp(elev)))


def uniform_comparison_block(points_xyz: np.ndarray, labels: np.ndarray,
                             cell_size: float, base_size: float) -> CellBlock:
    """The comparison grid: one uniform-resolution block over the same sweep,
    built through the same canonical dyadic fold as the foveated grid."""
    return build_block(points_xyz, labels, cell_size, base_size)


def log_frame(grid: FoveatedGrid, telemetry: Telemetry, frame_idx: int,
              ring_config: RingConfig, min_class_count: int = 1,
              show_rings: bool = True, neg_candidates=None,
              uniform_block: CellBlock | None = None) -> None:
    rr = _require_rerun()
    rr.set_time("frame", sequence=frame_idx)

    rects = _grid_rects(grid, min_class_count)
    if rects is None:
        # Empty sweep: state it on canvas, never a blank area.
        rr.log("world/grid", rr.TextDocument("No returns in range"))
    else:
        _log_rects(rr, "world/grid", rects)

    # Comparison layer: same timeline, sibling entity => toggling it in the
    # viewer cannot move the frame index (AC-E6).
    if uniform_block is not None:
        u_rects = _block_rects(uniform_block, min_class_count)
        if u_rects is not None:
            _log_rects(rr, "world/uniform_grid", u_rects)

    if show_rings:
        theta = np.linspace(0, 2 * np.pi, 128)
        for ring in ring_config.rings:
            circle = np.stack([ring.radius_max * np.cos(theta),
                               ring.radius_max * np.sin(theta)], axis=1)
            rr.log(f"world/rings/r{ring.id}", rr.LineStrips2D([circle]))

    if neg_candidates is not None:
        # Confidence tiers as toggleable entities (slider stand-in).
        for tier in NEG_CONF_TIERS:
            pts = np.array([[c.x, c.y] for c in neg_candidates if c.confidence >= tier])
            path = f"world/negative_obstacles/conf_ge_{int(tier * 100)}"
            if pts.size:
                selected = [c for c in neg_candidates if c.confidence >= tier]
                if all(hasattr(c, "size_m") for c in selected):
                    rr.log(path, rr.Boxes2D(centers=pts,
                           half_sizes=np.array([c.size_m for c in selected])/2,
                           colors=[[255, 0, 255]]))
                else:
                    rr.log(path, rr.Points2D(pts, radii=0.15, colors=[[255, 0, 255]]))
            else:
                rr.log(path, rr.Clear(recursive=False))

    legend = " | ".join(f"{name}={i}" for i, name in enumerate(CLASS_NAMES))
    rr.log("hud/legend", rr.TextDocument(f"Classes: {legend}"))
    rr.log("hud/status", rr.TextDocument(telemetry.status_line()))
    for i, banner in enumerate(telemetry.banners()):
        rr.log(f"hud/banner/{i}", rr.TextDocument(f"*** {banner} ***"))


def log_frame_failure(frame_idx: int, reason: str = "segmentation failed") -> None:
    """Timeline marker for a failed frame; playback continues (spec States)."""
    rr = _require_rerun()
    rr.set_time("frame", sequence=frame_idx)
    rr.log("world/frame_failed", rr.TextDocument(f"FRAME {frame_idx} FAILED: {reason}"))


# ---------------------------------------------------------------------------
# S-3 — benchmark results
# ---------------------------------------------------------------------------

def log_benchmark_results(out_dir: str | Path) -> None:
    """Log a finished benchmark run (results.json + pareto.png from the
    harness) into the viewer: Pareto plot, per-band table, run metadata."""
    import json
    rr = _require_rerun()
    out = Path(out_dir)
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))

    png = out / "pareto.png"
    if png.is_file():
        rr.log("benchmark/pareto", rr.EncodedImage(path=str(png)), static=True)

    bands = ["-".join(str(int(x)) for x in b) + " m" for b in payload["bands_m"]]
    lines = ["config | memory MB | " + " | ".join(f"recall {b}" for b in bands)]
    for name, rec in payload["recall"].items():
        lines.append(f"{name} | {payload['memory_mb'][name]:.2f} | "
                     + " | ".join(f"{v:.4f}" for v in rec))
    rr.log("benchmark/metrics", rr.TextDocument("\n".join(lines)), static=True)

    prov = payload.get("provenance", {})
    meta = "\n".join(f"{k}: {v}" for k, v in prov.items())
    rr.log("benchmark/provenance", rr.TextDocument(meta or "no provenance"), static=True)


# ---------------------------------------------------------------------------
# Section 12 spike
# ---------------------------------------------------------------------------

def spike(ring_config: RingConfig) -> None:
    """The section 12 verification: render a static variable-resolution grid
    with ring boundary overlays. If this fails, do not build Module E on rerun."""
    from src.grid.foveated_grid import build_grid
    rr = _require_rerun()
    rr.init("sih26053_spike", spawn=True)
    rng = np.random.default_rng(0)
    n = 60_000
    r = rng.uniform(2.5, 100.0, n)
    az = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([r * np.cos(az), r * np.sin(az),
                    rng.normal(-1.7, 0.1, n)], axis=1)
    labels = rng.choice([1, 2, 3], size=n, p=[0.8, 0.15, 0.05]).astype(np.int8)
    grid = build_grid(pts, labels, ring_config)
    tel = Telemetry()
    tel.frame(grid.memory_bytes())
    log_frame(grid, tel, 0, ring_config)
    print(f"Spike logged: {grid.n_cells} cells, {grid.memory_bytes() / 1e6:.2f} MB. "
          "Check the rerun window for the grid + ring overlays.")
