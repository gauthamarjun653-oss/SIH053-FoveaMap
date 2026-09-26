"""Recall metric (spec section 10) and the Pareto curve (spec 11.5).

The reference is a fixed 5 cm lattice, independent of any representation, so
the denominator is identical for all three configurations. Counting a
representation's own cells would give each configuration a different
denominator and produce a meaningless, near-tautological comparison.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.common.config import RingConfig
from src.grid.foveated_grid import FoveatedGrid
from src.grid.rings import assign_ring_scalar, OUT_OF_RANGE
from src.segmentation.class_map import STATIC, DYNAMIC

BAND_KEYS = ("0-10", "10-25", "25-50", "50-100")


def band_of(r: float, bands: tuple) -> int:
    for i, (lo, hi) in enumerate(bands):
        if lo <= r < hi or (i == len(bands) - 1 and r == hi):
            return i
    return -1


def reference_obstacle_cells(points_xyz: np.ndarray, gt_labels3: np.ndarray,
                             ref_cell: float, bands: tuple) -> dict[int, set[tuple[int, int]]]:
    """Ground-truth obstacle cells: reference cells containing >= 1 point
    labelled STATIC or DYNAMIC in the SemanticKITTI ground truth, keyed by
    distance band of the cell centre."""
    out: dict[int, set] = {i: set() for i in range(len(bands))}
    obst = np.isin(gt_labels3, (STATIC, DYNAMIC))
    if not np.any(obst):
        return out
    pts = points_xyz[obst]
    cx = np.floor(pts[:, 0] / ref_cell).astype(np.int64)
    cy = np.floor(pts[:, 1] / ref_cell).astype(np.int64)
    for a, b in set(zip(cx.tolist(), cy.tolist())):
        r = float(np.hypot((a + 0.5) * ref_cell, (b + 0.5) * ref_cell))
        band = band_of(r, bands)
        if band >= 0:
            out[band].add((a, b))
    return out


def foveated_obstacle_lookup(grid: FoveatedGrid, min_count: int = 1) -> dict[int, set[tuple[int, int]]]:
    """Per ring: cells flagged obstacle-occupied (>= min_count STATIC/DYNAMIC
    returns in the histogram — counts, never a majority vote)."""
    out = {}
    for ring_id, block in grid.blocks.items():
        flagged = (block.hist[:, STATIC].astype(np.int64)
                   + block.hist[:, DYNAMIC].astype(np.int64)) >= min_count
        out[ring_id] = set(zip(block.cx[flagged].tolist(), block.cy[flagged].tolist()))
    return out


def foveated_detects(ref_cell_idx: tuple[int, int], ref_cell: float,
                     ring_config: RingConfig, lookup: dict[int, set]) -> bool:
    """A representation detects a reference cell if the containing cell in that
    representation is flagged obstacle-occupied."""
    x = (ref_cell_idx[0] + 0.5) * ref_cell
    y = (ref_cell_idx[1] + 0.5) * ref_cell
    ring_id = assign_ring_scalar(float(np.hypot(x, y)), ring_config)
    if ring_id == OUT_OF_RANGE or ring_id not in lookup:
        return False
    size = ring_config.ring_by_id(ring_id).cell_size
    key = (int(np.floor(x / size)), int(np.floor(y / size)))
    return key in lookup[ring_id]


class RecallAccumulator:
    """Accumulates detected/total reference cells per band per configuration
    across frames."""

    def __init__(self, config_names: list[str], n_bands: int):
        self.total = np.zeros(n_bands, dtype=np.int64)
        self.detected = {name: np.zeros(n_bands, dtype=np.int64) for name in config_names}

    def add_frame(self, reference: dict[int, set], detectors: dict[str, callable]):
        for band, cells in reference.items():
            self.total[band] += len(cells)
            for name, detect in detectors.items():
                self.detected[name][band] += sum(1 for c in cells if detect(c))

    def recall(self) -> dict[str, list[float]]:
        out = {}
        for name, det in self.detected.items():
            with np.errstate(invalid="ignore"):
                r = np.where(self.total > 0, det / np.maximum(self.total, 1), np.nan)
            out[name] = [float(v) for v in r]
        return out


def export_results(out_dir: Path, recall: dict[str, list[float]],
                   memory_mb: dict[str, float], total_cells: list[int],
                   bands: tuple, provenance: dict) -> None:
    """CSV + JSON + PNG, regenerable from a single command (PAC-6)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bands_m": [list(b) for b in bands],
        "reference_cells_per_band": [int(t) for t in total_cells],
        "recall": recall,
        "memory_mb": memory_mb,
        "provenance": provenance,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "memory_mb"] + [f"recall_{k}m" for k in BAND_KEYS])
        for name in recall:
            w.writerow([name, f"{memory_mb[name]:.3f}"] + [f"{v:.4f}" for v in recall[name]])

    plot_pareto(out_dir / "pareto.png", recall, memory_mb, bands)


def plot_pareto(path: Path, recall: dict[str, list[float]],
                memory_mb: dict[str, float], bands: tuple) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {"foveated": "o", "voxel_5cm": "s", "occupancy_2d": "^"}
    fig, axes = plt.subplots(1, len(bands), figsize=(4.2 * len(bands), 4), sharey=True)
    if len(bands) == 1:
        axes = [axes]
    for i, ((lo, hi), ax) in enumerate(zip(bands, axes)):
        for name in recall:
            ax.scatter(memory_mb[name], recall[name][i],
                       marker=markers.get(name, "x"), s=90, label=name)
        ax.set_title(f"{lo:.0f}–{hi:.0f} m")
        ax.set_xlabel("memory (MB)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("obstacle recall")
    axes[0].legend()
    fig.suptitle("Memory vs recall, stratified by distance band")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
