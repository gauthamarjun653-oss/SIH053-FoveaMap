"""Benchmark harness (spec 11) — regenerates the Pareto curve from a single
command with no manual steps (PAC-6).

Per frame: project -> segment (live / random / gt-labels) -> back-project ->
build all three representations from the SAME predicted labels, score recall
against the ground-truth reference lattice, and measure each representation's
memory from its actual arrays. Provenance is collected up front; the run aborts
without it and refuses a cached inference path.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.common.config import PipelineConfig, RingConfig
from src.grid.foveated_grid import build_grid
from src.io.kitti_reader import SequenceReader
from src.offline.baselines import UniformVoxel3D, UniformOccupancy2D
from src.offline.pareto import (RecallAccumulator, export_results,
                                foveated_detects, foveated_obstacle_lookup,
                                reference_obstacle_cells)
from src.projection.range_projection import back_project, project
from src.segmentation.class_map import build_raw_to_class3
from src.benchmark.provenance import collect_provenance

log = logging.getLogger(__name__)

CONFIG_NAMES = ["foveated", "voxel_5cm", "occupancy_2d"]


def run_benchmark(cfg: PipelineConfig, ring_config: RingConfig,
                  ring_config_path: str, sequence: str, repo_root: str = ".",
                  inference_mode: str = "live", stride: int = 1,
                  max_frames: int | None = None,
                  out_dir: str | Path | None = None) -> dict:
    provenance = collect_provenance(repo_root, ring_config_path,
                                    dataset_split=f"sequence {sequence}",
                                    inference_mode=inference_mode)

    reader = SequenceReader(cfg.data_root, sequence)
    raw_to_3 = build_raw_to_class3()

    segmenter = None
    if inference_mode in ("live", "random"):
        from src.segmentation.inference import Segmenter
        segmenter = Segmenter(cfg.weights_path, cfg.semantic_kitti_yaml,
                              device=cfg.device, fp16=cfg.fp16,
                              random_init=(inference_mode == "random"))

    acc = RecallAccumulator(CONFIG_NAMES, len(cfg.bands_m))
    mem_bytes = {name: [] for name in CONFIG_NAMES}
    failed_frames = []

    n = len(reader) if max_frames is None else min(len(reader), max_frames)
    frame_idxs = list(range(0, n, stride))

    # Pipelined execution: disk read + projection + (GPU) inference run in a
    # producer thread while this thread does the CPU-heavy grid/baseline/recall
    # work for the PREVIOUS frame - otherwise the GPU idles during scoring and
    # the CPU idles during inference (measured 68% SM utilization sequential).
    # All accumulator state stays in this thread and frames arrive FIFO, so
    # results are bit-identical to the sequential loop.
    import queue as _queue
    import threading as _threading
    q: "_queue.Queue" = _queue.Queue(maxsize=3)
    _SENTINEL = object()

    def _producer():
        try:
            for i in frame_idxs:
                pts = reader.scan(i)
                gt_labels3 = raw_to_3[reader.labels(i)]
                proj = project(pts, cfg.projection)
                if segmenter is not None:
                    label_img = segmenter.segment(proj.range_image)
                    pred_labels3 = (back_project(label_img, proj.pixel_map)
                                    if label_img is not None else None)
                else:  # gt-labels mode: measurement independent of Module B
                    pred_labels3 = gt_labels3.astype(np.int8)
                q.put((i, pts, gt_labels3, proj, pred_labels3))
        except BaseException as e:  # surfaced in the consumer, never swallowed
            q.put(e)
        finally:
            q.put(_SENTINEL)

    _threading.Thread(target=_producer, name="benchmark-producer",
                      daemon=True).start()

    t_wall0 = time.perf_counter()
    n_processed = 0
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, BaseException):
            raise item
        i, pts, gt_labels3, proj, pred_labels3 = item
        if pred_labels3 is None:
            # NaN output: frame marked failed, empty grid, continue (T-23).
            failed_frames.append(i)
            log.warning("Frame %d failed (NaN model output); continuing.", i)
            continue

        in_range = proj.pixel_map[:, 0] >= 0
        xyz = pts[in_range, :3].astype(np.float64)
        lab = pred_labels3[in_range]

        grid = build_grid(xyz, lab, ring_config,
                          memory_ceiling_bytes=int(cfg.memory_ceiling_mb * 1e6))
        vox = UniformVoxel3D(xyz, lab, voxel=cfg.reference_cell_m)
        occ = UniformOccupancy2D(xyz, cell=cfg.reference_cell_m)

        mem_bytes["foveated"].append(grid.memory_bytes())
        mem_bytes["voxel_5cm"].append(vox.memory_bytes())
        mem_bytes["occupancy_2d"].append(occ.memory_bytes())

        gt_xyz = pts[:, :3].astype(np.float64)
        reference = reference_obstacle_cells(gt_xyz, gt_labels3,
                                             cfg.reference_cell_m, cfg.bands_m)
        fov_lookup = foveated_obstacle_lookup(grid, cfg.min_class_count)
        vox_cols = vox.obstacle_columns()
        occ_cells = occ.obstacle_cells()
        acc.add_frame(reference, {
            "foveated": lambda c: foveated_detects(c, cfg.reference_cell_m, ring_config, fov_lookup),
            "voxel_5cm": lambda c: c in vox_cols,
            "occupancy_2d": lambda c: c in occ_cells,
        })
        n_processed += 1
        if i % 100 == 0:
            log.info("benchmark: frame %d/%d", i, n)

    wall = time.perf_counter() - t_wall0
    recall = acc.recall()
    memory_mb = {name: float(np.mean(v)) / 1e6 if v else 0.0 for name, v in mem_bytes.items()}
    out = Path(out_dir) if out_dir else Path(cfg.output_dir) / f"seq{sequence}_{inference_mode}"
    export_results(out, recall, memory_mb, [int(t) for t in acc.total],
                   cfg.bands_m, provenance)

    results = {
        "recall": recall,
        "memory_mb": memory_mb,
        "reference_cells_per_band": [int(t) for t in acc.total],
        "failed_frames": failed_frames,
        # Pipelined: wall time over frames (effective throughput), NOT the
        # per-stage latency - stage-5 latency-b is the latency measurement.
        "mean_frame_time_s": (wall / n_processed) if n_processed else None,
        "out_dir": str(out),
        "provenance": provenance,
    }
    log.info("benchmark complete: %s", out)
    return results
