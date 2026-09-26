"""Step 1 - reproducible per-stage profiling.

Measures each pipeline stage separately across frames sampled from the full
span of every available sequence, and reports median / p95 / worst rather than
a mean. A mean hides exactly the tail this project needs to defend: one scene
in twenty costing 300 ms is a dropped scan, and an average will not show it.

Memory is reported twice and the two are never added together:

  * stored map bytes - the finished grid's payload arrays (what the Pareto
    curve is about), measured after compute_stats so derived layers exist.
  * process RSS high-water mark - the largest resident footprint reached,
    read from /proc. This covers working accumulators and temporaries that
    array accounting cannot see. It is process-wide and monotonic, so it is a
    ceiling for the run, not a per-frame figure; the per-frame RSS delta is
    reported alongside it and is the noisier of the two.

Stages needing torch (inference) are skipped and reported as absent rather
than silently omitted. Run:

    .venv/bin/python scripts/profile_stages.py --frames 40
    .venv/bin/python scripts/profile_stages.py --frames 40 --with-negative-obs
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.cell_stats import compute_stats  # noqa: E402
from src.common.config import RingConfig, RingSpec, load_rings  # noqa: E402
from src.grid.foveated_grid import build_grid  # noqa: E402
from src.io.kitti_reader import read_labels, read_scan  # noqa: E402
from src.projection.range_projection import project  # noqa: E402
from src.segmentation.class_map import build_raw_to_class3  # noqa: E402

PAGE_SIZE = resource.getpagesize()


def rss_bytes() -> int:
    """Resident set size from /proc/self/statm (Linux). 0 where unavailable."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            return int(f.read().split()[1]) * PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return 0


def peak_rss_bytes() -> int:
    """Process high-water mark. ru_maxrss is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def summarise(samples: list[float]) -> dict:
    """Median / p95 / worst. Averages are recorded but never led with."""
    if not samples:
        return {"n": 0}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "worst": float(a.max()),
        "best": float(a.min()),
        "mean": float(a.mean()),
    }


def uniform_config(cell_size: float, base: float, r_min: float, r_max: float) -> RingConfig:
    """Single-ring equivalent of the foveated config, same radial span."""
    return RingConfig(
        base_cell_size=base,
        rings=(RingSpec(id=1, radius_min=r_min, radius_max=r_max, cell_size=cell_size),),
    )


def sample_frames(n_frames: int, n_wanted: int) -> list[int]:
    """Evenly spread indices across the whole sequence, not a leading block:
    scene content varies along a drive and the first N frames are one scene."""
    if n_wanted >= n_frames:
        return list(range(n_frames))
    return [int(round(i)) for i in np.linspace(0, n_frames - 1, n_wanted)]


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def profile_sequence(seq_dir: Path, seq: str, rings: RingConfig, raw_to_3: np.ndarray,
                     n_wanted: int, with_neg: bool, proj_cfg) -> dict:
    velodyne = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    scans = sorted(velodyne.glob("*.bin"))
    idxs = sample_frames(len(scans), n_wanted)

    uni = uniform_config(rings.base_cell_size, rings.base_cell_size,
                         rings.rings[0].radius_min, rings.rings[-1].radius_max)

    stages: dict[str, list[float]] = {
        "read": [], "project": [], "build_grid": [], "compute_stats": [],
        "build_uniform": [],
    }
    if with_neg:
        stages["negative_obs"] = []

    map_mb: list[float] = []
    uniform_mb: list[float] = []
    cells: list[int] = []
    rss_delta_mb: list[float] = []

    neg_fn = None
    if with_neg:
        from src.analytics.negative_obs import negative_obstacle_candidates
        neg_fn = negative_obstacle_candidates

    for i in idxs:
        scan_path = scans[i]
        label_path = labels_dir / (scan_path.stem + ".label")
        if not label_path.is_file():
            continue

        rss_before = rss_bytes()

        t = time.perf_counter()
        pts = read_scan(str(scan_path))
        lab = read_labels(str(label_path))
        stages["read"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        proj = project(pts, proj_cfg)
        stages["project"].append((time.perf_counter() - t) * 1000)

        # Ground-truth labels. Predictions would come from Module B here; that
        # stage needs torch and is reported separately as absent.
        cls3 = raw_to_3[np.asarray(lab).astype(np.int64) & 0xFFFF]
        in_range = proj.pixel_map[:, 0] >= 0
        xyz = pts[in_range, :3].astype(np.float64)
        lab3 = cls3[in_range].astype(np.int64)

        t = time.perf_counter()
        grid = build_grid(xyz, lab3, rings)
        stages["build_grid"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        compute_stats(grid)
        stages["compute_stats"].append((time.perf_counter() - t) * 1000)

        blocks = grid.blocks
        map_mb.append(sum(b.memory_bytes() for b in blocks.values()) / 1e6)
        cells.append(sum(len(b) for b in blocks.values()))

        t = time.perf_counter()
        ugrid = build_grid(xyz, lab3, uni)
        compute_stats(ugrid)
        stages["build_uniform"].append((time.perf_counter() - t) * 1000)
        uniform_mb.append(sum(b.memory_bytes() for b in ugrid.blocks.values()) / 1e6)

        if neg_fn is not None:
            t = time.perf_counter()
            neg_fn(grid, rings)
            stages["negative_obs"].append((time.perf_counter() - t) * 1000)

        rss_delta_mb.append((rss_bytes() - rss_before) / 1e6)

    return {
        "sequence": seq,
        "frames_available": len(scans),
        "frames_profiled": len(stages["read"]),
        "frame_indices": idxs,
        "stages_ms": {k: summarise(v) for k, v in stages.items()},
        "stored_map_mb": summarise(map_mb),
        "uniform_map_mb": summarise(uniform_mb),
        "occupied_cells": summarise([float(c) for c in cells]),
        "rss_delta_per_frame_mb": summarise(rss_delta_mb),
    }


def fmt_ms(s: dict) -> str:
    if not s.get("n"):
        return "-"
    return f"{s['median']:8.1f} {s['p95']:8.1f} {s['worst']:8.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/semantickitti",
                    help="SemanticKITTI root containing sequences/")
    ap.add_argument("--rings", default="configs/rings.yaml")
    ap.add_argument("--pipeline", default="configs/pipeline.yaml")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames per sequence, spread across the whole drive")
    ap.add_argument("--with-negative-obs", action="store_true",
                    help="include negative-obstacle analysis (slow; viewer path only)")
    ap.add_argument("--out", default="output/profile_stages.json")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    rings = load_rings(args.rings)
    raw_to_3 = build_raw_to_class3()

    from src.common.config import load_pipeline
    proj_cfg = load_pipeline(args.pipeline).projection

    torch_available = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_available = False

    seq_root = Path(args.data_root) / "sequences"
    seq_dirs = sorted(d for d in seq_root.iterdir()
                      if d.is_dir() and (d / "velodyne").is_dir() and (d / "labels").is_dir())
    if not seq_dirs:
        print(f"No labelled sequences under {seq_root}", file=sys.stderr)
        return 1

    results = []
    for d in seq_dirs:
        print(f"profiling sequence {d.name} ...", flush=True)
        results.append(profile_sequence(d, d.name, rings, raw_to_3,
                                        args.frames, args.with_negative_obs, proj_cfg))

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(repo_root),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "labels": "ground truth (Module B not exercised)",
        "torch_available": torch_available,
        "stages_not_measured": ([] if torch_available else
                                ["segmentation_inference (torch not installed)"]),
        "negative_obs_included": args.with_negative_obs,
        "rings_config": args.rings,
        "peak_process_rss_mb": peak_rss_bytes() / 1e6,
        "sequences": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    for r in results:
        print(f"--- sequence {r['sequence']}  "
              f"({r['frames_profiled']}/{r['frames_available']} frames) ---")
        print(f"{'stage':<16}{'median':>9}{'p95':>9}{'worst':>9}   (ms)")
        for name, s in r["stages_ms"].items():
            print(f"{name:<16}{fmt_ms(s)}")
        m, u = r["stored_map_mb"], r["uniform_map_mb"]
        print(f"{'stored map MB':<16}{m['median']:9.2f}{m['p95']:9.2f}{m['worst']:9.2f}")
        print(f"{'uniform map MB':<16}{u['median']:9.2f}{u['p95']:9.2f}{u['worst']:9.2f}")
        if u["median"]:
            print(f"{'reduction %':<16}{100 * (1 - m['median'] / u['median']):9.2f}"
                  "   (median vs median)")
        print()

    print(f"peak process RSS over whole run: {report['peak_process_rss_mb']:.1f} MB "
          "(ceiling for the run, not per frame)")
    if not torch_available:
        print("NOT MEASURED: segmentation inference - torch not installed")
    if not args.with_negative_obs:
        print("NOT MEASURED: negative-obstacle analysis - pass --with-negative-obs")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
