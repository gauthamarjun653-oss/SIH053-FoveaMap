"""Step 5 measurement — cascade/float64 vs direct/compact.

Reports the four things the review asked to keep separate:

  1. stored map bytes (what the Pareto curve is about)
  2. peak process RSS (working accumulators, which smaller records do not shrink)
  3. build latency, median / p95 / worst
  4. per-field agreement: counts exact, heights bounded

Accumulation is float64 in both paths; only storage narrows. If the compact
path shrank peak RSS as well that would be a bonus, not the claim.

    .venv/bin/python scripts/compare_grid_layouts.py --frames 20
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.cell_stats import compute_stats  # noqa: E402
from src.common.config import load_rings  # noqa: E402
from src.grid.compact import (build_compact_grid, build_grid_direct, compare_blocks,  # noqa: E402
                              pack_block)
from src.grid.foveated_grid import build_grid  # noqa: E402
from src.io.kitti_reader import read_labels, read_scan  # noqa: E402
from src.segmentation.class_map import build_raw_to_class3  # noqa: E402

PAGE = resource.getpagesize()


def rss() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * PAGE
    except (OSError, IndexError, ValueError):
        return 0


def peak_rss() -> int:
    # Linux VmHWM belongs to this exec's memory image. getrusage can retain
    # a pre-exec high-water mark inherited from the parent comparison process.
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])*1024
    except (OSError, ValueError, IndexError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def stat(v):
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "worst": float(a.max())}


def layout_worker(args):
    """One variant per fresh interpreter; no other map is retained."""
    rings = load_rings(args.rings)
    lut = build_raw_to_class3()
    baseline = rss()
    times, maps, samples = [], [], []
    for directory in sorted((Path(args.data_root)/"sequences").iterdir()):
        scans = sorted((directory/"velodyne").glob("*.bin"))
        if not scans:
            continue
        for i in np.linspace(0, len(scans)-1, min(args.frames, len(scans))).round().astype(int):
            scan = scans[i]
            label = directory/"labels"/(scan.stem+".label")
            if not label.is_file():
                continue
            xyz = read_scan(str(scan))[:, :3].astype(np.float64)
            cls = lut[read_labels(str(label)).astype(np.int64)&65535]
            builder = (lambda: build_compact_grid(xyz,cls,rings)) if args.worker_layout == "compact" else (lambda: compute_stats(build_grid(xyz,cls,rings)))
            if not times:
                warmup = builder()
                del warmup
            started = time.perf_counter()
            grid = builder()
            times.append((time.perf_counter()-started)*1000)
            maps.append(grid.memory_bytes()/1e6)
            samples.append([directory.name, scan.stem])
            del grid, xyz, cls
    if not times:
        raise ValueError("No labelled frames found")
    result = {"layout": args.worker_layout, "samples": samples,
              "build_ms": stat(times), "stored_map_mb": stat(maps),
              "baseline_process_rss_mb": baseline/1e6,
              "peak_process_rss_mb": peak_rss()/1e6,
              "peak_rss_source": "Linux VmHWM" if Path("/proc/self/status").exists() else "getrusage fallback",
              "scope": "load + build + stats + optional pack; no detector/inference/viewer"}
    Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


def isolated_comparison(args):
    results = {}
    with tempfile.TemporaryDirectory(prefix="foveamap-rss-") as temp:
        for layout in ("cascade", "compact"):
            out = str(Path(temp)/(layout+".json"))
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--worker-layout", layout, "--frames", str(args.frames),
                            "--data-root", str(Path(args.data_root).resolve()),
                            "--rings", str(Path(args.rings).resolve()), "--out", out],
                           check=True, capture_output=True, text=True)
            results[layout] = json.loads(Path(out).read_text())
    if results["cascade"]["samples"] != results["compact"]["samples"]:
        raise RuntimeError("Memory workers evaluated different frames")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/semantickitti")
    ap.add_argument("--rings", default="configs/rings.yaml")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--mean-tol", type=float, default=1e-3,
                    help="metres; max acceptable per-cell mean height error")
    ap.add_argument("--var-tol", type=float, default=1e-4,
                    help="m^2; max acceptable per-cell height variance error")
    ap.add_argument("--out", default="output/grid_layout_comparison.json")
    ap.add_argument("--worker-layout", choices=["cascade", "compact"], help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.frames < 1:
        ap.error("--frames must be positive")
    if args.worker_layout:
        return layout_worker(args)

    rings = load_rings(args.rings)
    raw_to_3 = build_raw_to_class3()
    seq_root = Path(args.data_root) / "sequences"
    seqs = sorted(d for d in seq_root.iterdir() if (d / "velodyne").is_dir())

    t_cas, t_dir = [], []
    mb_wide, mb_compact = [], []
    rss_cas, rss_dir = [], []
    mean_err, var_err, rel_var_err = [], [], []
    counts_ok = hist_ok = zmin_ok = zmax_ok = cells_ok = True
    n_frames = 0

    for d in seqs:
        scans = sorted((d / "velodyne").glob("*.bin"))
        idxs = [int(round(i)) for i in np.linspace(0, len(scans) - 1,
                                                   min(args.frames, len(scans)))]
        print(f"sequence {d.name}: {len(idxs)} frames", flush=True)
        for i in idxs:
            sp = scans[i]
            lp = d / "labels" / (sp.stem + ".label")
            if not lp.is_file():
                continue
            pts = read_scan(str(sp))
            lab = read_labels(str(lp))
            cls = raw_to_3[np.asarray(lab).astype(np.int64) & 0xFFFF].astype(np.int64)
            xyz = pts[:, :3].astype(np.float64)
            n_frames += 1

            r0 = rss()
            t = time.perf_counter()
            g_cas = compute_stats(build_grid(xyz, cls, rings))
            t_cas.append((time.perf_counter() - t) * 1000)
            rss_cas.append((rss() - r0) / 1e6)
            mb_wide.append(g_cas.memory_bytes() / 1e6)

            r0 = rss()
            t = time.perf_counter()
            g_dir = build_compact_grid(xyz, cls, rings)
            packed = g_dir.blocks
            t_dir.append((time.perf_counter() - t) * 1000)
            rss_dir.append((rss() - r0) / 1e6)
            mb_compact.append(sum(b.memory_bytes() for b in packed.values()) / 1e6)

            cells_ok &= set(g_cas.blocks) == set(packed)
            for rid, b_cas in g_cas.blocks.items():
                b_pk = packed.get(rid)
                if b_pk is None:
                    cells_ok = False
                    continue
                c = compare_blocks(b_cas, b_pk)
                cells_ok &= c.same_cells
                counts_ok &= c.count_exact
                hist_ok &= c.hist_exact
                zmin_ok &= c.z_min_exact
                zmax_ok &= c.z_max_exact
                if c.same_cells:
                    mean_err.append(c.max_abs_mean_err)
                    var_err.append(c.max_abs_var_err)
                    rel_var_err.append(c.max_rel_var_err)

    if not n_frames:
        raise ValueError("No labelled frames found; comparison cannot pass")
    isolated = isolated_comparison(args)
    wide, comp = float(np.median(mb_wide)), float(np.median(mb_compact))
    report = {
        "frames": n_frames,
        "stored_map_mb_cascade_f64": stat(mb_wide),
        "stored_map_mb_direct_compact": stat(mb_compact),
        "reduction_pct_median": 100 * (1 - comp / wide) if wide else 0.0,
        "bytes_per_cell": {"cascade_f64": 84, "direct_compact": 56},
        "build_ms_cascade": stat(t_cas),
        "build_ms_direct_compact": stat(t_dir),
        "rss_delta_mb_cascade": stat(rss_cas),
        "rss_delta_mb_direct_compact": stat(rss_dir),
        "agreement_process_peak_rss_mb": peak_rss() / 1e6,
        "isolated_variants": isolated,
        "shared_process_timings_note": "Use isolated_variants for performance comparisons",
        "exact_fields": {"same_cells": cells_ok, "count": counts_ok,
                         "hist": hist_ok, "z_min": zmin_ok, "z_max": zmax_ok},
        "bounded_fields": {"max_abs_mean_err_m": stat(mean_err),
                           "max_abs_var_err_m2": stat(var_err),
                           "max_rel_var_err": stat(rel_var_err)},
        "tolerances": {"mean_m": args.mean_tol, "var_m2": args.var_tol},
    }
    worst_mean = max(mean_err) if mean_err else 0.0
    worst_var = max(var_err) if var_err else 0.0
    report["within_tolerance"] = bool(
        cells_ok and counts_ok and hist_ok and zmin_ok and zmax_ok
        and worst_mean <= args.mean_tol and worst_var <= args.var_tol)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n--- {n_frames} frames ---")
    print(f"{'':30}{'median':>10}{'p95':>10}{'worst':>10}")
    print(f"{'stored MB cascade/f64':<30}{stat(mb_wide)['median']:10.2f}"
          f"{stat(mb_wide)['p95']:10.2f}{stat(mb_wide)['worst']:10.2f}")
    print(f"{'stored MB direct/compact':<30}{stat(mb_compact)['median']:10.2f}"
          f"{stat(mb_compact)['p95']:10.2f}{stat(mb_compact)['worst']:10.2f}")
    print(f"{'build ms cascade':<30}{stat(t_cas)['median']:10.1f}"
          f"{stat(t_cas)['p95']:10.1f}{stat(t_cas)['worst']:10.1f}")
    print(f"{'build ms direct+pack':<30}{stat(t_dir)['median']:10.1f}"
          f"{stat(t_dir)['p95']:10.1f}{stat(t_dir)['worst']:10.1f}")
    print(f"\nstored reduction: {report['reduction_pct_median']:.2f}% "
          f"(84 -> 56 bytes/cell)")
    for name, measured in isolated.items():
        print(f"isolated {name}: peak RSS {measured['peak_process_rss_mb']:.1f} MB; "
              f"median build {measured['build_ms']['median']:.1f} ms")
    print(f"\nexact fields:   {report['exact_fields']}")
    print(f"mean error  max {worst_mean:.3e} m   (tolerance {args.mean_tol:g})")
    print(f"var  error  max {worst_var:.3e} m^2 (tolerance {args.var_tol:g})")
    print(f"WITHIN TOLERANCE: {report['within_tolerance']}")
    print(f"\nwrote {out}")
    return 0 if report["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
