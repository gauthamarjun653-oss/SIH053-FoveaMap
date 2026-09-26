"""Step 4 — accuracy and cost comparison, old detector vs new.

Reports what docs/ground-visibility-definition.md §5 asks for: candidates per
frame split by kind, the visibility/ground state distribution that explains
*why* each decision was made, latency as median/p95/worst, and the range
distribution of surviving candidates.

What this does NOT do is score recall. There is no hand-labelled depression set
yet, so no missed-hazard or false-alarm number can be produced honestly. Use
scripts/extract_examples.py to build that set first. Until then this measures
cost and behaviour, not correctness, and says so.

    .venv/bin/python scripts/eval_negative_obs.py --frames 20
    .venv/bin/python scripts/eval_negative_obs.py --frames 20 --skip-old
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.cell_stats import compute_stats  # noqa: E402
from src.analytics.negative_obs import negative_obstacle_candidates  # noqa: E402
from src.analytics.negative_obs2 import detect  # noqa: E402
from src.common.config import load_rings  # noqa: E402
from src.grid.foveated_grid import build_grid  # noqa: E402
from src.io.kitti_reader import read_labels, read_scan  # noqa: E402
from src.segmentation.class_map import build_raw_to_class3  # noqa: E402

RANGE_EDGES = [2.5, 5, 7.5, 10, 15, 20, 25, 30]


def stat(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "worst": float(a.max()),
            "mean": float(a.mean())}


def range_histogram(ranges: list[float]) -> dict:
    if not ranges:
        return {}
    a = np.asarray(ranges)
    out = {}
    for lo, hi in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        out[f"{lo}-{hi}m"] = int(np.count_nonzero((a >= lo) & (a < hi)))
    out[f">={RANGE_EDGES[-1]}m"] = int(np.count_nonzero(a >= RANGE_EDGES[-1]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/semantickitti")
    ap.add_argument("--rings", default="configs/rings.yaml")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--skip-old", action="store_true",
                    help="skip the old detector (it costs ~1 s per frame)")
    ap.add_argument("--max-range-missing", type=float, default=25.0,
                    help="radius for missing-return candidates (absence evidence)")
    ap.add_argument("--max-range-below", type=float, default=30.0,
                    help="radius for below-plane candidates (measured evidence)")
    ap.add_argument("--no-require-supported", action="store_true",
                    help="open decision 5: allow EXTRAPOLATED ground to emit")
    ap.add_argument("--out", default="output/eval_negative_obs.json")
    args = ap.parse_args(argv)

    logging.disable(logging.INFO)
    rings = load_rings(args.rings)
    raw_to_3 = build_raw_to_class3()

    seq_root = Path(args.data_root) / "sequences"
    seq_dirs = sorted(d for d in seq_root.iterdir()
                      if d.is_dir() and (d / "velodyne").is_dir())

    report = {"frames_per_sequence": args.frames,
              "max_range_missing_m": args.max_range_missing,
              "max_range_below_m": args.max_range_below,
              "require_supported": not args.no_require_supported,
              "recall_scored": False,
              "candidate_units": {"old": "cells", "new": "clusters"},
              "comparison_note": "Cell and cluster counts are not comparable false-alarm rates; accuracy needs independent region annotations.",
              "labels": "ground truth; segmentation inference not measured",
              "beam_model": "approximate uniform elevations; no calibrated blind mask",
              "recall_note": "no hand-labelled depression set exists yet; "
                             "cost and behaviour only",
              "sequences": []}

    for d in seq_dirs:
        scans = sorted((d / "velodyne").glob("*.bin"))
        idxs = [int(round(i)) for i in np.linspace(0, len(scans) - 1,
                                                   min(args.frames, len(scans)))]
        new_t, old_t = [], []
        new_n, old_n = [], []
        new_mr, new_bp, new_cells = [], [], []
        samples = []
        availability = {}
        vis_acc, ground_acc = {}, {}
        ranges_mr, ranges_bp = [], []
        rejected_small, rejected_shape, inconsistent = [], [], []
        heights = []

        print(f"sequence {d.name}: {len(idxs)} frames", flush=True)
        for i in idxs:
            sp = scans[i]
            lp = d / "labels" / (sp.stem + ".label")
            if not lp.is_file():
                continue
            pts = read_scan(str(sp))
            lab = read_labels(str(lp))
            cls = raw_to_3[np.asarray(lab).astype(np.int64) & 0xFFFF].astype(np.int64)
            grid = compute_stats(build_grid(pts[:, :3].astype(np.float64), cls, rings))

            t0 = time.perf_counter()
            r = detect(grid, rings,
                       max_range_missing=args.max_range_missing,
                       max_range_below=args.max_range_below,
                       require_supported=not args.no_require_supported,
                       points_xyz=pts[:, :3])
            new_t.append((time.perf_counter() - t0) * 1000)

            availability[r.missing_return_status] = availability.get(r.missing_return_status, 0)+1
            samples.append(sp.stem)
            new_cells.append(sum(c.n_cells for c in r.candidates))
            kinds = r.counts_by_kind()
            new_n.append(len(r.candidates))
            new_mr.append(kinds["missing_return"])
            new_bp.append(kinds["below_plane"])
            for k, v in r.visibility_counts.items():
                vis_acc[k] = vis_acc.get(k, 0) + v
            for k, v in r.ground_counts.items():
                ground_acc[k] = ground_acc.get(k, 0) + v
            rejected_small.append(r.clusters_rejected_small)
            rejected_shape.append(r.clusters_rejected_shape)
            inconsistent.append(r.inconsistent_rays)
            if r.sensor_height is not None:
                heights.append(r.sensor_height)
            ranges_mr += [c.mean_range for c in r.candidates if c.kind == "missing_return"]
            ranges_bp += [c.mean_range for c in r.candidates if c.kind == "below_plane"]

            if not args.skip_old:
                t0 = time.perf_counter()
                old = negative_obstacle_candidates(grid, rings, max_range=args.max_range_missing)
                old_t.append((time.perf_counter() - t0) * 1000)
                old_n.append(sum(np.hypot(c.x,c.y) <=
                                 (args.max_range_below if c.kind == "below_plane" else args.max_range_missing)
                                 for c in old))

        entry = {
            "sequence": d.name,
            "frames": len(new_t),
            "frame_ids": samples,
            "new": {
                "latency_ms": stat(new_t),
                "emitted_cells_per_frame": stat(new_cells),
                "candidates_per_frame": stat([float(v) for v in new_n]),
                "missing_return_per_frame": stat([float(v) for v in new_mr]),
                "below_plane_per_frame": stat([float(v) for v in new_bp]),
                "visibility_states": vis_acc,
                "missing_return_status_frames": availability,
                "ground_states": ground_acc,
                "clusters_rejected_small": int(np.sum(rejected_small)),
                "clusters_rejected_shape": int(np.sum(rejected_shape)),
                "inconsistent_rays": stat([float(v) for v in inconsistent]),
                "sensor_height_m": stat(heights),
                "range_hist_missing_return": range_histogram(ranges_mr),
                "range_hist_below_plane": range_histogram(ranges_bp),
            },
        }
        if not args.skip_old:
            entry["old"] = {"latency_ms": stat(old_t),
                            "candidates_per_frame": stat([float(v) for v in old_n])}
        report["sequences"].append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for e in report["sequences"]:
        n = e["new"]
        print(f"\n--- sequence {e['sequence']} ({e['frames']} frames) ---")
        print(f"{'':22}{'median':>10}{'p95':>10}{'worst':>10}")
        print(f"{'new latency ms':<22}{n['latency_ms']['median']:10.1f}"
              f"{n['latency_ms']['p95']:10.1f}{n['latency_ms']['worst']:10.1f}")
        if "old" in e:
            o = e["old"]
            print(f"{'old latency ms':<22}{o['latency_ms']['median']:10.1f}"
                  f"{o['latency_ms']['p95']:10.1f}{o['latency_ms']['worst']:10.1f}")
            print(f"{'old cells/frame':<22}{o['candidates_per_frame']['median']:10.0f}")
        print(f"{'new clusters/frame':<22}{n['candidates_per_frame']['median']:10.0f}"
              f"{n['candidates_per_frame']['p95']:10.0f}{n['candidates_per_frame']['worst']:10.0f}")
        print(f"  missing_return {n['missing_return_per_frame']['median']:.0f}/frame, "
              f"below_plane {n['below_plane_per_frame']['median']:.0f}/frame")
        print(f"  sensor height  {n['sensor_height_m'].get('median', float('nan')):.3f} m (estimated)")
        print(f"  visibility     {n['visibility_states']}")
        print(f"  ground         {n['ground_states']}")
        print(f"  range hist (missing_return) {n['range_hist_missing_return']}")
        print(f"  range hist (below_plane)    {n['range_hist_below_plane']}")
    print(f"\nrecall NOT scored — {report['recall_note']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
