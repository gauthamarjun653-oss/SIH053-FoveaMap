"""Build the hand-check example set required by the ground/visibility definition §5.

Recall cannot be scored without labelled depressions, and labelling them is a
human job: the ground truth for this set is inspection of geometry, not model
output, because the segmentation model saw part of sequences 00-10 during
training. This script does the part a machine can do — find and crop the
candidates and the negative classes, dump a reviewable profile of each, and
freeze the selection under a seed so the set cannot drift once thresholds start
moving.

For each example it writes a local height cross-section: the ground plane
prediction against the actual returns within a small radius. A depression shows
as returns sitting below the plane, or as a run with no returns at all behind a
near lip. Flat road shows returns tracking the plane. That is enough to
adjudicate most cases without a 3D viewer, and matplotlib is not installed here.

    .venv/bin/python scripts/extract_examples.py --per-class 8
    # then fill in "verdict" in output/examples/manifest.json by hand

Verdicts stay empty until a person sets them. Nothing in this repository should
write that field.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.cell_stats import compute_stats  # noqa: E402
from src.analytics.ground import (EXTRAPOLATED, SUPPORTED,  # noqa: E402
                                  build_ground_field)
from src.analytics.negative_obs import _OcclusionIndex  # noqa: E402
from src.analytics.negative_obs2 import detect  # noqa: E402
from src.analytics.visibility import (BLIND, EXPECTED_EMPTY,  # noqa: E402
                                      NO_GROUND, OCCLUDED, STATE_NAMES,
                                      compute_visibility)
from src.common.config import load_rings  # noqa: E402
from src.grid.foveated_grid import build_grid  # noqa: E402
from src.io.kitti_reader import read_labels, read_scan  # noqa: E402
from src.segmentation.class_map import build_raw_to_class3  # noqa: E402

# The six classes the definition names, plus the positives from the detector.
NEGATIVE_CLASSES = {
    "unreviewed_expected_empty": EXPECTED_EMPTY,
    "negative_occluded": OCCLUDED,
    "negative_blind_zone": BLIND,
    "negative_no_ground": NO_GROUND,
}


def cross_section(pts: np.ndarray, cx: float, cy: float, plane, radius: float = 1.5,
                  n_bins: int = 12) -> dict:
    """Radial height profile around a point: plane prediction vs actual returns."""
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    m = d <= radius
    r_hat = np.hypot(cx, cy)
    out = {"n_returns_within_radius": int(m.sum()), "radius_m": radius,
           "bins": []}
    if plane is not None:
        a, b, c = plane
        out["plane_z_at_centre"] = float(a * cx + b * cy + c)
    if not np.any(m):
        return out
    px, py, pz = pts[m, 0], pts[m, 1], pts[m, 2]
    # Bin along the sensor-radial direction, the axis a depression shadows.
    pr = np.hypot(px, py)
    edges = np.linspace(r_hat - radius, r_hat + radius, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (pr >= lo) & (pr < hi)
        entry = {"r_lo": round(float(lo), 3), "r_hi": round(float(hi), 3),
                 "n": int(k.sum())}
        if np.any(k):
            entry["z_min"] = round(float(pz[k].min()), 4)
            entry["z_mean"] = round(float(pz[k].mean()), 4)
            if plane is not None:
                a, b, c = plane
                entry["z_plane"] = round(float(np.mean(a * px[k] + b * py[k] + c)), 4)
                entry["below_plane_m"] = round(entry["z_plane"] - entry["z_min"], 4)
        out["bins"].append(entry)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/semantickitti")
    ap.add_argument("--rings", default="configs/rings.yaml")
    ap.add_argument("--frames", type=int, default=8, help="frames per sequence")
    ap.add_argument("--per-class", type=int, default=8,
                    help="examples sampled per class per sequence")
    ap.add_argument("--seed", type=int, default=20260922,
                    help="frozen so the set does not drift as thresholds move")
    ap.add_argument("--out-dir", default="output/examples")
    args = ap.parse_args(argv)
    output_path = Path(args.out_dir)/"manifest.json"
    if output_path.exists():
        ap.error(f"{output_path} already exists; choose a new --out-dir to preserve annotations")

    logging.disable(logging.INFO)
    rng = np.random.default_rng(args.seed)
    rings = load_rings(args.rings)
    raw_to_3 = build_raw_to_class3()
    seq_root = Path(args.data_root) / "sequences"
    seqs = sorted(d for d in seq_root.iterdir() if (d / "velodyne").is_dir())

    examples = []
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
            grid = compute_stats(build_grid(xyz, cls, rings))
            ground = build_ground_field(grid)
            def local_plane(x, y):
                coefficients, hit = ground.plane_coeffs_at(np.array([x]), np.array([y]))
                return tuple(coefficients[0]) if hit[0] else None

            # Positives: what the detector emitted, highest confidence first.
            res = detect(grid, rings, points_xyz=xyz)
            for kind in ("below_plane", "missing_return"):
                cands = sorted([c for c in res.candidates if c.kind == kind],
                               key=lambda c: -c.confidence)[:args.per_class]
                for c in cands:
                    examples.append({
                        "sequence": d.name, "frame": sp.stem,
                        "klass": f"detector_{kind}",
                        "x": round(c.centroid[0], 3), "y": round(c.centroid[1], 3),
                        "range_m": round(c.mean_range, 2),
                        "confidence": round(c.confidence, 3),
                        "n_cells": c.n_cells,
                        "extent_m": [round(c.size_m[0], 2), round(c.size_m[1], 2)],
                        "shadow_predicted_m": round(c.shadow_predicted_m, 3),
                        "profile": cross_section(xyz, c.centroid[0], c.centroid[1], local_plane(*c.centroid)),
                        "verdict": "", "notes": "",
                    })

            # Negatives: sites in each visibility state that produced no candidate.
            vis = compute_visibility(grid, ground, rings, _OcclusionIndex(grid), points_xyz=xyz)
            if vis.x.size:
                for name, state in NEGATIVE_CLASSES.items():
                    pool = np.flatnonzero(vis.state == state)
                    if pool.size == 0:
                        continue
                    pick = rng.choice(pool, size=min(args.per_class, pool.size),
                                      replace=False)
                    for j in pick.tolist():
                        examples.append({
                            "sequence": d.name, "frame": sp.stem,
                            "klass": name,
                            "x": round(float(vis.x[j]), 3),
                            "y": round(float(vis.y[j]), 3),
                            "range_m": round(float(np.hypot(vis.x[j], vis.y[j])), 2),
                            "visibility_state": STATE_NAMES[state],
                            "expected_returns": round(float(vis.expected_returns[j]), 2),
                            "profile": cross_section(xyz, float(vis.x[j]),
                                                     float(vis.y[j]), local_plane(float(vis.x[j]), float(vis.y[j]))),
                            "verdict": "", "notes": "",
                        })

            # Ring-boundary negatives: flat ground within 0.5 m of a transition.
            for rg in rings.rings[:-1]:
                rb = rg.radius_max
                if vis.x.size == 0:
                    break
                rr = np.hypot(vis.x, vis.y)
                near = np.flatnonzero((np.abs(rr - rb) < 0.5) & (vis.state == EXPECTED_EMPTY))
                if near.size == 0:
                    continue
                for j in rng.choice(near, size=min(2, near.size), replace=False).tolist():
                    examples.append({
                        "sequence": d.name, "frame": sp.stem,
                        "klass": "unreviewed_ring_boundary",
                        "x": round(float(vis.x[j]), 3), "y": round(float(vis.y[j]), 3),
                        "range_m": round(float(rr[j]), 2),
                        "boundary_radius_m": rb,
                        "profile": cross_section(xyz, float(vis.x[j]), float(vis.y[j]), local_plane(float(vis.x[j]), float(vis.y[j]))),
                        "verdict": "", "notes": "",
                    })

            # Extrapolated-ground negatives: occupied road cells whose tile plane
            # was fitted from cells too far away to support them.
            for block in grid.blocks.values():
                if len(block) == 0:
                    continue
                bx = (block.cx + 0.5) * block.cell_size
                by = (block.cy + 0.5) * block.cell_size
                _, gstate = ground.height_and_state(bx, by)
                pool = np.flatnonzero(gstate == EXTRAPOLATED)
                if pool.size == 0:
                    continue
                for j in rng.choice(pool, size=min(2, pool.size), replace=False).tolist():
                    examples.append({
                        "sequence": d.name, "frame": sp.stem,
                        "klass": "negative_extrapolated_ground",
                        "x": round(float(bx[j]), 3), "y": round(float(by[j]), 3),
                        "range_m": round(float(np.hypot(bx[j], by[j])), 2),
                        "profile": cross_section(xyz, float(bx[j]), float(by[j]), local_plane(float(bx[j]), float(by[j]))),
                        "verdict": "", "notes": "",
                    })
                break  # one ring per frame is enough for this class

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "frozen": True,
        "purpose": "diagnostic inspection; not an independent recall evaluation set",
        "recall_eligible": False,
        "sampling_bias": "Detector outputs and visibility states select examples. Annotate complete independent regions to measure missed hazards.",
        "ground_truth_policy": "manual inspection of geometry; NOT model output. "
                               "The segmentation model saw part of sequences 00-10 "
                               "in training, so its labels cannot serve as truth here.",
        "verdict_values": ["", "true_positive", "false_positive",
                           "true_negative", "false_negative", "unclear"],
        "instructions": "Set 'verdict' for each example by inspecting 'profile'. "
                        "A depression shows returns below z_plane, or a run of "
                        "empty bins behind a near lip. Flat road tracks the plane. "
                        "Leave 'unclear' where the geometry does not settle it — "
                        "do not guess.",
        "n_examples": len(examples),
        "by_class": {},
        "examples": examples,
    }
    for e in examples:
        manifest["by_class"][e["klass"]] = manifest["by_class"].get(e["klass"], 0) + 1

    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(examples)} examples")
    for k, v in sorted(manifest["by_class"].items()):
        print(f"  {k:32s} {v:5d}")
    print(f"\nwrote {path}")
    print("verdicts are empty; this diagnostic sample cannot establish recall. "
          "Use independently annotated complete regions for accuracy evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
