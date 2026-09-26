"""SIH26053 command-line entry point.

The three checks the spec orders FIRST (section 0):
  1. python -m src.main build-map --seq 00          (asserts z-extent 10-40 m)
  2. python -m src.main precheck-negatives --seq 00 (counts instances)
  3. python -m src.main latency-b                   (Module B alone, demo GPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.common.config import load_pipeline, load_rings

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RINGS = REPO_ROOT / "configs" / "rings.yaml"
DEFAULT_PIPELINE = REPO_ROOT / "configs" / "pipeline.yaml"

log = logging.getLogger("sih26053")


def _load(args):
    ring_config = load_rings(args.rings)
    cfg = load_pipeline(args.pipeline)
    if getattr(args, "data_root", None):
        cfg.data_root = str(Path(args.data_root).expanduser())
    return cfg, ring_config


def cmd_validate_config(args) -> int:
    cfg, ring_config = _load(args)
    print(f"Ring table OK: {[(r.id, r.cell_size) for r in ring_config.rings]}")
    print(f"Pipeline OK: data_root={cfg.data_root}, "
          f"projection {cfg.projection.height}x{cfg.projection.width}")
    return 0


def cmd_build_map(args) -> int:
    from src.io.kitti_reader import SequenceReader
    from src.offline.build_map import build_global_map, assert_z_extent, wall_thickness_stats

    cfg, _ = _load(args)
    reader = SequenceReader(cfg.data_root, args.seq)
    print(f"Building global map for sequence {args.seq} ({len(reader)} frames)...")
    pts = build_global_map(reader, voxel=cfg.map_voxel_m,
                           stride=args.stride, max_frames=args.max_frames)
    extent = assert_z_extent(pts, cfg.z_extent_min_m, cfg.z_extent_max_m)
    print(f"AC-F1 PASS: z-extent {extent:.1f} m (bounds {cfg.z_extent_min_m}-{cfg.z_extent_max_m} m)")
    thickness = wall_thickness_stats(pts)
    status = "PASS" if thickness < 0.15 else "FAIL"
    print(f"AC-F2 {status}: median facade thickness {thickness * 100:.1f} cm (< 15 cm required)")
    out = Path(args.out or f"runs/map_seq{args.seq}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pts.astype(np.float32))
    print(f"Map ({pts.shape[0]} deduplicated points) saved to {out}")
    return 0 if thickness < 0.15 else 1


def cmd_precheck_negatives(args) -> int:
    from src.analytics.negative_obs import derive_ground_truth_depressions

    cfg, _ = _load(args)
    map_path = Path(args.map or f"runs/map_seq{args.seq}.npy")
    if not map_path.is_file():
        print(f"Global map '{map_path}' not found. Run build-map first "
              "(the pre-check needs the validated map).", file=sys.stderr)
        return 2
    pts = np.load(map_path)
    instances = derive_ground_truth_depressions(
        pts, depth_threshold=cfg.neg_depth_m, max_extent_m=cfg.neg_max_extent_m)
    n = len(instances)
    print(f"Negative-obstacle pre-check, sequence {args.seq}: {n} instances "
          f"(depth >= {cfg.neg_depth_m} m, extent <= {cfg.neg_max_extent_m} m)")
    if n >= 30:
        print("Decision (spec 11.3): >= 30 — proceed as specified; AC-D6 is a measured result.")
    else:
        reframed = derive_ground_truth_depressions(
            pts, depth_threshold=cfg.neg_reframed_depth_m, max_extent_m=None)
        print(f"Decision (spec 11.3): < 30. Reframed count (depth >= "
              f"{cfg.neg_reframed_depth_m} m, no extent cap): {len(reframed)} "
              "road-edge elevation discontinuities (NOT potholes — naming discipline).")
        if len(reframed) < 30:
            print("~0 instances: present the capability as designed and unit-tested "
                  "against synthetic geometry, not claimed as measured (PAC-8).")
        instances = reframed
    out = Path(f"runs/neg_precheck_seq{args.seq}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([vars(i) for i in instances], indent=2), encoding="utf-8")
    print(f"Instances written to {out}. Spot-check >= 20 manually before scoring anything.")
    return 0


def cmd_latency_b(args) -> int:
    """Measure Module B alone on this machine (spec section 0, check 3)."""
    from src.segmentation.inference import Segmenter

    cfg, _ = _load(args)
    seg = Segmenter(cfg.weights_path, cfg.semantic_kitti_yaml, device=cfg.device,
                    fp16=cfg.fp16, random_init=args.random)
    H, W = cfg.projection.height, cfg.projection.width
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (H, W, 5)).astype(np.float32)
    for _ in range(3):
        seg.segment(img)  # warm-up
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        seg.segment(img)
        times.append((time.perf_counter() - t0) * 1000)
    mode = "RANDOM WEIGHTS" if args.random else "pretrained"
    dev = f"{seg.device}{' (CPU FALLBACK — not representative)' if seg.cpu_fallback else ''}"
    print(f"Module B latency ({mode}, {dev}, {H}x{W}, fp16={seg.fp16}): "
          f"median {np.median(times):.1f} ms, p90 {np.percentile(times, 90):.1f} ms "
          f"over {args.iters} runs. Budget: 40 ms.")
    if np.median(times) > 40 and not seg.cpu_fallback:
        print("Over budget. In order: FP16 inference, width 1024, then pruning.")
    return 0


def cmd_run(args) -> int:
    """PAC-1: process a complete sequence without manual intervention."""
    if args.implementation == 'core':
        from scripts.run_core import main as run_core
        cfg, _ = _load(args)
        run_core(['--sequence',args.seq,'--dataset',cfg.data_root,
                  '--mode',{'live':'model','random':'random','gt-labels':'gt-labels'}[args.mode],
                  '--frames',str(args.max_frames if args.max_frames is not None else 1000000),
                  '--stride',str(args.stride),'--rings',str(args.rings),'--pipeline',str(args.pipeline),
                  '--out',args.report or f'output/core-{args.seq}.json'])
        return 0
    from src.io.kitti_reader import SequenceReader
    from src.projection.range_projection import project, back_project
    from src.frame_pipeline import process_frame
    from src.segmentation.class_map import build_raw_to_class3
    from src.viz.telemetry import Telemetry

    cfg, ring_config = _load(args)
    reader = SequenceReader(cfg.data_root, args.seq)
    segmenter = None
    if args.mode in ("live", "random"):
        from src.segmentation.inference import Segmenter
        segmenter = Segmenter(cfg.weights_path, cfg.semantic_kitti_yaml,
                              device=cfg.device, fp16=cfg.fp16,
                              random_init=(args.mode == "random"))
    raw_to_3 = build_raw_to_class3()
    tel = Telemetry()
    tel.cpu_fallback = bool(segmenter and segmenter.cpu_fallback)
    n = len(reader) if args.max_frames is None else min(len(reader), args.max_frames)
    timings, build_times, detection_times = [], [], []
    for i in range(0, n, args.stride):
        started = time.perf_counter()
        pts = reader.scan(i)
        proj = project(pts, cfg.projection)
        if segmenter is not None:
            label_img = segmenter.segment(proj.range_image)
            if label_img is None:
                tel.failed_frames.append(i)
                continue
            labels3 = back_project(label_img, proj.pixel_map)
        else:
            labels3 = raw_to_3[reader.labels(i)].astype(np.int8)
        in_range = proj.pixel_map[:, 0] >= 0
        result = process_frame(pts[in_range, :3].astype(np.float64), labels3[in_range],
                               ring_config, implementation=args.implementation,
                               image_width=cfg.projection.width,
                               memory_ceiling_bytes=int(cfg.memory_ceiling_mb*1e6),
                               depth_threshold=cfg.neg_depth_m, max_extent_m=cfg.neg_max_extent_m)
        grid = result.grid
        timings.append((time.perf_counter()-started)*1000)
        build_times.append(result.build_ms)
        detection_times.append(result.detection_ms)
        tel.frame(grid.memory_bytes())
        if i % 50 == 0:
            for banner in tel.banners():
                print(f"*** {banner} ***")
            print(f"frame {i}/{n}: {grid.n_cells} cells, {tel.status_line()}, "
                  f"discarded {proj.n_discarded}, collisions {proj.n_collisions}")
    if timings:
        summary = {"implementation": args.implementation, "label_mode": args.mode,
                   "frames": len(timings), "rendering_included": False,
                   "processing_ms_median": float(np.median(timings)),
                   "processing_ms_p95": float(np.percentile(timings, 95)),
                   "build_ms_median": float(np.median(build_times)),
                   "detection_ms_median": float(np.median(detection_times)),
                   "last_map_payload_bytes": grid.memory_bytes()}
        print(json.dumps(summary, indent=2))
        if args.report:
            report = Path(args.report)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Sequence {args.seq} complete. Failed frames: {tel.failed_frames or 'none'}.")
    return 0


def cmd_benchmark(args) -> int:
    from src.benchmark.harness import run_benchmark

    cfg, ring_config = _load(args)
    results = run_benchmark(cfg, ring_config, str(args.rings), args.seq,
                            repo_root=str(REPO_ROOT), inference_mode=args.mode,
                            stride=args.stride, max_frames=args.max_frames)
    print(json.dumps({k: v for k, v in results.items() if k != "provenance"}, indent=2))
    print(f"Artifacts in {results['out_dir']}")
    return 0


def cmd_viz_spike(args) -> int:
    from src.viz.rerun_viewer import spike
    _, ring_config = _load(args)
    spike(ring_config)
    return 0


def cmd_analytics_d2(args) -> int:
    """Stage 4: AC-D2 kerb-vs-road variance measurement and occlusion-exclusion
    counters (AC-D5 behaviour) on real frames. Writes a JSON artifact to runs/."""
    from src.io.kitti_reader import SequenceReader
    from src.grid.foveated_grid import build_grid
    from src.analytics.cell_stats import compute_stats, kerb_road_variance_cells
    from src.analytics.negative_obs import negative_obstacle_candidates
    from src.segmentation.class_map import build_raw_to_class3

    cfg, ring_config = _load(args)
    reader = SequenceReader(cfg.data_root, args.seq)
    raw_to_3 = build_raw_to_class3()
    ring1 = ring_config.rings[0]

    road_vars, kerb_vars, kerb_list = [], [], []
    occ_totals = {"below_plane": 0, "missing_flagged": 0,
                  "occlusion_suppressed": 0, "no_plane": 0}
    n = len(reader) if args.max_frames is None else min(len(reader), args.max_frames)
    frames = list(range(0, n, args.stride))
    for i in frames:
        pts = reader.scan(i)
        raw = reader.labels(i)
        rv, kv, cells = kerb_road_variance_cells(
            pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64),
            pts[:, 2].astype(np.float64), raw,
            ring1.cell_size, ring1.radius_min, ring1.radius_max)
        road_vars.append(rv)
        kerb_vars.append(kv)
        kerb_list.extend((i, x, y, v, c) for x, y, v, c in cells)

        labels3 = raw_to_3[raw].astype(np.int8)
        grid = build_grid(pts[:, :3].astype(np.float64), labels3, ring_config)
        compute_stats(grid, cfg.projection.width)
        stats: dict = {}
        negative_obstacle_candidates(grid, ring_config, stats_out=stats)
        for k in occ_totals:
            occ_totals[k] += stats.get(k, 0)
        if i % (args.stride * 10) == 0:
            print(f"frame {i}/{n}")

    road = np.concatenate(road_vars) if road_vars else np.zeros(0)
    kerb = np.concatenate(kerb_vars) if kerb_vars else np.zeros(0)
    if road.size == 0 or kerb.size == 0:
        print("Not enough road/kerb cells found — nothing to report.", file=sys.stderr)
        return 2

    med_road, med_kerb = float(np.median(road)), float(np.median(kerb))
    # Separation: fraction of kerb cells above the road p90 (a threshold a
    # renderer could actually use).
    road_p90 = float(np.percentile(road, 90))
    frac_kerb_above = float(np.mean(kerb > road_p90))
    kerb_list.sort(key=lambda t: -t[3])
    spot_check = [{"frame": f, "x_m": round(x, 2), "y_m": round(y, 2),
                   "z_var": round(v, 5), "points": c}
                  for f, x, y, v, c in kerb_list[:30]]

    out = {
        "sequence": args.seq, "stride": args.stride, "frames_sampled": len(frames),
        "ring1_cell_m": ring1.cell_size,
        "road_interior_cells": int(road.size), "kerb_cells": int(kerb.size),
        "z_var_road": {p: float(np.percentile(road, q))
                       for p, q in (("p10", 10), ("p50", 50), ("p90", 90))},
        "z_var_kerb": {p: float(np.percentile(kerb, q))
                       for p, q in (("p10", 10), ("p50", 50), ("p90", 90))},
        "median_ratio_kerb_over_road": med_kerb / max(med_road, 1e-12),
        "frac_kerb_above_road_p90": frac_kerb_above,
        "spot_check_candidates": spot_check,
        "occlusion_counters": occ_totals,
    }
    path = Path(f"runs/stage4_analytics_seq{args.seq}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"AC-D2: road-interior cells {road.size}, kerb cells {kerb.size}; "
          f"median z-var road {med_road:.5f} vs kerb {med_kerb:.5f} "
          f"(ratio {out['median_ratio_kerb_over_road']:.1f}x); "
          f"{100 * frac_kerb_above:.0f}% of kerb cells above road p90.")
    print(f"Occlusion (AC-D5 on real data): {occ_totals}")
    print(f"Artifact: {path} — spot-check the top candidates manually "
          "(AC-D2 needs >= 20 verified cells).")
    return 0


def _spotcheck_html(title: str, intro: str, entries: list, out_path: Path) -> None:
    """Static review sheet: one figure per candidate with a verdict line the
    reviewer fills in. Plain HTML, self-contained (images embedded), no
    external assets."""
    rows = []
    for i, (caption, png_b64) in enumerate(entries, 1):
        rows.append(
            f'<div class="c"><h3>Candidate {i:02d}</h3>'
            f'<p class="cap">{caption}</p>'
            f'<img alt="candidate {i}" src="data:image/png;base64,{png_b64}">'
            f'<p class="verdict">Verdict (circle one): GENUINE / NOT / UNSURE'
            f' &nbsp; Notes: ______________________</p></div>')
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>"
        "body{background:#101418;color:#e8edf2;font-family:sans-serif;"
        "max-width:1000px;margin:0 auto;padding:2rem 1rem}"
        "h1{font-size:1.4rem} h3{margin:1.6rem 0 0.3rem}"
        ".cap{color:#9db0bf;font-family:monospace;font-size:0.85rem;margin:0 0 0.4rem}"
        "img{max-width:100%;border:1px solid #2a3540;border-radius:4px}"
        ".verdict{font-family:monospace;color:#c9d6e0;margin:0.4rem 0 0}"
        ".c{border-bottom:1px solid #2a3540;padding-bottom:1rem}"
        "</style></head><body>"
        f"<h1>{title}</h1><p>{intro}</p>" + "".join(rows) + "</body></html>")
    out_path.write_text(html, encoding="utf-8")


def cmd_spotcheck_d2(args) -> int:
    """AC-D2 manual verification sheet: for each top kerb candidate, render the
    real scan points around the cell (top-down by raw class, plus both height
    profiles) so a human can confirm a genuine kerb step in seconds."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.io.kitti_reader import SequenceReader

    cfg, _ = _load(args)
    src_json = Path(f"runs/stage4_analytics_seq{args.seq}.json")
    cands = json.loads(src_json.read_text(encoding="utf-8"))["spot_check_candidates"]
    reader = SequenceReader(cfg.data_root, args.seq)

    entries = []
    for c in cands:
        pts = reader.scan(c["frame"])
        raw = reader.labels(c["frame"])
        m = (np.abs(pts[:, 0] - c["x_m"]) < 1.5) & (np.abs(pts[:, 1] - c["y_m"]) < 1.5)
        x, y, z, lab = pts[m, 0], pts[m, 1], pts[m, 2], raw[m]
        col = np.where(lab == 40, "#3C78D8", np.where(lab == 48, "#E0A93E", "#888888"))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        axes[0].scatter(x, y, c=col, s=4)
        axes[0].add_patch(plt.Rectangle((c["x_m"] - 0.025, c["y_m"] - 0.025), 0.05, 0.05,
                                        fill=False, edgecolor="red", lw=1.5))
        axes[0].set_title("top-down (blue=road, amber=sidewalk)", fontsize=8)
        axes[0].set_aspect("equal")
        for ax, h, name in ((axes[1], x, "x"), (axes[2], y, "y")):
            ax.scatter(h, z, c=col, s=4)
            ax.set_title(f"{name} vs height - look for the step", fontsize=8)
        for ax in axes:
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        entries.append((
            f"frame {c['frame']}, cell centre ({c['x_m']}, {c['y_m']}) m, "
            f"z-var {c['z_var']}, {c['points']} points in cell (red box = the 5 cm cell)",
            base64.b64encode(buf.getvalue()).decode("ascii")))

    out = Path(f"runs/spotcheck_d2_seq{args.seq}.html")
    _spotcheck_html(
        "AC-D2 kerb spot-check - sequence " + args.seq,
        "Spec requires at least 20 manually verified cells. A candidate is "
        "GENUINE when the height profile shows a clear step between road "
        "(blue) and sidewalk (amber) at the marked cell. Record verdicts and "
        "keep this sheet with the run artifacts.",
        entries, out)
    print(f"{len(entries)} candidates rendered to {out}")
    return 0


def cmd_spotcheck_neg(args) -> int:
    """Stage-2 ground-truth spot-check sheet: pothole-plausible depression
    instances rendered from the global map (top-down by height + radial depth
    profile) for manual verification before any AC-D6 scoring."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inst = json.loads(Path(f"runs/neg_precheck_seq{args.seq}.json").read_text(encoding="utf-8"))
    plaus = [i for i in inst if i["n_cells"] >= 4 and i["depth"] < 0.5]
    plaus.sort(key=lambda i: -i["confidence"])
    plaus = plaus[:args.count]
    print(f"{len(inst)} instances total; reviewing top {len(plaus)} "
          "pothole-plausible (>=4 cells, depth < 0.5 m) by confidence")
    pts = np.load(f"runs/map_seq{args.seq}.npy")

    entries = []
    for c in plaus:
        cx, cy = c["centroid"]
        m = (np.abs(pts[:, 0] - cx) < 2.0) & (np.abs(pts[:, 1] - cy) < 2.0)
        x, y, z = pts[m, 0], pts[m, 1], pts[m, 2]
        r = np.hypot(x - cx, y - cy)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        sc = axes[0].scatter(x, y, c=z, s=3, cmap="viridis")
        fig.colorbar(sc, ax=axes[0], label="z (m)")
        axes[0].plot([cx], [cy], "rx", markersize=10)
        axes[0].set_title("top-down, colour = height", fontsize=8)
        axes[0].set_aspect("equal")
        axes[1].scatter(r, z, s=3, c="#7fb2d9")
        axes[1].set_title("distance from centroid vs height - look for the dip", fontsize=8)
        for ax in axes:
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        entries.append((
            f"centroid ({cx:.1f}, {cy:.1f}) m, depth {c['depth']:.2f} m, "
            f"{c['n_cells']} cells, confidence {c['confidence']:.2f}",
            base64.b64encode(buf.getvalue()).decode("ascii")))

    out = Path(f"runs/spotcheck_negatives_seq{args.seq}.html")
    _spotcheck_html(
        "Negative-obstacle ground-truth spot-check - sequence " + args.seq,
        "Stage-2 decision support (spec 11.3): these are the most confident "
        "pothole-plausible instances from the pre-check. GENUINE means the "
        "radial profile shows a real depression at the centroid; expect many "
        "to be pose-drift artifacts - that is what this sheet exists to "
        "establish before anything is scored against this ground truth.",
        entries, out)
    print(f"{len(entries)} instances rendered to {out}")
    return 0


def cmd_list_sequences(args) -> int:
    """S-1: sequence selector data as a CLI table."""
    from src.viz.rerun_viewer import list_sequences, format_sequence_table
    cfg, _ = _load(args)
    print(format_sequence_table(list_sequences(cfg.data_root), cfg.data_root))
    return 0


def cmd_viz(args) -> int:
    """Module E main viewer: replay a sequence into rerun (S-1 + S-2, and S-3
    when --benchmark-dir points at a finished run). Headless machines use
    --save and open the .rrd in a local viewer (or stream over an SSH tunnel)."""
    import rerun as rr

    from src.io.kitti_reader import SequenceReader
    from src.projection.range_projection import project, back_project
    from src.frame_pipeline import process_frame
    from src.analytics.cell_stats import compute_stats
    from src.grid.foveated_grid import FoveatedGrid
    from src.grid.compact import pack_block
    from src.segmentation.class_map import build_raw_to_class3
    from src.viz.telemetry import Telemetry
    from src.viz import rerun_viewer as rv

    cfg, ring_config = _load(args)
    rr.init(f"sih26053_seq{args.seq}", spawn=(args.save is None))
    if args.save is not None:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save)

    rr.log("selector", rr.TextDocument(
        rv.format_sequence_table(rv.list_sequences(cfg.data_root), cfg.data_root)),
        static=True)
    if args.benchmark_dir:
        rv.log_benchmark_results(args.benchmark_dir)

    reader = SequenceReader(cfg.data_root, args.seq)
    segmenter = None
    if args.mode in ("live", "random"):
        from src.segmentation.inference import Segmenter
        segmenter = Segmenter(cfg.weights_path, cfg.semantic_kitti_yaml,
                              device=cfg.device, fp16=cfg.fp16,
                              random_init=(args.mode == "random"))
    raw_to_3 = build_raw_to_class3()
    tel = Telemetry()
    tel.cpu_fallback = bool(segmenter and segmenter.cpu_fallback)
    n = len(reader) if args.max_frames is None else min(len(reader), args.max_frames)
    for i in range(0, n, args.stride):
        pts = reader.scan(i)
        proj = project(pts, cfg.projection)
        if segmenter is not None:
            label_img = segmenter.segment(proj.range_image)
            if label_img is None:
                tel.failed_frames.append(i)
                rv.log_frame_failure(i)
                continue
            labels3 = back_project(label_img, proj.pixel_map)
        else:
            labels3 = raw_to_3[reader.labels(i)].astype(np.int8)
        in_range = proj.pixel_map[:, 0] >= 0
        xyz = pts[in_range, :3].astype(np.float64)
        lab = labels3[in_range]
        result = process_frame(xyz, lab, ring_config, implementation=args.implementation,
                               image_width=cfg.projection.width,
                               memory_ceiling_bytes=int(cfg.memory_ceiling_mb*1e6),
                               depth_threshold=cfg.neg_depth_m, max_extent_m=cfg.neg_max_extent_m)
        grid, neg = result.grid, result.candidates
        # Same planar extent and same finished fields/layout in both maps.
        radius = np.hypot(xyz[:, 0], xyz[:, 1])
        selected = (radius > ring_config.radius_min) & (radius <= ring_config.radius_max)
        uniform = rv.uniform_comparison_block(xyz[selected], lab[selected], args.uniform_cell,
                                               ring_config.base_cell_size)
        temp = FoveatedGrid(ring_config)
        temp.blocks[0] = uniform
        compute_stats(temp, cfg.projection.width)
        if args.implementation == "corrected":
            uniform = pack_block(uniform)
        del temp
        tel.frame(grid.memory_bytes(), uniform.memory_bytes())
        rv.log_frame(grid, tel, i, ring_config, cfg.min_class_count,
                     show_rings=not args.no_rings, neg_candidates=neg,
                     uniform_block=uniform)
        if i % 50 == 0:
            print(f"frame {i}/{n}: {grid.n_cells} cells, {tel.status_line()}")
    print(f"Sequence {args.seq} logged. Failed frames: {tel.failed_frames or 'none'}."
          + (f" Recording saved to {args.save}" if args.save else ""))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="sih26053")
    p.add_argument("--rings", default=str(DEFAULT_RINGS))
    p.add_argument("--pipeline", default=str(DEFAULT_PIPELINE))
    p.add_argument("--data-root", default=None, help="override dataset location")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate-config").set_defaults(fn=cmd_validate_config)

    s = sub.add_parser("build-map")
    s.add_argument("--seq", default="00")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_build_map)

    s = sub.add_parser("precheck-negatives")
    s.add_argument("--seq", default="00")
    s.add_argument("--map", default=None)
    s.set_defaults(fn=cmd_precheck_negatives)

    s = sub.add_parser("latency-b")
    s.add_argument("--random", action="store_true",
                   help="random weights (timing only; never used for results)")
    s.add_argument("--iters", type=int, default=50)
    s.set_defaults(fn=cmd_latency_b)

    s = sub.add_parser("run")
    s.add_argument("--seq", default="00")
    s.add_argument("--mode", choices=["live", "random", "gt-labels"], default="live")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--implementation", choices=["core", "corrected", "legacy"], default="core")
    s.add_argument("--report", default=None)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("benchmark")
    s.add_argument("--seq", default="08")
    s.add_argument("--mode", choices=["live", "random", "gt-labels"], default="live")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(fn=cmd_benchmark)

    sub.add_parser("viz-spike").set_defaults(fn=cmd_viz_spike)

    sub.add_parser("list-sequences").set_defaults(fn=cmd_list_sequences)

    s = sub.add_parser("analytics-d2")
    s.add_argument("--seq", default="00")
    s.add_argument("--stride", type=int, default=20)
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(fn=cmd_analytics_d2)

    s = sub.add_parser("spotcheck-d2")
    s.add_argument("--seq", default="00")
    s.set_defaults(fn=cmd_spotcheck_d2)

    s = sub.add_parser("spotcheck-neg")
    s.add_argument("--seq", default="00")
    s.add_argument("--count", type=int, default=30)
    s.set_defaults(fn=cmd_spotcheck_neg)

    s = sub.add_parser("viz")
    s.add_argument("--seq", default="00")
    s.add_argument("--mode", choices=["live", "random", "gt-labels"], default="gt-labels")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--save", default=None,
                   help="write a .rrd recording instead of spawning a viewer "
                        "(for headless machines; open it locally)")
    s.add_argument("--uniform-cell", type=float, default=0.05,
                   help="cell size of the toggleable uniform comparison grid "
                        "(default: the 5 cm reference the memory claim is against)")
    s.add_argument("--no-rings", action="store_true")
    s.add_argument("--benchmark-dir", default=None,
                   help="finished benchmark run directory to show as S-3")
    s.add_argument("--implementation", choices=["corrected", "legacy"], default="corrected")
    s.set_defaults(fn=cmd_viz)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
