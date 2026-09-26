#!/usr/bin/env python3
"""Dual view on one VNC window: LEFT = raw 3D scan, RIGHT = the same scan
processed through the real pipeline into the 2.5D foveated grid. One shared
camera (orbiting one orbits both) and one frame index (n/b step both), so
the 3D-to-2.5D transformation is directly comparable scan by scan.

Keys: n / b step scans, r reset camera, q quit. Drag = orbit, wheel = zoom.
"""
import argparse
import sys
from time import perf_counter
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vispy import app, scene  # noqa: E402

from src.common.config import RingConfig, RingSpec, load_pipeline, load_rings              # noqa: E402
from src.io.kitti_reader import SequenceReader                       # noqa: E402
from src.segmentation.class_map import build_raw_to_class3           # noqa: E402
from src.grid.compact import build_compact_grid
from src.grid.foveated_grid import build_grid
from src.frame_pipeline import process_frame                        # noqa: E402
from src.analytics.cell_stats import compute_stats, render_class     # noqa: E402

CLASS_RGB = {
    0: (0.35, 0.35, 0.35),
    1: (0.24, 0.47, 0.85),   # ROAD - blue
    2: (0.88, 0.66, 0.24),   # STATIC - amber
    3: (0.90, 0.24, 0.24),   # DYNAMIC - red
}


VIRIDIS = np.array([   # compact viridis control points
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]])

def depth_colormap(rng):
    """Original-dataset look: range-coloured raw points (PRBonn-style)."""
    t = np.clip(rng / 60.0, 0, 1) * (len(VIRIDIS) - 1)
    i = np.floor(t).astype(int)
    i2 = np.minimum(i + 1, len(VIRIDIS) - 1)
    f = (t - i)[:, None]
    return VIRIDIS[i] * (1 - f) + VIRIDIS[i2] * f


def process(reader, raw_to_3, rings, cfg, idx, pt_stride=2, implementation="corrected"):
    started = perf_counter()
    pts = reader.scan(idx)
    labels3 = raw_to_3[reader.labels(idx)].astype(np.int8)
    keep = labels3 > 0
    xyz_all = pts[keep, :3].astype(np.float64)
    lab_all = labels3[keep]

    # RAW 3D: the ORIGINAL dataset, untouched - every point of the scan
    # (no label masking, no semantic colours), shaded by range with the
    # classic depth colormap, exactly the raw-data look.
    prep_started = perf_counter()
    xyz = pts[::pt_stride, :3].astype(np.float64)
    rng = np.hypot(xyz[:, 0], xyz[:, 1])
    col = np.ones((xyz.shape[0], 4))
    col[:, :3] = depth_colormap(rng)

    raw_prep_ms = (perf_counter()-prep_started)*1000

    # 2.5D: the real pipeline
    result = process_frame(xyz_all, lab_all, rings, implementation=implementation,
                           image_width=cfg.projection.width,
                           memory_ceiling_bytes=int(cfg.memory_ceiling_mb*1e6),
                           depth_threshold=cfg.neg_depth_m, max_extent_m=cfg.neg_max_extent_m)
    grid = result.grid
    centers, sizes, colors = [], [], []
    for block in grid.blocks.values():
        n = len(block)
        if n == 0:
            continue
        cls = render_class(block.hist, cfg.min_class_count)
        cx = (block.cx + 0.5) * block.cell_size
        cy = (block.cy + 0.5) * block.cell_size
        centers.append(np.stack([cx, cy, block.mean], axis=1))
        sizes.append(np.full(n, block.cell_size))
        base = np.array([CLASS_RGB[int(c)] for c in cls])
        t = np.clip((block.mean + 2.2) / 3.4, 0.25, 1.0)[:, None]
        colors.append(np.concatenate([base * t, np.ones((n, 1))], axis=1))
    # A comparable baseline: identical points, extent, fields and precision.
    reference_rings = RingConfig(rings.base_cell_size, (RingSpec(
        1, rings.radius_min, rings.radius_max, rings.base_cell_size),))
    baseline_started = perf_counter()
    if implementation == "corrected":
        reference = build_compact_grid(xyz_all, lab_all, reference_rings,
                                        image_width=cfg.projection.width)
    else:
        reference = compute_stats(build_grid(xyz_all, lab_all, reference_rings),
                                    cfg.projection.width)
    uniform_ms = (perf_counter()-baseline_started)*1000
    measurements = {
        "raw_scan_bytes": pts.nbytes,  # full original XYZI, not downsampled display
        "raw_points": len(pts), "displayed_points": len(xyz),
        "raw_prep_ms": raw_prep_ms,
        "grid_bytes": grid.memory_bytes(), "grid_cells": grid.n_cells,
        "grid_build_ms": result.build_ms, "hazard_ms": result.detection_ms,
        "uniform_bytes": reference.memory_bytes(), "uniform_cells": reference.n_cells,
        "uniform_build_ms": uniform_ms,
        "payload_saved_pct": 100*(1-grid.memory_bytes()/reference.memory_bytes()) if reference.memory_bytes() else 0.,
        "instrumented_frame_ms": (perf_counter()-started)*1000,
    }
    del reference  # comparison payload is not retained across frames
    return (xyz, col,
            np.concatenate(centers) if centers else np.empty((0, 3)),
            np.concatenate(sizes) if sizes else np.empty(0),
            np.concatenate(colors) if colors else np.empty((0, 4)), grid, result, measurements)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="09")
    ap.add_argument("--dataset", default=str(REPO / "data" / "semantickitti"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--implementation", choices=["corrected", "legacy"], default="corrected")
    ap.add_argument("--play", action="store_true", help="automatically replay scans")
    ap.add_argument("--interval", type=float, default=0.5, help="minimum playback interval in seconds")
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval must be positive")

    cfg = load_pipeline(REPO / "configs" / "pipeline.yaml")
    rings = load_rings(REPO / "configs" / "rings.yaml")
    reader = SequenceReader(args.dataset, args.sequence)
    raw_to_3 = build_raw_to_class3()

    canvas = scene.SceneCanvas(title="FoveaMap: raw 3D vs 2.5D - seq " + args.sequence,
                               keys="interactive", size=(1600, 900),
                               bgcolor="#0b0e12", show=True)
    grid_w = canvas.central_widget.add_grid()
    view_l = grid_w.add_view(row=0, col=0, border_color=(1, 1, 1, 0.15))
    view_r = grid_w.add_view(row=0, col=1, border_color=(1, 1, 1, 0.15))
    cam = scene.TurntableCamera(up="+z", fov=45, distance=55,
                                elevation=32, azimuth=-90)
    view_l.camera = cam
    view_r.camera = scene.TurntableCamera(up="+z", fov=45, distance=55,
                                         elevation=32, azimuth=-90)
    cam.link(view_r.camera)       # separate scene nodes, synchronized views

    raw_vis = scene.visuals.Markers(parent=view_l.scene)
    raw_vis.antialias = 0
    cell_vis = scene.visuals.Markers(parent=view_r.scene)
    cell_vis.scaling = "scene"    # cell squares at true footprint
    cell_vis.antialias = 0
    hazard_vis = scene.visuals.Markers(parent=view_r.scene)

    theta = np.linspace(0, 2 * np.pi, 128)
    for v in (view_l, view_r):
        for ring in rings.rings:
            pts_ring = np.stack([ring.radius_max * np.cos(theta),
                                 ring.radius_max * np.sin(theta),
                                 np.full_like(theta, -1.7)], axis=1)
            scene.visuals.Line(pts_ring, color=(1, 1, 1, 0.22), parent=v.scene)

    def text_visual(size=12, color="#e8f0fa", anchor_y="top"):
        node = scene.visuals.Text("", parent=canvas.scene, color=color,
                                  anchor_x="left", anchor_y=anchor_y, font_size=size)
        node.order = 101
        node.set_gl_state("translucent", depth_test=False)
        return node

    # Overlay cards are positioned in screen pixels, independently of the camera.
    cards = []
    for _ in range(4):
        card = scene.visuals.Rectangle(center=(1,1), width=1, height=1,
                      color=(.035,.065,.10,.94), border_color=(.22,.34,.46,1),
                      parent=canvas.scene)
        card.order = 100
        card.set_gl_state("translucent", depth_test=False)
        cards.append(card)
    title_l, title_r = text_visual(14,"#79caff"), text_visual(14,"#80e0ad")
    main_l, main_r = text_visual(18), text_visual(18)
    detail_l, detail_r = text_visual(11), text_visual(11)
    bottom_l, bottom_r = text_visual(11), text_visual(11)
    controls = text_visual(10,"#c8d3df")
    state = {"idx": args.start, "playing": args.play, "metrics": None,
             "drawn_idx": None, "pending_idx": None}
    frame_times = deque(maxlen=30)
    history = deque(maxlen=30)

    def update_hud():
        width,height = canvas.size
        half = width/2
        panel_width = max(240, half-32)
        for card, left, top, card_h in [(cards[0],16,16,190),(cards[1],half+16,16,190),
                                       (cards[2],16,height-139,101),(cards[3],half+16,height-139,101)]:
            card.center = (left+panel_width/2, top+card_h/2)
            card.width,card.height = panel_width,card_h
        for title,main,detail,left in [(title_l,main_l,detail_l,30),(title_r,main_r,detail_r,half+30)]:
            title.pos=(left,42);main.pos=(left,95);detail.pos=(left,182)
        bottom_l.pos=(30,height-56);bottom_r.pos=(half+30,height-56)
        controls.pos=(20,height-24)
        m=state["metrics"]
        if m is None:
            return
        fps=(len(frame_times)-1)/(frame_times[-1]-frame_times[0]) if len(frame_times)>1 else 0.
        fps_text=f"{fps:.2f}" if state["playing"] and len(frame_times)>1 else ("warming up" if state["playing"] else "paused")
        mean_raw=np.mean([v["raw_prep_ms"] for v in history])
        mean_grid=np.mean([v["grid_build_ms"] for v in history])
        mean_uniform=np.mean([v["uniform_build_ms"] for v in history])
        mean_hazard=np.mean([v["hazard_ms"] for v in history])
        title_l.text="RAW 3D LiDAR"
        title_r.text="FOVEATED 2.5D MAP"
        main_l.text=f"Memory  {m['raw_scan_bytes']/1e6:.2f} MB     Playback  {fps_text} FPS"
        main_r.text=f"Memory  {m['grid_bytes']/1e6:.2f} MB     Playback  {fps_text} FPS"
        detail_l.text=(f"{m['raw_points']:,} original returns | {m['displayed_points']:,} displayed\n"
                       f"Preparation: {mean_raw:.1f} ms | capacity {1000/max(mean_raw,1e-9):.1f} FPS\n"
                       "Memory = original XYZI scan; no map or semantic layers\n"
                       "Prep excludes file reading, mapping and drawing")
        detail_r.text=(f"{m['grid_cells']:,} occupied cells | ground-truth semantics\n"
                       f"Grid build: {mean_grid:.1f} ms | capacity {1000/max(mean_grid,1e-9):.1f} FPS\n"
                       f"Hazard detection: {mean_hazard:.1f} ms | pink = candidates\n"
                       "Memory = stored map arrays; excludes working/GPU memory")
        bottom_l.text=("HOW TO READ THESE NUMBERS\n"
                       "Playback FPS counts new scan frames drawn; shared by both views.\n"
                       "Capacity = 1000 / average stage ms; not end-to-end FPS.\n"
                       "Raw scan and map store different information; compare grids at right.")
        build_change=100*(1-mean_grid/max(mean_uniform,1e-9))
        speed_word="less" if build_change>=0 else "more"
        bottom_r.text=("FAIR REFERENCE: uniform 5 cm grid, same scan / fields / precision\n"
                       f"Uniform: {m['uniform_bytes']/1e6:.2f} MB | build {mean_uniform:.1f} ms ({1000/max(mean_uniform,1e-9):.1f} FPS capacity)\n"
                       f"Foveated saves {m['payload_saved_pct']:.1f}% map memory | {abs(build_change):.1f}% {speed_word} build time\n"
                       "Reference is measured live; its extra work slows this comparison demo.")
        controls.text=(f"SEQ {args.sequence}  |  scan {state['idx']}/{len(reader)-1}  |  "
                       f"{'PLAYING' if state['playing'] else 'PAUSED'}  |  Space: play/pause  N/B: step  R: reset  |  drag: rotate  scroll: zoom")
        canvas.update()

    @canvas.events.resize.connect
    def on_resize(_event):
        update_hud()

    @canvas.events.draw.connect
    def on_draw(_event):
        if state["pending_idx"] is not None and state["drawn_idx"] != state["pending_idx"]:
            state["drawn_idx"] = state["pending_idx"]
            if state["playing"]:
                frame_times.append(perf_counter())
            update_hud()

    def show(idx):
        idx = max(0, min(len(reader) - 1, idx))
        state["idx"] = idx
        xyz, col, centers, sizes, colors, grid, result, measurements = process(
            reader, raw_to_3, rings, cfg, idx, implementation=args.implementation)
        raw_vis.set_data(pos=xyz, face_color=col, size=2.2, edge_width=0)
        cell_vis.set_data(pos=centers, size=sizes, face_color=colors,
                          edge_width=0, symbol="square")
        h = result.detection.sensor_height if result.detection else None
        hazard_pos = np.array([[c.x, c.y, -(h or 1.73)] for c in result.candidates],
                              dtype=np.float32).reshape(-1, 3)
        hazard_vis.set_data(pos=hazard_pos, face_color=(1, 0, 1, .8), size=7, edge_width=0)
        state["metrics"] = measurements
        state["pending_idx"] = idx
        history.append(measurements)
        update_hud()

    @canvas.events.key_press.connect
    def on_key(ev):
        if ev.key == "N":
            show(state["idx"] + 1)
        elif ev.key == "B":
            show(state["idx"] - 1)
        elif ev.key == "Space":
            state["playing"] = not state["playing"]
            frame_times.clear()
            update_hud()
        elif ev.key == "Q":
            canvas.close()
        elif ev.key == "R":
            cam.reset()
            cam.distance = 55
            cam.elevation = 32
            cam.azimuth = -90

    def advance(_event):
        if state["playing"]:
            if state["idx"] + 1 >= len(reader):
                state["playing"] = False
                frame_times.clear()
                update_hud()
            else:
                show(state["idx"] + 1)

    show(args.start)
    timer = app.Timer(interval=args.interval, connect=advance, start=True)
    app.run()


if __name__ == "__main__":
    main()
