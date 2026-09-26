#!/usr/bin/env python3
"""Export REAL KITTI keyframes for the demoUI live replay.

Usage, from the repo root (venv active):
    python3 demoUI/export_frames.py --seq 08 --frames 3000 3005 3010

Writes demoUI/frames.js -> window.FOVEA_FRAMES. Per frame:
  - pts: flat [x_cm, y_cm, z_cm, cls, ...] int arrays, azimuth-SORTED so the
    dashboard's rotating sweep reveals points in scan order (real replay, not
    simulation). Points subsampled by --point-stride to bound file size.
  - rings: per-ring flat [cx, cy, cls, maxz_cm, meanz_cm, ...] cell arrays
    from the pipeline's own ring table; cell class is highest-severity
    (matches the render rule), heights are the real per-cell statistics the
    2.5D extrusion draws.
  - uniform_cells: the 5 cm uniform-grid occupied-cell COUNT for the live
    ratio stat (the uniform grid itself is not rendered).

Classes: ROAD->0, STATIC->1, DYNAMIC->2; IGNORE dropped. All real data, all
derived through src/ modules - nothing synthetic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.common.config import load_pipeline, load_rings                # noqa: E402
from src.io.kitti_reader import SequenceReader                         # noqa: E402
from src.segmentation.class_map import build_raw_to_class3             # noqa: E402

HERE = Path(__file__).resolve().parent
RANGE_MIN, RANGE_MAX = 2.5, 100.0


def ring_cells(x, y, z, cls, cell_size):
    cx = np.floor(x / cell_size).astype(np.int64)
    cy = np.floor(y / cell_size).astype(np.int64)
    order = np.lexsort((cls, cy, cx))
    cx_s, cy_s, z_s, cl_s = cx[order], cy[order], z[order], cls[order]
    new = np.empty(cx_s.shape[0], dtype=bool)
    new[0] = True
    new[1:] = (cx_s[1:] != cx_s[:-1]) | (cy_s[1:] != cy_s[:-1])
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], cx_s.shape[0]]
    group = np.cumsum(new) - 1
    zmax = np.maximum.reduceat(z_s, starts)
    zsum = np.add.reduceat(z_s, starts)
    counts = (ends - starts)
    zmean = zsum / counts
    flat = np.empty(starts.shape[0] * 5, dtype=np.int64)
    flat[0::5] = cx_s[starts]
    flat[1::5] = cy_s[starts]
    flat[2::5] = cl_s[ends - 1]              # highest severity: sorted last
    flat[3::5] = np.round(zmax * 100)
    flat[4::5] = np.round(zmean * 100)
    return flat.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, nargs="+", default=[3000, 3005, 3010])
    ap.add_argument("--point-stride", type=int, default=2)
    args = ap.parse_args()

    cfg = load_pipeline(str(REPO / "configs" / "pipeline.yaml"))
    rings = load_rings(str(REPO / "configs" / "rings.yaml"))
    reader = SequenceReader(cfg.data_root, args.seq)
    raw_to_3 = build_raw_to_class3()

    frames = []
    for fi in args.frames:
        pts = reader.scan(fi)
        labels3 = raw_to_3[reader.labels(fi)]
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        z = pts[:, 2].astype(np.float64)
        r = np.hypot(x, y)
        keep = (r >= RANGE_MIN) & (r <= RANGE_MAX) & (labels3 > 0)
        x, y, z, r = x[keep], y[keep], z[keep], r[keep]
        cls = (labels3[keep] - 1).astype(np.int64)

        # Ring cells and the uniform count use ALL kept points (the real grid).
        ring_layers = []
        for ring in rings.rings:
            m = (r >= ring.radius_min) & (r <= ring.radius_max)
            ring_layers.append({
                "cell": ring.cell_size,
                "cells": ring_cells(x[m], y[m], z[m], cls[m], ring.cell_size) if np.any(m) else [],
            })
        ux = np.floor(x / 0.05).astype(np.int64)
        uy = np.floor(y / 0.05).astype(np.int64)
        uniform_count = int(np.unique(ux * 8192 + uy).shape[0])

        # Replay points: subsampled, azimuth-sorted for the sweep reveal.
        sub = slice(None, None, args.point_stride)
        xs, ys, zs, cs = x[sub], y[sub], z[sub], cls[sub]
        az = np.mod(np.arctan2(ys, xs), 2 * np.pi)
        order = np.argsort(az, kind="stable")
        flat = np.empty(order.shape[0] * 4, dtype=np.int64)
        flat[0::4] = np.round(xs[order] * 100)
        flat[1::4] = np.round(ys[order] * 100)
        flat[2::4] = np.round(zs[order] * 100)
        flat[3::4] = cs[order]

        frames.append({
            "frame": fi,
            "n_points_full": int(x.shape[0]),
            "pts": flat.tolist(),
            "rings": ring_layers,
            "uniform_cells": uniform_count,
        })
        print(f"frame {fi}: {x.shape[0]} pts ({order.shape[0]} exported), "
              f"fov cells {sum(len(l['cells']) // 5 for l in ring_layers)}, "
              f"uniform {uniform_count}")

    payload = {
        "source": "KITTI / SemanticKITTI ground-truth labels",
        "sequence": args.seq,
        "point_stride": args.point_stride,
        "frames": frames,
    }
    out = HERE / f"frames_{args.seq}.js"
    out.write_text("// Generated by export_frames.py - REAL scan replay data.\n"
                   "window.FOVEA_FRAMES_SEQS = window.FOVEA_FRAMES_SEQS || {};\n"
                   f"window.FOVEA_FRAMES_SEQS[{json.dumps(args.seq)}] = "
                   + json.dumps(payload) + ";\n",
                   encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
