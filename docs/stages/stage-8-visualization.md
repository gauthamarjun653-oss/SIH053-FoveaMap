# Stage 8 — Visualization and demo

**Status: VIEWER CODE COMPLETE (2026-08-27); demo-machine measurements pending**

Purpose: the demonstration surface — the S-1/S-2/S-3 dashboard on rerun.io,
at 10 or more FPS sustained with live inference on the demo machine.

## What is built and tested (60/60 tests; smoke-verified on real seq-00 frames)

- S-1 sequence selector: `python -m src.main list-sequences` and a logged
  panel — frame counts, label availability, disabled-with-reason (missing
  poses, truncated labels), empty state names the configured path.
- S-2 main viewer: `python -m src.main viz --seq 00 [--mode gt-labels|random|live]
  [--save out.rrd]` — class and elevation layers, toggleable ring overlays,
  uniform 5 cm comparison grid logged as a sibling entity on the same frame
  timeline (comparison BY TOGGLE per spec; switching layers cannot move the
  frame index, so AC-E6 holds structurally), both grids' memory plus FPS
  always in the status line (AC-E4), negative-obstacle confidence tiers
  (0.3 / 0.5 / 0.7 — rerun has no data-driven slider; the tiers are the
  documented stand-in), frame-failure timeline markers, cached-mode and
  CPU-fallback banners (AC-E5).
- S-3 benchmark panel: `--benchmark-dir <runs/...>` logs the Pareto PNG,
  per-band metrics table, and provenance from a finished harness run.
- Fixed a latent API break: rerun 0.36 removed `set_time_sequence`; a new
  API-canary test now runs `log_frame` against the installed SDK.

## Headless usage (this VM has no display)

```
python -m src.main viz --seq 00 --save runs/viz_seq00.rrd
```

Open the `.rrd` in a local rerun viewer.

## Demo recording artifact (2026-08-27)

A real recording exists: `runs/viz_seq00_gt.rrd` (336 MB, also in S3
`runs/azure-vm/`) - 200 frames of sequence 00 (stride 5, gt-labels mode)
with the foveated map, uniform comparison layer, ring overlays, negative-
obstacle tiers, live status line, and the S-3 benchmark panel embedded
(preliminary Pareto plot, per-band metrics, provenance). Open it in a local
rerun viewer:

```
rerun runs/viz_seq00_gt.rrd
```

## Remaining for this stage

- AC-E1: 10 or more FPS sustained with live inference — measured on the demo
  machine only (official FPS figures come from the RTX 3050 demo machine;
  the physical demo laptop is a finale-day decision).
- AC-E2: all classes distinguishable, verified by someone outside the team.
