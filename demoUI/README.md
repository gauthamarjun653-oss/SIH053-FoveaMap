# FOVEA MAP Demo Dashboard (Module E / S-2 shell)

A single self-contained dark-theme dashboard: uniform 5 cm baseline vs the
foveated 2.5D grid, same synthetic scene on both canvases, ring boundaries at
10/25/50 m, metrics panels, and a fixed telemetry strip. No build step, no
server required, no external dependencies.

## Run it

Open `index.html` in a browser. That is all.

Hover to inspect (raycast picking): on the left panel the cursor picks the
nearest 3D point (coordinates, class, range, destination ring; auto-orbit
pauses while hovering); on the right panel it picks a grid cell — column
faces first, then the ground plane — and the tooltip shows the cell's stored
statistics (class, mean z, max z, ring, cell size) read from the same maps
the renderer draws, so the render and the inspection cannot disagree. The
middle 2D occupancy panel picks too - its tooltip honestly reports "class
unknown, height unknown", because occupancy stores neither; that is the
comparison made tangible.

Note on `file://`: Chrome blocks `fetch()` from local files, so the dashboard
loads `metrics.json` when served over http and falls back to `metrics.js`
(which sets `window.FOVEA_METRICS`, same data) when opened from disk. Both
files are generated together - never edit `metrics.js` by hand.

## Where the numbers come from

Every displayed value comes from `metrics.json`. Nothing is hardcoded in the
markup; missing or null fields render as NOT MEASURED, never a default.

- `"measured": false` - persistent hazard banner: SIMULATED DATA. Not
  dismissible (mirrors AC-E5: the banner is a functional requirement).
- `"measured": true` - the banner is replaced by the provenance line (commit,
  hardware, sequence, inference mode, timestamp).
- No metrics file at all - NO METRICS FILE banner and NOT MEASURED in every
  slot.
- The memory-reduction hero figure is computed at runtime from
  `baseline.memory_mb / foveated.memory_mb`.
- Telemetry latency split is derived at runtime from the fps figure; if fps
  is null the strip shows NOT MEASURED. If `provenance.inference` is
  `"cached"` the strip shows CACHED - FPS NOT REPRESENTATIVE in amber; any
  non-"live" mode is flagged as NOT LIVE.

## Wiring real data

Numbers (metrics panels, banner/provenance):

```bash
python3 wire_metrics.py ../runs/seq08_gt-labels   # from a benchmark run dir
python3 wire_metrics.py --placeholder             # restore placeholder values
```

Scene (both canvases). A real KITTI scan replaces the synthetic ray-cast when
`scene.js`/`scene.json` exist (produced from a scan + ground-truth labels by
the pipeline's own reader, class map, and ring table):

```bash
python3 export_scene.py --seq 08 --frame 3000     # run from the repo root, venv active
```

The legend line then reads REAL KITTI with the sequence/frame, and the cell
counts shown there are counted from the very lists being drawn - the render
and the count cannot disagree. Delete `scene.js` + `scene.json` to fall back
to the synthetic scene (the legend says so explicitly). In real-scene mode
the simulated moving cars are disabled; real vehicles are in the scan as the
dynamic class.

The wiring script fills what the harness actually measured (memory, recall
per band, full provenance) and writes null for what it did not (fps,
occupied cells) - those slots then read NOT MEASURED on screen. fps arrives
with the stage-5/6 GPU runs.

## Known divergence to reconcile (owner decision)

This dashboard follows the build prompt's class colours (terrain green,
static red, dynamic blue). The rerun viewer and pitch simulation use the
project's earlier fixed palette (road blue, static amber, dynamic red).
AC-E2 wants one consistent palette everywhere - pick one and align the other
surface.
