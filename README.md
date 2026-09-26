# FoveaMap — SIH26053

See [the complete file guide](FILE_GUIDE.md) and [clone/setup notes](REPOSITORY_SETUP.md).
Datasets, model binaries, local environments, and private credentials are excluded.

Adaptive variable-resolution 2.5D LiDAR mapping with separate ground/obstacle
heights, detailed semantics, vehicle-dependent traversability, and estimated
object motion. The current pipeline is `src/core`; historical benchmark modules
remain available for comparison. No guaranteed safety, lossless 3D compression,
or real-time throughput is claimed.

## Current implementation (September 25, 2026)

- **Terrain:** local ground fitting, slope/roughness/step checks, vertical clearance,
  vehicle-width obstacle inflation, and an explicit unknown state.
- **Objects:** SalsaNext's 19 semantic classes plus unknown; geometric tracking
  with ego-motion compensation estimates stationary/moving/unknown motion.
- **Grid:** disjoint occupied leaves at 5/10/20/40 cm; 40-byte records, ground and
  obstacle layers, minority-class presence masks, and explicit payload limits.
- Experimental pink pothole candidates are disabled in the default pipeline.
  There are no verified pothole annotations for our local sequences 09/10.

```bash
# Dataset labels: tests grid/terrain/tracking, not neural inference accuracy.
.venv/bin/python scripts/run_core.py --sequence 10 --mode gt-labels --frames 30
# Actual network predictions; labels are read only afterward for diagnostics.
.venv/bin/python scripts/run_core.py --sequence 10 --mode model --frames 30 --evaluate-labels
# Equal-input map payload comparisons (not total RAM).
.venv/bin/python scripts/compare_core_memory.py
# Three-panel live viewer; right panel uses actual model predictions.
./scripts/kitti_vnc.sh start 10 --mode model
.venv/bin/python -m pytest tests/ -q
```

See [implementation and measured limits](docs/core-improvements-2026-09-25.md),
[VNC controls](docs/kitti-vnc.md), and [vehicle settings](configs/core.yaml).
Sequence 09/10 diagnostics are **not held-out model validation**. Map payload,
process RAM, GPU memory, processing throughput, and displayed playback FPS are
separate measurements. The map is not necessarily smaller than raw points.

The historical [engineering spec](resource/SIH26053_Claude_Code_Spec.md) and
commands below describe earlier experiments; their old three-class, negative
obstacle, and bit-identical aggregation claims do not describe the new core.

## Layout

| Path | Module |
|---|---|
| `configs/` | Ring table (validated at startup, AC-C5), pipeline config, semantic-kitti extract |
| `src/core/` | Current terrain, detailed semantics, motion tracking and 40-byte sparse grid |
| `src/common/` | Config loader, ring validation, error types |
| `src/io/` | KITTI calib/poses/scans/labels with count verification (T-19) |
| `src/projection/` | Module A — spherical projection, collision handling, back-projection |
| `src/segmentation/` | Module B — SalsaNext, the learning-idx→3-class mapping (T-04) |
| `src/grid/` | **Module C — foveated grid engine (the contribution)**: shared lattice, ring assignment, Welford accumulation, bit-identical lossless aggregation (AC-C1) |
| `src/analytics/` | Module D — cell stats, safety-preserving render rule, negative obstacles |
| `src/offline/` | Module F — global map (`inv(Tr)`!), baselines, recall + Pareto |
| `src/benchmark/` | Provenance-gated benchmark harness (refuses cached inference) |
| `src/viz/` | Module E — telemetry (FPS/memory/banners), rerun.io viewer + spike |

The dataset location is set by `data_root` in `configs/pipeline.yaml` (locally `data/semantickitti`),
read-only, laid out as `<data_root>/sequences/00/{calib.txt, poses.txt,
velodyne/, labels/}`. Note: the calib archive ships poses at top-level
`poses/XX.txt` — move each to `sequences/XX/poses.txt`.

## Setup

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

## The three first checks (spec §0 — run before anything else)

```
# 1. Global map for sequence 00; asserts z-extent 10–40 m (catches the inv(Tr) bug)
python -m src.main build-map --seq 00

# 2. Negative-obstacle pre-check; counts instances and prints the §11.3 decision
python -m src.main precheck-negatives --seq 00

# 3. Module B latency in isolation on this machine (budget: 40 ms)
python -m src.main latency-b
```

## Everything else

```
python -m src.main validate-config          # ring table + pipeline config
python -m src.main run --seq 00 --mode live # full pipeline (PAC-1); --mode random proves AC-B3
python -m src.main benchmark --seq 08       # Pareto curve, single command (PAC-6); aborts without provenance
python -m src.main viz-spike                # the §12 rerun.io verification spike
python -m src.main list-sequences           # S-1 selector: sequences, counts, availability
python -m src.main viz --seq 00 --save runs/viz.rrd  # S-2/S-3 dashboard (headless: open the .rrd locally)
```

Per-stage status, data inventory, and measured results live in
[`docs/STAGE_STATUS.md`](docs/STAGE_STATUS.md) and [`docs/stages/`](docs/stages/).

Benchmark artifacts (results.json / results.csv / pareto.png + provenance)
land in `runs/`. Every number reported anywhere must trace to one of these runs.

## Weights

Place a SalsaNext **state dict** at `weights/salsanext_state_dict.pt`
(checkpoint from github.com/TiagoCortinhal/SalsaNext, converted offline if it
is a whole-model pickle). Missing weights abort at startup — there is no
silent random fallback.
