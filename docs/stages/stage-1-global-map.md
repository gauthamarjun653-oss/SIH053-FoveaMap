# Stage 1 — First check 1: global map

**Status: DONE** (2026-08-27, on the Azure VM)

Purpose: build the sequence-00 global map and validate it before anything is
derived from it. This is the check that catches the `inv(Tr)` camera-frame bug
(spec 11.1 / T-17).

## Command

```
python -m src.main build-map --seq 00
```

## Data required

Sequence 00 velodyne + labels + calib + poses on the local machine. All
present on the VM (see stage 0).

## Measured results

| Check | Result |
|---|---|
| AC-F1 z-extent (bounds 10-40 m) | PASS: 23.6 m (robust p99.9-p0.1; raw 53.8 m, see below) |
| AC-F2 median facade thickness (< 15 cm) | PASS: 13.1 cm |
| Map size | 46,802,227 deduplicated points, `runs/map_seq00.npy` (gitignored, local to the VM) |

## Finding: mirror-reflection phantom points

The first run failed AC-F1 at 53.8 m raw z-extent on a map that was actually
correct (pose y-spread 25.5 m; bulk z-distribution p0.1-p99.9 = 22.4 m). The
excess came from a few dozen phantom points down to 29 m below ground
(mirror reflections off glass, clustered near frames ~1040 and ~1480).

Fixes, both unit-tested:

- `assert_z_extent` measures p99.9-p0.1 instead of raw max-min. The
  camera-frame bug this gate exists for shifts the bulk of the distribution
  (about 500 m span), so the robust gate still catches it; a warning reports
  raw vs robust whenever they diverge.
- `derive_ground_truth_depressions` gained a phantom guard (cells with
  apparent depth over 5 m are sensor artifacts, excluded with a logged
  count) — without it one phantom point becomes a fake 25 m-deep
  "depression" in stage 2 ground truth.

Downstream rule this establishes: elevation statistics over the global map
must stay percentile-based.
