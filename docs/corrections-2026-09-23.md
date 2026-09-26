# Review corrections — 23 September 2026

## Implemented

- Compact coordinates are promoted before cell-key arithmetic. Component lookup
  uses integer coordinate pairs. Packed maps now run through the detector.
- Ray iteration compares the producing plane's tile with its landing tile;
  oscillating/unconverged intersections cannot generate missing-return evidence.
- BLIND requires a validated external azimuth mask. It is never inferred from
  the nearest return in the current frame. Without a mask, self-occlusion remains
  uncalibrated; candidate outputs are not confirmed hazards.
- Ground fits reject collinear/narrowly distributed samples and poor residuals.
  Each plane owns its own convex-hull support region; adjacent tiles cannot lend
  support. Mixed semantic cells are excluded from fitting. ROAD grouping still
  includes surfaces which may not be drivable.
- KD-tree queries check actual XY distance and height agreement. Raw scan
  positions are supplied by the application. Grid-only callers use approximate
  cell centres, explicitly.
- Direct construction packs one ring at a time and releases its wide temporary
  arrays. The retained layout is 56 bytes/cell. Counts are range-checked before
  narrowing. Empty comparisons are valid and mismatched cell sizes fail.
- A shared `src/frame_pipeline.py` path connects compact maps and corrected
  analytics to CLI `run`, CLI `viz`, and the dual VisPy viewer. `--implementation
  legacy` retains the old path. The canonical aggregation implementation/tests
  remain unchanged, with their original bit-identity contract.
- Repeated detection clears old flags. Rerun draws cluster extents and clears
  hazard entities when the next frame has no candidates. The dual viewer labels
  GT semantics and displays build/detection timings, not a claimed full FPS.
- The comparison viewer uses equivalent finished fields/layouts and planar
  extents for uniform/foveated map payload comparisons.
- Memory measurements run each builder in a separate fresh subprocess. Shared
  comparison-process RSS remains separately labelled. Neither measurement
  includes model inference or rendering.
- Detector reports explicitly distinguish old cells from new clusters, list
  frame IDs, report emitted cell counts and missing-return availability, and
  apply common per-kind range gates to reported outputs. Latencies measure the
  full respective detector; output filtering is not an algorithmic speedup.
- Diagnostic extraction refuses to overwrite an existing manifest, uses each
  site's local plane, and states that output-selected examples cannot establish
  recall. Existing 419 examples/empty verdicts are untouched.

## Validation and limits

Regression tests cover overflow, ray oscillation, geometric support, mixed
cells, XY/height matching, stale flags, numeric agreement, memory ceilings,
empty inputs and viewer clearing. SciPy is added to runtime and CI dependencies;
pytest and hypothesis are test dependencies.

Fresh checks include both SemanticKITTI 09 and 10, using ground-truth labels.
The smaller layout still reduces stored map payload from 84 to 56 bytes/cell
(33.3%). Whole-process peak RAM is measured separately and must not inherit that
percentage. Tightened geometric checks can cost more than the earlier rewrite;
new timings must replace previous presentation numbers.

Physical laser calibration and a validated ego-vehicle occlusion mask are not
available here. Uniform beam angles and projection-row medians are approximate,
not actual firing records. Ground bootstrap can fail: the detector then records
missing-return inference as unavailable, rather than declaring the area safe.
The unreached-cell estimate includes failed/unsupported intersections and is
not a measured unsampled-state distribution. Sensor height is an estimate.

The sparse ring map still assigns points by radius, with aligned lattices;
this does not solve full square-footprint overlap at radial boundaries. Compact
storage preserves counts and bounds numerical error on tested scans, not full
3D geometry. This work does not implement motion tracking, temporal mapping,
hazard-preserving budget refinement, or prove safety/novelty.

## Running the corrected path

From the repository, use its Python environment:

```bash
.venv/bin/python -m src.main --data-root data/semantickitti run --seq 09 --mode gt-labels --max-frames 10 --report output/corrected-run09.json
.venv/bin/python scripts/compare_grid_layouts.py --frames 15 --out output/corrected-layout-comparison.json
.venv/bin/python scripts/eval_negative_obs.py --frames 12 --out output/corrected-detector-evaluation.json
.venv/bin/python -m pytest tests/ -q
```

The GUI can be launched separately when desired. No server/VNC startup is needed
for the headless checks above. The historical benchmark command retains its
original algorithm and scoring protocol; use the new comparison script for
compact storage measurements.

## Accuracy work still required

Freeze independent frame IDs and complete spatial regions **before** tuning.
Annotate every visible hazard and explicit unknown/occluded regions, not only
emitted candidates. Keep tuning and evaluation regions separate. Score both
methods with the same instance-matching rule and range limits, reporting
precision, recall, missed hazards and false alarms per reviewed frame. Have a
person review ambiguous geometry; do not auto-fill truth from predictions.

The existing diagnostic manifest can help inspect failures, but labelling its
419 selected examples is not by itself a complete recall evaluation.

## Recorded verification

- Full suite: **76 passed, 3 skipped** (optional Torch/Rerun packages absent).
- Added 17 regression tests; both seq09/seq10 CLI runs completed without errors.
- Dual viewer preparation succeeded on seq09 frame000000; no GUI/server started.
- Ten-frame map comparison: median 3.49 MB
  original versus 2.33 MB compact.
- Separate-process build medians: 23.7 ms original,
  19.5 ms compact. Linux per-process VmHWM peaks:
  58.4 MB versus 55.7 MB.
  This small run is not a guaranteed whole-application memory/FPS improvement.
- Raw reports: `output/corrections-2026-09-23/`. The reports distinguish
  detector-only timing from headless GT-labelled processing (no rendering).
- Linux memory measurement uses VmHWM for the child process, avoiding the
  pre-exec high-water mark that getrusage may inherit from its parent.
