# Core improvements and verified limits — 25 September 2026

The three primary problem-statement tasks now run through `src/core`. The work
extends the existing SalsaNext, SemanticKITTI reader, grid configuration and
PRBonn viewer. Historical implementations remain available for comparisons.

## 1. Terrain analysis

**Implemented:** fit a robust local road/parking ground plane; retain separate
ground mean, variance and min/max heights; check slope, roughness and step-height
changes; compare obstacle clearance with vehicle height; expand observed
obstacles by vehicle half-width plus lateral margin. Decisions are **drivable,
blocked, or unknown**. Low support, poor fits and uncertain semantic evidence do
not silently become drivable. Sidewalk/other-ground/terrain are blocked by the
current conservative vehicle policy; this is not a universal off-road policy.

Example: a road return at -1.7 m and a branch at +1.0 m share a cell. Their
heights are kept separately instead of averaging them into fictitious ground.
A low branch blocks the cell when clearance is below vehicle height + margin.
An obstacle near a road cell also blocks that cell when the configured vehicle
width would overlap it.

Defaults in `configs/core.yaml`: vehicle height 1.5 m, width 1.8 m, vertical and
lateral margins 0.2 m, maximum slope 10 degrees, roughness 0.08 m, step 0.15 m.
These require calibration for the real vehicle. Only observed geometry is tested;
unobserved space is not certified clear. A negative height discontinuity can be
blocked, but this is **not validated pothole identification**. Seq 09/10 pothole
presence remains unverified. No pink hazard-candidate overlay runs by default.

## 2. Objects and motion

**Implemented:** actual pretrained SalsaNext predictions retain all 19 semantic
classes plus unknown. Low-confidence pixels become unknown. Movable-class points
are geometrically clustered and associated across timestamped scans. KITTI
LiDAR-to-world poses compensate for the sensor vehicle's own motion. Consistent
centroid displacement yields estimated moving/stationary states; ambiguous,
short or unstable tracks remain unknown. History and track count are bounded.

Example: a parked car moves backward in the sensor view as our vehicle advances.
Its compensated world position stays approximately fixed, so it can be labelled
stationary after sufficient observations. A driving car changes world position.
Simply predicting the class `car` does not establish motion.

Ground-truth moving labels and instance IDs never drive the tracker. The model
mode reads evaluation labels only after timed processing. The viewer's middle
panel is explicitly a ground-truth reference; the right panel declares its
semantic source. This is segmentation plus baseline geometric tracking, not a
trained 3D bounding-box detector or a production motion-segmentation model.

Remaining failure cases: occlusion changes centroids, nearby same-class objects
merge, cluster splits create new tracks, pose errors create apparent motion, and
slow movement between thresholds remains unknown. No online odometry has been
implemented; a live vehicle must provide reliable poses and timestamps.

## 3. Adaptive spatial representation and memory

**Implemented:** direct construction of sparse occupied leaves on a shared 5 cm
lattice. The configured levels are 5 cm at 2.5–10 m, 10 cm at 10–25 m, 20 cm at
25–50 m and 40 cm at 50–100 m. Whole parent squares split where they touch a
finer zone, so coarse/fine leaves cannot overlap at circular boundaries. Fine
resolution extends slightly beyond a nominal ring boundary. 40 cm is chosen for
dyadic nesting; the PS's 50 cm example is not the actual configured value.

Each occupied cell stores **40 bytes**: compact coordinates/level; dominant
semantic, motion and terrain states; counts; ground height statistics; obstacle
height bounds; and a 20-bit semantic-presence mask. A minority pedestrian return
is retained in the presence mask even when road points dominate the cell.

We removed per-cell class-frequency arrays and persistent derived arrays from
the current map. This deliberately compresses information: exact within-cell XY
geometry and per-class frequencies are not retained. Count conservation and
non-overlapping ownership are tested; **lossless reconstruction of 3D points is
not claimed**. The 8 MB default ceiling limits final map payload, not total RAM.
Overflow or budget excess raises an error instead of silently dropping points.
There is no budget-driven refinement/merging controller yet.

### Equal-input memory check

10 scans per sequence, identical finite points inside planar 2.5–100 m including
unknown labels. All figures below are median array payload in decimal MB.

| Sequence | Previous map | New map | Uniform 5 cm, same new fields | Minimal sparse 3D reference |
|---|---:|---:|---:|---:|
| 09 | 3.423 | 2.449 | 3.166 | 1.964 |
| 10 | 3.782 | 2.704 | 3.527 | 1.932 |

Median per-frame savings: **28.44% / 28.49% versus the previous map**, and
**22.54% / 23.27% versus the equivalent-field uniform 2.5D map** for 09 / 10.
The old and new map schemas differ, so the old/new comparison measures a real
storage tradeoff, not identical information preservation. The minimal 3D
reference stores just coordinates, count and semantic presence (20 bytes/voxel);
it lacks the new elevation/clearance summaries. Our map is larger than this
minimal reference and can be larger than raw XYZI scans. A significant reduction
against a fair high-resolution 3D navigation-map baseline remains to be shown.

Reproduce: `.venv/bin/python scripts/compare_core_memory.py`.
Artifact: `output/core-memory-comparison.json`.

A verified live screenshot is saved at `output/core-live-demo.png`. At seq-10
frame 358 it showed 2.82 FPS playback, CUDA predictions, a 1.56 MB map and a
1.99 MB raw scan. This is a single frame, not an average or a universal saving.

## Actual inference performance

Python 3.14, PyTorch 2.12.1+cu130, FP16 SalsaNext, NVIDIA RTX 3050 6 GB Laptop GPU.
30 consecutive frames starting at frame 0 per sequence, with evaluation labels
read after each timed frame. No viewer or uniform reference in these runs.
Processing includes scan loading, projection/inference, back-projection, tracking,
grid construction and terrain assessment. First-frame warmup is included.

| Sequence | Processing throughput | Median frame latency | Median map payload | Peak process RSS |
|---|---:|---:|---:|---:|
| 09 | 3.82 FPS | 250.3 ms | 2.413 MB | 1372.8 MB |
| 10 | 4.14 FPS | 227.4 ms | 2.818 MB | 1367.5 MB |

Peak PyTorch CUDA allocations: 191.0 MB; reserved: 245.4 MB. These are allocator
measurements, not all GPU memory (driver/context/rendering also consume memory).
Map payload is not process RSS. Viewer playback includes rendering and may be
slower. Seq-10 median core time is 168.6 ms, of which terrain is 119.8 ms, grid
32.9 ms and tracking 15.3 ms. Median components need not sum exactly. Terrain
fitting is the next optimization target. This is **not yet a 10 Hz pipeline**.

The CPU diagnostic processed 10 seq-10 frames at 1.03 FPS (median 854 ms).
Different frame counts and environments mean this is illustrative, not a
controlled CPU-versus-GPU speedup benchmark.

### Accuracy interpretation

Seq 09/10 belong to the pretrained model's training split. These are diagnostic
smoke tests, not held-out accuracy. No per-distance generalization claim is made.
JSON reports include per-class and distance-band IoU, motion confusion matrices,
unknown coverage, timings, RSS and GPU memory.

In the GPU samples, motion was decided on **50.6% of relevant GT points in 09**
and **92.6% in 10**. The decided points matched the dataset motion labels, but
this small, highly imbalanced sample must not be advertised as 100% tracking
accuracy. Unknown points remain in the denominator for coverage. Longer balanced
samples, occlusions and transitions are needed; dataset motion labels are not
always instantaneous velocity truth.

Reports: `output/core-model-gpu09.json`, `output/core-model-gpu10.json`.

## Verification and implementation assessment

**100 tests passed; 1 optional visualization test skipped.** New tests cover
non-overlapping ring boundaries, count accounting, overflow/budget handling,
minority semantic presence, ground/overhang separation, sparse/unknown terrain,
steep terrain, height depressions, vehicle width, ego translation/rotation,
stationary/moving synthetic objects, bounded/expired tracks, seek resets,
disabled hazard computation, confidence abstention and non-finite model output.

For a prototype review, my provisional engineering assessment is **terrain 7/10,
objects/motion 7/10, grid 8/10**. These are subjective readiness scores, not
accuracy percentages or judge guarantees. The scores rely on functionality and
unit tests; field validation and throughput still limit overall readiness.

## What to do next, in order

1. Validate on the official held-out sequence 08 with fixed thresholds. Report
   per-class/per-distance scores, traversability precision/recall against a
   curated terrain set, and motion accuracy together with unknown coverage.
2. Profile and batch local terrain fits, reuse neighborhoods where safe, and
   measure p50/p95 processing plus displayed FPS on the final demo hardware.
   Keep the uniform comparison off except while demonstrating memory.
3. Add a labelled sequence of real depressions, curbs and overhangs before making
   pothole/safety claims. Absence of returns must remain unknown.
4. Stress tracking with parked/moving transitions, close vehicles and occlusion;
   add predictive association and track-level uncertainty before richer motion
   claims. Incorporate real odometry for live deployment.
5. Evaluate a sparse 3D navigation baseline with matched useful fields and
   accuracy constraints. Then investigate hazard-preserving refinement under a
   fixed memory budget. This potential contribution is not established novelty.

## Main changed files

- `src/core/{semantics,grid,terrain,motion,pipeline}.py`: the three-task pipeline.
- `configs/core.yaml`: vehicle limits and final map payload ceiling.
- `src/segmentation/inference.py`: detailed predictions, confidence abstention,
  strict weight validation, CPU fallback and GPU inference.
- `scripts/run_core.py`: reproducible model/GT runs and diagnostic reports.
- `scripts/compare_core_memory.py`: equal-input payload comparison.
- `scripts/viz_semantickitti_fovea.py`, `scripts/kitti_vnc.sh`: current live demo.
- `src/main.py`: core becomes the default run implementation.
- `src/frame_pipeline.py`: historical hazard path requires explicit opt-in.
- `tests/test_core_tasks.py`, `tests/test_telemetry_and_weights.py`: regression checks.

Start: `./scripts/kitti_vnc.sh start 10 --mode model`.
Stop: `./scripts/kitti_vnc.sh stop`.
Browser: http://127.0.0.1:6081/vnc.html?autoconnect=true&resize=scale
