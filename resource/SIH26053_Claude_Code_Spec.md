# SIH26053 — Implementation Specification

## Adaptive Variable-Resolution 2.5D LiDAR Mapping

**Status:** Ready to build. **Audience:** Claude Code.
**This document is the source of truth.** Where it conflicts with any earlier
document, this wins.

---

## 0. How to use this document

Build order is flexible, but **three checks must run before writing module code**,
because each determines something you would otherwise design around blindly.

| # | Check | Why it goes first | Time |
|---|---|---|---|
| 1 | Build the sequence-00 global map (§8.1) and assert z-extent 10–40 m | Catches the `inv(Tr)` bug immediately; unblocks all ground truth | ~1 h |
| 2 | Run the negative-obstacle pre-check (§8.3) and **count instances** | Determines whether Module D has a real evaluation or a synthetic one | ~10 min after (1) |
| 3 | Measure Module B latency alone on the demo GPU (§9.2) | Determines whether 10 FPS is reachable without FP16 | ~30 min |

After those three, build in any order. Only hard dependency: **D needs C**.

---

## 1. Objective

Convert raw LiDAR sweeps into a variable-resolution 2.5D semantic elevation map,
and produce a reproducible measurement of memory reduction against uniform-grid
baselines with no recall loss inside the safety radius.

**The primary deliverable is the Pareto curve, not the demo.** A working pipeline
with no measurement fails the submission; a modest pipeline with a rigorous
measurement does not.

---

## 2. Repository layout

```
sih26053/
├── config/
│   ├── rings.yaml            # ring table (§3)
│   └── pipeline.yaml         # projection dims, range gates, paths
├── sih26053/
│   ├── common/               # config loader, telemetry, errors
│   ├── io/                   # kitti_io: calib, poses, scans, labels
│   ├── projection/           # Module A
│   ├── segmentation/         # Module B
│   ├── grid/                 # Module C: lattice, rings, storage, aggregation
│   ├── stats/                # Module D: cell stats, ground plane, neg obstacles
│   ├── viz/                  # Module E
│   ├── offline/              # Module F: map, ground truth, baselines
│   └── benchmark/            # harness, Pareto, provenance
├── tests/
│   ├── property/             # AC-C1 and other invariants (Hypothesis)
│   ├── contract/             # module interface shapes
│   └── unit/
└── docker/
```

Dataset lives outside the repo at a configured path, read-only.

---

## 3. Ring configuration

**Use exactly this table. Do not substitute 25 cm or 50 cm.**

| Ring | Radius | Cell size | Nests into next | Fine cells per cell |
|---|---|---|---|---|
| R1 | 2.5 – 10 m | 5 cm | 10 ÷ 5 = 2 ✓ | 1 |
| R2 | 10 – 25 m | 10 cm | 20 ÷ 10 = 2 ✓ | 4 |
| R3 | 25 – 50 m | 20 cm | 40 ÷ 20 = 2 ✓ | 16 |
| R4 | 50 – 100 m | 40 cm | — | 64 |

Every size is a power-of-two multiple of every finer size, so AC-C1 holds for
**every ring pair**, not just base-to-ring. A 5/10/25/50 table fails at R2→R3
(25 ÷ 10 = 2.5) and must be rejected at startup.

`config/rings.yaml`:

```yaml
base_cell_size: 0.05
rings:
  - {id: 1, radius_min: 2.50, radius_max:  10.0, cell_size: 0.05}
  - {id: 2, radius_min: 10.0, radius_max:  25.0, cell_size: 0.10}
  - {id: 3, radius_min: 25.0, radius_max:  50.0, cell_size: 0.20}
  - {id: 4, radius_min: 50.0, radius_max: 100.0, cell_size: 0.40}
```

**Startup validation (must abort, not warn):**

```python
def validate_rings(rings, base):
    for r in rings:
        ratio = r.cell_size / base
        if abs(ratio - round(ratio)) > 1e-9 or (round(ratio) & (round(ratio)-1)) != 0:
            raise ValueError(
                f"Ring {r.id} cell {r.cell_size} is not a power-of-two multiple "
                f"of base {base}. AC-C1 cannot hold."
            )
    for a, b in zip(rings, rings[1:]):
        if abs(b.radius_min - a.radius_max) > 1e-9:
            raise ValueError(f"Rings {a.id}/{b.id} are not contiguous.")
```

**Ring assignment.** Boundary points go to the **inner (finer)** ring:

```python
# r = sqrt(x^2 + y^2), planar radius
ring = next((rg for rg in rings if r <= rg.radius_max), None)
# r <= radius_max, so r == 10.0 lands in R1, not R2
```

---

## 4. Shared lattice

All rings index into one lattice anchored at the sensor origin. **Do not divide
per ring independently** — independent division produces misaligned boundaries,
which is the alignment error Requirement 3 prohibits.

```python
def cell_index(x, y, cell_size):
    return (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))
```

Because every ring is anchored at 0 and sizes are power-of-two multiples, a
40 cm cell at index `k` contains exactly 5 cm cells `8k … 8k+7`. Nesting is
structural, not something to check at runtime.

Cell key: `(ring_id, cell_x, cell_y)`, unique.

---

## 5. Class taxonomy and the mapping trap

### 5.1 The three classes

| Class | Index | Meaning |
|---|---|---|
| IGNORE | 0 | Unlabeled / outlier |
| ROAD | 1 | Drivable ground |
| STATIC | 2 | Static obstacle |
| DYNAMIC | 3 | Vehicles, people |

### 5.2 The namespace trap — read this before writing Module B

Pretrained SalsaNext outputs **learning-space indices 0–19**, not raw
SemanticKITTI IDs. Both are small integers, so applying the raw-ID table directly
to an argmax output produces silently wrong labels that look plausible and never
raise.

The full chain is:

```
logits[20] → argmax → learning idx (0-19) → learning_map_inv → raw ID → 3-class
```

### 5.3 Collapse the chain into one lookup

Precompute once at load. Per-frame it becomes a single array index.

```python
import yaml, numpy as np

def build_learning_to_class3(yaml_path):
    cfg = yaml.safe_load(open(yaml_path))          # semantic-kitti.yaml
    inv = cfg["learning_map_inv"]                  # learning idx -> raw id

    ROAD    = {40, 44, 48, 49, 60, 72}
    STATIC  = {50, 51, 52, 70, 71, 80, 81, 99}
    DYNAMIC = {10, 11, 13, 15, 16, 18, 20, 30, 31, 32} | set(range(252, 260))

    table = np.zeros(20, dtype=np.uint8)
    for i in range(20):
        raw = inv[i]
        table[i] = 1 if raw in ROAD else 2 if raw in STATIC else 3 if raw in DYNAMIC else 0

    # Guard: this is the table the spec was written against.
    expected = np.array([0,3,3,3,3,3,3,3,3,1,1,1,1,2,2,2,2,1,2,2], dtype=np.uint8)
    assert np.array_equal(table, expected), (
        f"learning_map_inv produced {table.tolist()}, expected {expected.tolist()}. "
        "Do not proceed — inspect semantic-kitti.yaml before continuing."
    )
    return table

# per frame:
labels3 = LEARNING_TO_CLASS3[argmax_output]   # int8[H,W]
```

**Derive the table from the YAML and assert it against the literal.** The literal
alone risks a silent mismatch if the YAML differs; the YAML alone risks a silent
mismatch if it is misread. Together, either failure aborts loudly.

### 5.4 Two consequences to be aware of

**Raw IDs 252–259 are unreachable after this chain.** SemanticKITTI's
`learning_map` already merges moving classes into their static counterparts for
single-scan models (252 → learning 1 → raw 10). Those entries in the DYNAMIC set
never fire. Harmless — 10, 11, 13 … already cover them — but do not spend time
debugging why.

**Raw 52 (other-structure) and 99 (other-object) map to learning index 0**, so
they return as unlabeled and land in **IGNORE, not STATIC**. Both are rare. This
is an accepted trade, recorded here so it is not discovered later: a small number
of structure returns will be ignored rather than treated as obstacles.

---

## 6. Module A — Ingestion and Projection

**Signature:** `project(sweep_path, config) -> (RangeImage, PixelMap, Mask)`

| Item | Value |
|---|---|
| Input | `.bin`, flat float32 `(x, y, z, intensity)`, no header |
| Range image | `float32[64, 2048, 5]` — channels `[range, x, y, z, intensity]` |
| Pixel map | `int32[N, 2]` — one `(u, v)` per input point |
| Mask | `bool[64, 2048]` — pixel occupied |

**Confirm 64 × 2048 against the data before building on it.** It is the standard
HDL-64E geometry but has not been verified for this dataset copy.

### Behaviour

1. Reject if file size is not divisible by 16, reporting the byte count.
2. Range-gate to `[2.5, 100.0] m`. **Count discards; never drop silently.**
3. Spherical projection:
   ```
   r   = sqrt(x² + y² + z²)
   u   = floor(0.5 * W * (1 - atan2(y, x) / π))
   v   = floor(H * (1 - (asin(z / r) + |fov_down|) / fov_total))
   ```
4. On collision, **nearest return wins**; record every loser's index against the
   winning pixel. Emit a per-frame collision count.
5. Back-projection assigns the winning pixel's label to **all** points that
   mapped to it, winners and losers alike.

### Acceptance

- **AC-A1** Round-trip project→back-project, median positional error < 1 cm over 100 sampled sweeps.
- **AC-A2** 100% of input points receive a label. Assert `assigned + discarded == N`.
- **AC-A3** Deterministic: identical input → bit-identical output.

---

## 7. Module B — Semantic Segmentation

**Signature:** `segment(range_image, model) -> int8[H, W]`
**Then:** `back_project(labels, pixel_map, n_points) -> int8[N]`

### Behaviour

1. Load pretrained SalsaNext (19-class SemanticKITTI). **Do not train from scratch.**
2. `torch.load(..., weights_only=True)`. If the checkpoint is a whole model object rather than a state dict, convert it once offline and commit the state dict.
3. Missing or corrupt weights → **abort at startup.** Never fall back to random weights.
4. Normalise channels with training-set statistics.
5. Forward → `[H, W, 20]` logits → argmax → learning idx.
6. `LEARNING_TO_CLASS3[idx]` → 3-class labels (§5.3).
7. GPU unavailable → CPU inference with a **persistent banner stating FPS is not representative.**
8. NaN output → mark the frame failed, emit an empty grid, continue.

### Acceptance

- **AC-B1** Every point receives exactly one class index.
- **AC-B2** Class-mapping test: feed a scan with known `.label` ground truth; the collapsed prediction namespace matches the collapsed ground-truth namespace (labels need not agree, **namespaces must**).
- **AC-B3** Full pipeline completes end-to-end with a randomly initialised model — proving Module C does not depend on Module B's accuracy.
- **AC-B4** Per-class accuracy on sequence 08, **stratified by distance band**. Aggregate-only reporting is prohibited.

> Verify a pretrained SalsaNext checkpoint actually exists at
> `github.com/TiagoCortinhal/SalsaNext` before committing to this path. If it
> does not, train a minimal model on sequences 00/02/05 and **freeze it
> immediately** — accuracy is explicitly not the contribution.
---

## 8. Module C — Foveated Grid Engine *(the contribution)*

**Signature:** `build_grid(points, labels, ring_config) -> Grid`

### Storage

Sparse hash map keyed `(ring_id, cell_x, cell_y)`. **Only occupied cells are
allocated** — an empty scene must cost approximately zero. A dense array over
100 m at 5 cm is 4000 × 4000 cells before a single point arrives and defeats the
entire memory claim.

Prefer a structure-of-arrays layout over a dict of Python objects; per-object
overhead can dominate the measured footprint and undercut the very number you
are reporting.

### Accumulation — Welford, not naive variance

Each cell holds `(count, mean, M2)` plus a class histogram. **Variance is not
additive.** Averaging child variances is silently wrong and looks correct.

```python
def update(cell, z, cls):
    cell.count += 1
    delta = z - cell.mean
    cell.mean += delta / cell.count
    cell.M2 += delta * (z - cell.mean)
    cell.hist[cls] += 1
    cell.z_min = min(cell.z_min, z)
    cell.z_max = max(cell.z_max, z)

def combine(a, b):                       # parallel combination — use this
    n = a.count + b.count
    if n == 0: return zero_cell()
    delta = b.mean - a.mean
    return Cell(
        count = n,
        mean  = a.mean + delta * b.count / n,
        M2    = a.M2 + b.M2 + delta * delta * a.count * b.count / n,
        z_min = min(a.z_min, b.z_min),
        z_max = max(a.z_max, b.z_max),
        hist  = a.hist + b.hist,
    )

variance = cell.M2 / cell.count if cell.count > 1 else 0.0   # never NaN
```

### Aggregation topology

Canonical path is **base lattice → target ring**. Adjacent-ring rollup is
permitted only where §3's nesting makes it provably equivalent. One definition,
one thing to test.

### AC-C1 — the critical test

```python
from hypothesis import given, strategies as st

@given(points=random_point_clouds(), ring_pair=st.sampled_from(ALL_RING_PAIRS))
def test_lossless_aggregation(points, ring_pair):
    fine, coarse = ring_pair
    aggregated = aggregate(build_grid(points, ring=fine), to_ring=coarse)
    native     = build_grid(points, ring=coarse)
    assert cells_bit_identical(aggregated, native)
```

Generated sets must include degenerate cases: empty, single point, all points in
one cell, points exactly on ring boundaries, points at `ρ = 0`.

**This runs in CI and blocks merge.** It is the primary evidence for the
project's strongest claim; example-based tests prove considerably less than the
claim asserts.

### Acceptance

- **AC-C1** Lossless aggregation, every ring pair, bit-identical, in CI.
- **AC-C2** Empty scene → memory ≈ 0. Single point → exactly one cell.
- **AC-C3** Every in-range point assigned or explicitly counted as discarded.
- **AC-C4** Ring table changeable via config alone, no code edit.
- **AC-C5** Invalid config (e.g. 5/10/25/50) rejected at startup, naming the offending pair.
- **AC-C6** Variance combination test with hand-computed expected values.

### Failure behaviour

Memory ceiling exceeded → **abort loudly with the occupied cell count.** Never
silently reduce resolution: that reports a footprint for a grid you quietly
changed, which invalidates the measurement while appearing to succeed.

---

## 9. Module D — Cell Statistics and Negative Obstacles

**Signature:** `compute_stats(grid) -> Grid` (in place)

### Cell payload

| Field | Type | Purpose |
|---|---|---|
| `z_min` | float32 | Ground estimate |
| `z_max` | float32 | Overhead clearance |
| `z_mean` | float32 | Central tendency |
| `z_var` | float32 | **Load-bearing** — separates flat road from kerb when means match |
| `class_hist` | uint16[4] | **Counts, never a majority vote** |
| `return_count` | uint32 | Points contributing |
| `occupancy_conf` | float32 | Observed returns ÷ expected returns for that range and ring, clipped to [0,1] |
| `neg_obstacle_conf` | float32 | Candidate confidence, 0 if not flagged |

**Why the histogram and not a vote:** a cell with 99 road points and 1 pedestrian
point votes ROAD, and the pedestrian ceases to exist. In far-field rings, where
cells are large and a distant person yields a handful of returns, that is the
case that matters. This is a safety property, not a precision refinement.

**Why not max-height-only:** a pothole is *below* the ground plane. Storing only
the highest return mathematically fills it in and deletes the differentiator.

### Rendering rule

Render by **highest-severity class present above a threshold**, not by majority.
Otherwise the display discards the property the storage preserves.

### Negative obstacle inference

1. Estimate the local ground plane per region. On failure: skip that region and log. **Do not guess.**
2. Ray-cast: find cells where a ground intersection was expected but no return arrived.
3. **Rule out occlusion** by classified static or dynamic obstacles.
4. Emit unexplained cells as **candidates with confidence in [0,1]** — never as hard classifications.

A cell with `return_count == 0` is not automatically a negative obstacle.
Occlusion must be excluded first.

### Acceptance

- **AC-D1** No NaN in any field across a full sequence.
- **AC-D2** Variance separates flat road from kerb on ≥20 manually verified cells.
- **AC-D3** Synthetic-depression unit test: known geometry, known answer.
- **AC-D4** Safety test: cell with 99 ROAD + 1 DYNAMIC surfaces DYNAMIC in both payload and render.
- **AC-D5** Occlusion test: cell behind a classified vehicle is **not** flagged.
- **AC-D6** Precision/recall against §11 ground truth — **conditional, see §11.3.**

---

## 10. Recall definition

The headline metric. Without this, PAC-3 is untestable.

> Fix a **5 cm reference lattice**, independent of any representation.
> For distance band D:
> - **Ground-truth obstacle cells** = reference cells containing ≥1 point labelled STATIC or DYNAMIC in the SemanticKITTI ground truth.
> - A representation **detects** a reference cell if the cell (in that representation) containing it is flagged obstacle-occupied.
> - **Recall(R, D) = detected / total ground-truth cells in D.**

Bands: 0–10, 10–25, 25–50, 50–100 m.

**Why this shape.** The denominator is the same reference lattice for all three
configurations, so the numbers are comparable. Counting a representation's own
cells would give each configuration a different denominator and produce a
meaningless comparison — and would be near-tautological besides, since a cell
exists precisely because points landed in it.

Expected behaviour, useful as a sanity check:

| Configuration | Expected recall |
|---|---|
| Uniform 5 cm voxel | ~100% — it *is* the reference lattice; upper bound |
| Uniform 2D occupancy | Poor — height discarded, obstacle indistinguishable from ground |
| Foveated 2.5D | ~100% in R1, degrading outward |

**PAC-3 tolerance:** < 1% relative recall drop inside 0–10 m versus the uniform
5 cm voxel baseline.

---

## 11. Module F — Offline Toolchain and Benchmark

Runs **once**, offline. **Not part of the deployed system.** Keep this separation
visible in every diagram and in the report — blurring it invites the reading that
your latency figures come from cached results.

### 11.1 Global map

```python
Tr   = load_calib(...)                       # 4x4, velodyne -> camera
T_i  = inv(Tr) @ pose_i @ Tr                 # OUTER INVERSE IS MANDATORY
world = points_velo @ T_i[:3,:3].T + T_i[:3,3]
```

`poses.txt` gives the pose of the **left camera**, not the LiDAR. Omitting the
outer `inv(Tr)` yields a map that registers correctly and looks entirely
plausible — but sits in the camera convention with **y pointing down**, silently
corrupting every elevation statistic downstream. Since this project is
elevation statistics, that is the highest-consequence low-visibility bug in the
codebase.

Also: exclude dynamic-class points (otherwise moving cars smear into comet
trails), and voxel-deduplicate incrementally rather than stacking then
downsampling once — 4541 × 120k points is ~6.5 GB before downsampling.

**Assertions, run before anything is derived from the map:**

- **AC-F1** z-extent within 10–40 m for an urban sequence. *This is what catches the `inv(Tr)` bug.*
- **AC-F2** Building façade thickness < 15 cm (wall-normal standard deviation).

### 11.2 Layout note

The calib archive ships poses at a top-level `poses/XX.txt`, **not** at
`sequences/XX/poses.txt` where every module expects them. Move them, then verify:

```
scan count == pose count == label count      # abort naming all three on mismatch
```

A truncated extract is common and fails silently otherwise.

### 11.3 Negative-obstacle ground truth — run the pre-check first

**Before writing Module D's ray casting**, derive ground truth on sequence 00 and
**count instances.**

Initial threshold: depressions ≥ 15 cm deep over ≤ 1 m horizontal.

| Instance count | Action |
|---|---|
| **≥ 30** | Proceed as specified. AC-D6 is a measured result. |
| **< 30** | Reframe to negative obstacles generally: kerb drop-offs, road-edge falls, gutter depressions. Threshold ~8 cm, drop the horizontal constraint. |
| **≈ 0** | Present the capability as designed and unit-tested against synthetic geometry, **not** claimed as measured. Honest and still creditable. |

**Why this check exists.** KITTI was recorded on well-maintained German municipal
roads. There may be almost no potholes meeting the initial threshold. Precision
and recall over four samples means nothing, and you would be evaluating your
headline differentiator against a near-empty set.

**Naming discipline if the reframe triggers:** describe the targets as *road-edge
elevation discontinuities*, not potholes. A perception specialist will notice if
pothole detection is demonstrated on kerbs, and the precise term costs nothing.

Output per instance: centroid, dimensions, depth, confidence. Spot-check ≥20
manually before using it to score anything.

### 11.4 Baselines

| Baseline | Definition |
|---|---|
| Uniform 5 cm 3D voxel | High-fidelity, high-cost reference. **Measure its memory; do not hardcode a figure.** |
| Uniform 2D occupancy | Cheap, height-free reference |

### 11.5 Pareto curve

X-axis memory (MB), Y-axis recall, stratified by distance band, three
configurations. **Regenerates from a single command with no manual steps.**

### 11.6 Provenance — required, not optional

Every run records git commit hash, ring-config hash, hardware spec, dataset
split, and whether inference was live or cached. **The run aborts without them.**
Optional metadata is omitted under deadline pressure, which is exactly when it
matters.

The benchmark harness must **refuse to run against a cached inference path.**
---

## 12. Module E — Visualisation and Telemetry

**Blocked on one verification.** Before building anything here, prototype
rerun.io rendering a static variable-resolution grid with ring boundary overlays.
It takes about a day. If it comes back negative, fall back to Open3D with reduced
ambition. Building Module E on an unverified rendering assumption is the most
expensive avoidable mistake available in this project.

### Screens

**S-1 Sequence selector** — list sequences with frame count and label
availability. A sequence lacking poses is **disabled with a stated reason**, not
hidden. Empty state names the configured path.

**S-2 Main viewer** — the demonstration surface.

- Class colours: distinct, fixed, legend on screen. **Not colour alone** — pair with pattern or label.
- Ring boundary overlay, toggleable.
- Elevation encoding: colour ramp (clearer than extrusion for this data).
- Negative-obstacle overlay with a confidence threshold slider.
- Live memory (MB) and FPS (rolling 30-frame average), **visible without interaction**.
- **Comparison by toggle, not split-screen.** Split-screen only if the rerun spike shows synchronised views are cheap.
- **Switching comparison mode must preserve the frame index.** If the scene changes when the mode changes, the comparison proves nothing.

**S-3 Benchmark results** — Pareto plot, per-band metrics table, run metadata
including hardware, CSV/JSON/PNG export. Loading state names the current stage;
benchmarks take minutes.

### States

| State | Behaviour |
|---|---|
| Empty sweep | "No returns in range" on canvas — not a blank area |
| Frame failure | Timeline marker; playback continues |
| CPU fallback | Persistent banner: FPS not representative |
| **Cached mode** | **Persistent banner. Mandatory.** |

**AC-E5 — the cached-mode banner is a functional requirement.** Its silent
failure is a defect, because it would let cached-throughput figures be presented
as live perception throughput.

### Acceptance

- **AC-E1** ≥10 FPS sustained on the demo machine with live inference.
- **AC-E2** All classes distinguishable — verified by someone outside the team.
- **AC-E3** Ring boundaries visible and toggleable.
- **AC-E4** Memory and FPS visible without interaction.
- **AC-E5** Cached-mode banner asserted present whenever cache is active.
- **AC-E6** Frame index preserved across comparison-mode switch.

---

## 13. Hardware and performance

| Component | Spec |
|---|---|
| Demo machine | RTX 3050 Laptop, 6 GB |
| OS | Ubuntu 22.04 |
| CUDA | 12.x |
| PyTorch | 2.0+ |

**Every latency and FPS figure must cite this machine.** A number without its
hardware is not a measurement.

### Latency budget — inferred, not yet validated

| Stage | Budget |
|---|---|
| A projection | ≤ 15 ms |
| B inference | ≤ 40 ms |
| C grid build | ≤ 25 ms |
| D statistics | ≤ 15 ms |
| **Total** | **≤ 100 ms (10 FPS)** |

**10 FPS on a 6 GB mobile GPU is not a given.** SalsaNext's published throughput
is on desktop-class hardware. Measure Module B in isolation in week one. If it
exceeds 40 ms, in order of preference: FP16 inference, narrower range image
(1024 instead of 2048), then network pruning. Finding this out in the final week
is expensive; finding it out on day three is a config change.

---

## 14. Acceptance criteria — the "done" list

| ID | Criterion |
|---|---|
| PAC-1 | Full pipeline processes a complete sequence without manual intervention |
| PAC-2 | Memory footprint ~10× lower than uniform 5 cm voxel — **measured, on stated hardware** |
| PAC-3 | Obstacle recall inside 0–10 m shows < 1% relative loss vs uniform baseline (§10) |
| PAC-4 | Sustained ≥ 10 FPS with **live** inference on the demo machine |
| PAC-5 | AC-C1 lossless aggregation passes in CI on every commit |
| PAC-6 | Pareto curve regenerates from a single command |
| PAC-7 | Third party reproduces metrics within 2% from the repository alone |
| **PAC-8** | **Negative obstacles evaluated. Conditional on §11.3:** if ≥30 ground-truth instances, report precision/recall with sample size. Otherwise report as unit-tested against synthetic geometry and state so explicitly. |

PAC-8 exists because the pre-check must have an outcome that lands somewhere in
the done list. Without it, the differentiator has no completion condition either
way.

---

## 15. Test list

| ID | Type | Scenario | Expected |
|---|---|---|---|
| T-01 | Unit | `.bin` size not divisible by 16 | Rejected, byte count reported |
| T-02 | Property | Project → back-project, 100 sweeps | Median error < 1 cm |
| T-03 | Unit | Pixel collision | Nearest wins; loser still labelled |
| T-04 | **Unit** | **`build_learning_to_class3` vs literal** | **Assertion passes, or abort** |
| T-05 | Integration | Known-label scan through Module B | Namespaces match ground-truth collapse |
| T-06 | Integration | Randomly initialised model | Pipeline completes end to end |
| T-07 | Failure | Weights absent | Startup abort; no random fallback |
| T-08 | **Property** | **AC-C1, every ring pair, random point sets** | **Bit-identical** |
| T-09 | Unit | Config 5/10/25/50 | Rejected at startup, naming R2/R3 |
| T-10 | Unit | Welford combine, two known groups | Matches union variance exactly |
| T-11 | Edge | Point exactly at r = 10.0 | Assigned to R1 |
| T-12 | Performance | Empty scene | Memory ≈ 0 |
| T-13 | Edge | Single point in sweep | Exactly one cell, variance 0, no NaN |
| T-14 | **Safety** | **99 ROAD + 1 DYNAMIC in one cell** | **DYNAMIC surfaces in payload and render** |
| T-15 | Unit | Cell occluded by classified vehicle | Not flagged negative |
| T-16 | Unit | Synthetic depression, known geometry | Detected, confidence in [0,1] |
| T-17 | **Integrity** | **Global map built** | **z-extent 10–40 m** |
| T-18 | Integrity | Wall thickness in map | < 15 cm |
| T-19 | Unit | Scan/pose/label count mismatch | Abort naming all three |
| T-20 | Unit | Benchmark run without provenance | Aborts, not warns |
| T-21 | Unit | Benchmark against cache | Refused |
| T-22 | Regression | Benchmark run twice | Within 2% |
| T-23 | Failure | Injected NaN frame | Marked; playback continues |
| T-24 | **Failure** | **Memory ceiling exceeded** | **Aborts loudly; no silent degradation** |
| T-25 | Automated | Cached inference active | Banner present |
| T-26 | Integration | Comparison mode switch | Frame index preserved |
| T-27 | System | Clean machine, repo only | Metrics within 2% |
| T-28 | Security | `weights_only=True` load | Succeeds; state-dict format confirmed |

---

## 16. What NOT to build

| Item | Why |
|---|---|
| Authentication, user accounts, RBAC | No users |
| Database, ORM, migrations | Filesystem only; grids are per-sweep and discarded |
| HTTP API, FastAPI, REST layer | In-process function calls; a network boundary costs latency for no gain |
| React/WebGL frontend | rerun.io or Open3D; a web stack is a whole workstream |
| Cloud deployment, CI/CD to a server | The demo runs locally |
| Message queues, background jobs, caching layer | Synchronous per-sweep chain |
| Sparse convolutional backbone | Future work; CUDA toolchain risk |
| Temporal fusion across sweeps | Future work |
| **Hardcoded memory figures** | **Measure. Never assert.** |
| Split-screen rendering by default | Toggle first; split-screen only if the spike shows it is cheap |

---

## 17. Data

| Sequence | Frames | Size | Role |
|---|---|---|---|
| 00 | 4,541 | 8.3 GB | Demo + train. Start here — covers projection, grid engine, map, AC-C1 |
| 02 | 4,661 | 8.6 GB | Train |
| 05 | 2,761 | 5.1 GB | Train |
| 08 | 4,071 | 7.5 GB | **Validate + benchmark. Standard split — do not substitute** |

Plus: calib archive (~1 MB — contains `Tr` and poses) and SemanticKITTI labels
(~180 MB). **Poses are ~3.6 MB of text, not 180 MB** — if your poses directory is
180 MB you have two copies of the labels archive.

Sequences 01 and 11–21 are not used. 01 is highway: sparse, featureless, adds
training difficulty without adding anything a 3-class problem needs. 11–21 have
neither poses nor labels.

Layout:

```
/data/kitti/sequences/00/
    calib.txt
    poses.txt          # moved from the archive's top-level poses/00.txt
    velodyne/000000.bin
    labels/000000.label
```

---

## 18. Open items — record the decision, do not leave them implicit

| # | Item | Recommendation | When |
|---|---|---|---|
| 1 | Range image dimensions | Confirm 64 × 2048 against the data | Before Module A |
| 2 | rerun.io capability | Spike it | Before Module E |
| 3 | Pretrained SalsaNext checkpoint availability | Verify it exists; fall back to a minimal trained model, frozen | Before Module B |
| 4 | Negative-obstacle instance count | Run the §11.3 pre-check | Before Module D |
| 5 | Module B latency on the demo GPU | Measure in isolation | Week one |
| 6 | Slope vs step disambiguation | Both produce high variance. If unresolved, state it in the report rather than papering over it | Before the report |

---

## 19. Principles, when a decision is not covered here

1. **Nothing degrades silently.** Failures abort loudly or are marked visibly. A system that quietly changes behaviour under pressure reports numbers for something it is no longer doing.
2. **The measurement is the deliverable.** When effort must be cut, cut features before evidence.
3. **Stage B accuracy is not the contribution.** The pipeline must run end to end with an untrained model. Effort spent tuning segmentation is effort stolen from the grid engine.
4. **Measure, never assert.** Any number in the report, the deck, or a diagram must trace to a run with recorded provenance.
5. **Report stratified by distance band.** Aggregate accuracy conceals precisely the distance-dependent behaviour this problem is about.

---

## 20. First session checklist

- [ ] Repository scaffold, config loader, ring validation (**T-09 passes**)
- [ ] `kitti_io`: calib, poses, scans, labels; verify counts match (**T-19**)
- [ ] Global map for sequence 00; **assert z-extent (T-17)**
- [ ] Negative-obstacle pre-check; **count instances, record the number**
- [ ] Module B latency measured in isolation on the demo GPU
- [ ] `build_learning_to_class3` with the literal assertion (**T-04**)
- [ ] rerun.io spike started

Nothing on this list depends on the full dataset. All of it can proceed while
extraction runs.
