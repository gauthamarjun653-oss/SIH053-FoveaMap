# Current implementation notice — 2026-09-23

The review corrections in [corrections-2026-09-23.md](corrections-2026-09-23.md)
supersede conflicting implementation claims below. In particular, current-scan
nearest returns no longer define BLIND; support is a per-plane convex hull;
occupancy uses exact distances and compatible heights; unconverged rays are
rejected. The numerical results below are historical, not measurements of the
corrected implementation. Accuracy still requires independently annotated scenes.

---

# Ground support and visibility — correctness definition

**Status:** implemented. §1-§7 are the specification; §8 records what building
it changed, and supersedes the earlier text where they disagree.
**Purpose:** define what the system is allowed to *claim* about ground height and
about unobserved space, before the negative-obstacle detector is rewritten. The
detector depends on both; rewriting it first would optimise a rule whose inputs
are undefined.

**Decision this document forces:** a missing LiDAR return is currently treated as
evidence of a depression. It is not. This document defines the three cases that
must be separated before any such claim is made, and what must be measured to
show the separation works.

---

## 1. What the code does today (measured, not asserted)

20 frames spread across sequences 09 and 10, ground-truth labels, default config.
Counters come from `negative_obstacle_candidates(stats_out=...)`.

| Decision | Count | Share |
|---|---:|---:|
| `no_plane` — site visited, no ground estimate | 2,300,995 | 43.7% |
| `missing_flagged` — emitted as candidate | 1,741,007 | 33.1% |
| `occlusion_suppressed` — ruled out by line of sight | 1,210,538 | 23.0% |
| `below_plane` — occupied cell below plane | 7,044 | 0.1% |

**87,403 candidates per frame**, of which 99.6% are `missing_return`. Median
confidence of a `missing_return` candidate is 0.440 — when 1.74M sites share a
score, that score ranks nothing. Ground planes fit for a median of 66 tiles per
frame; 43.7% of visited sites had no plane at all.

Occlusion reasoning is doing real work — it suppresses 41.0% of missing-return
sites — so the existing `_OcclusionIndex` is worth keeping. The defect is not
that occlusion is ignored. It is that everything *else* is.

---

## 2. The core defect: unsampled ground is indistinguishable from missing ground

A Velodyne HDL-64E has 64 beams over a 28° vertical span — 0.444° apart. On flat
ground at sensor height 1.73 m, consecutive beams strike the ground at radii that
diverge quadratically with range:

| Range | Beam pitch | Next beam lands at | Ground gap | Cell size | Cells between beam rings |
|---:|---:|---:|---:|---:|---:|
| 5 m | 19.09° | 5.1 m | 0.13 m | 0.05 m | 2.6 |
| 10 m | 9.82° | 10.5 m | 0.48 m | 0.05 m | 9.7 |
| 15 m | 6.58° | 16.1 m | 1.10 m | 0.10 m | 11.0 |
| 25 m | 3.96° | 28.2 m | 3.17 m | 0.10 m | 31.7 |
| 30 m | 3.30° | 34.7 m | 4.68 m | 0.20 m | 23.4 |

At 25 m, **31 consecutive cell rows between two beam rings can never receive a
return** on perfectly flat, unobstructed road. The current scan flags every one
of them. This — not occlusion, not thresholds — is the dominant source of the
1.74M candidates.

The code states the opposite assumption explicitly:

```python
# src/analytics/cell_stats.py:23
def expected_returns(r, cell_size, azimuth_res_rad):
    e = cell_size / (np.maximum(r, 1e-6) * azimuth_res_rad)
    return np.maximum(e, 1.0)          # <- asserts every cell expects >= 1 return
```

Two faults in four lines. The model counts **azimuthal** samples only and ignores
**elevation** sampling entirely, which is the axis the gap lives on. Then the
clamp to `1.0` makes it structurally impossible to express "this cell was never
sampled". Unsampled ground cannot be represented, so it is reported as missing.

A corrected expectation is the product of both axes:

```
E(cell) = n_azimuth(cell) * n_elevation(cell)
n_azimuth   = cell_size / (r * azimuth_res)          # existing term
n_elevation = radial_extent(cell) / beam_ring_gap(r) # missing term, NOT clamped
```

At 30 m with a 0.20 m cell, `n_elevation ≈ 0.20 / 4.68 ≈ 0.043`. The honest
statement is "this cell expected 0.04 returns", and no absence claim may be built
on it.

---

## 3. Definitions

### 3.1 Ground support

Every location carries exactly one ground-support state.

- **SUPPORTED** — a plane was fitted from at least `min_cells` ROAD-bearing cells
  with RMS residual ≤ `max_rms`, **and the location lies inside the planar convex
  hull of the cells that produced the fit**.
- **EXTRAPOLATED** — a plane exists for the tile, but the location lies outside
  that hull. The estimate is a projection, not a measurement.
- **UNSUPPORTED** — no plane, or the fit failed its residual test. No ground
  height may be claimed and no expectation may be formed.

The hull condition is new and it fixes a live defect. `fit_ground_planes` fits one
plane per 5 m × 5 m tile and applies it to *every* location in that tile. A tile
whose 8 qualifying ROAD cells sit in one corner yields a plane applied across the
full 25 m². Today that is indistinguishable from a plane fitted from cells spread
across the whole tile. Extrapolated ground beneath a curb, a driveway lip or a
kerbside drop reads as a depression.

**Rule: only SUPPORTED ground may generate a negative-obstacle candidate.**
EXTRAPOLATED ground may be displayed, marked as such. UNSUPPORTED stays unknown —
per the PDF, *unknown ground stays unknown, and sidewalk does not automatically
mean drivable*.

### 3.2 Visibility

Every location carries exactly one visibility state. The first three exist in the
code today; **UNSAMPLED does not, and is the substance of this change.**

- **OBSERVED** — a return is present within the radial tolerance of §8.2.
- **OCCLUDED** — line of sight from the sensor origin is blocked by a cell
  classified STATIC or DYNAMIC before reaching the expected ground intersection.
  Absence is explained; no claim. (Current `_OcclusionIndex`, retained.)
- **UNSAMPLED** — `E(cell) < E_min`: sensor geometry predicts fewer than the
  minimum number of beam-ground intersections. Absence carries no information.
  **No candidate may be emitted.**
- **BLIND** — nearer than anything the sensor has ever returned along that
  azimuth: the ego vehicle's own shadow. See §8.1. No claim.
- **EXPECTED-EMPTY** — `E(cell) ≥ E_min`, line of sight clear, ground SUPPORTED,
  and no return arrived. **This is the only state in which absence is evidence.**

### 3.3 Cell ownership

Exactly one ring owns each location. The convention must be stated once and used
by every consumer.

There is a live inconsistency. `assign_rings` places *points* by point radius, with
boundary radii going to the inner, finer ring. The missing-return scan walks each
ring's lattice and tests the *cell-centre* radius. Near a ring boundary the two
disagree: a point at r = 24.99 m is stored in ring 2 (0.10 m cells), while the
ring-3 lattice cell covering it has a centre at r > 25 m and is scanned as ring 3.
That cell is absent from ring 3's occupied set, so it is flagged as a missing
return — a false alarm generated purely by the boundary convention, in a ring of
concentric false alarms at every transition radius.

**Rule: ownership is determined by cell centre radius, with boundaries resolving
inward, and point-to-cell assignment must use the owning cell's ring.** Points and
cells must not use different conventions. A test must assert that no candidate is
ever emitted for a location whose points are stored in a different ring.

---

## 4. Emission rule

A `missing_return` candidate may be emitted **only** when all hold:

1. ground state is SUPPORTED (§3.1);
2. visibility state is EXPECTED-EMPTY (§3.2);
3. the cell is owned by the ring being scanned (§3.3);
4. the expected ground intersection lies below the plane by ≥ `depth_m`, given
   the vertical resolution actually available at that range.

Confidence must be a function of evidence strength — expected returns, ground fit
quality, agreement across neighbouring cells — not of range alone. Candidates
remain confidences in [0, 1]; the existing refusal to emit hard classifications is
correct and stays.

The two kinds are range-bounded separately, because they rest on different
evidence. `missing_return` rests on *absence*, and past ~25 m the beam-ring gap
exceeds the feature size (§2), so absence there says nothing: bounded at
**25 m**. `below_plane` rests on *measured returns* sitting below the plane —
direct evidence that degrades only as the plane's fitting error grows against
the 0.15 m being claimed, which is slower: bounded at **30 m**. One shared
radius was tried first; it cost below_plane 5-8% of its candidates to remove
1.5-1.9% of missing-return ones, the wrong trade. Cells are tested at their
centre radius, per §3.3. Range histograms are reported per kind so the
different caps cannot mislead.

Otherwise `below_plane` (occupied cells with floor below the plane) is the
stronger signal and is barely affected: 7,044 across 20 frames, ~352 per frame.
They should be scored separately, never pooled with missing-return candidates into
one recall number.

---

## 5. What must be measured

Recall alone cannot validate this. Suppressing 1.74M false candidates will not
move recall if the true positives survive — which is the point — so the metric
must be able to show the suppression.

Report, per sequence, split by `below_plane` / `missing_return` and by distance band:

- **missed hazards** — hand-labelled depressions with no candidate;
- **false alarms per frame** — candidates on ground hand-confirmed as flat;
- **candidates per frame** — the headline; today 87,403;
- **state distribution** — share of sites in each of the four visibility states,
  and the three ground states. This shows *why* a candidate was or was not
  emitted, and it is the diagnostic that is missing today;
- **latency** — median, p95, worst (per [scripts/profile_stages.py](../scripts/profile_stages.py)).

### Hand-checked example set

Required before the detector is rewritten, so the rewrite has a target. Drawn from
sequences 09 and 10, each frame and cell recorded by index:

- **Positive, returns present** — real depressions that produced points below the
  local plane. Must be detected.
- **Positive, no returns** — real depressions that swallowed the beam. The hard
  case; the honest outcome may be "not detectable at this range", which is a valid
  result to record.
- **Negative, unsampled** — flat road between beam rings beyond 20 m. Must be
  UNSAMPLED, never a candidate. This class alone should account for most of the
  1.74M.
- **Negative, occluded** — flat road shadowed by a vehicle. Must be OCCLUDED.
- **Negative, extrapolated ground** — flat ground near a kerb or driveway where
  the tile plane is fitted from cells on one side only. Must be EXTRAPOLATED and
  emit nothing.
- **Negative, ring boundary** — flat road within 0.5 m of a ring transition
  radius. Must emit nothing.

Sequence 09 and 10 are the only labelled sequences held locally, and the
segmentation model saw some of 00–10 during training. Ground truth for this set is
manual inspection of geometry, not model output, and the set must be frozen before
thresholds are tuned against it.

---

## 6. Open decisions — these need a human call

1. **`E_min`.** How many expected beam-ground intersections before absence counts
   as evidence? 1.0 is defensible and conservative; 0.5 admits more range at more
   risk. Calibrate against the hand-checked set, then freeze.
2. **Effective detection range.** The table in §2 implies missing-return detection
   is near-worthless past ~25 m for a 0.15 m depression, because beam spacing
   exceeds the feature size. `max_range` is 30 m today. Lowering it to what the
   sensor can actually resolve is honest and cheap; it should be a stated,
   measured limit rather than an unadvertised failure region.
3. **Flat-ground assumption.** The beam-gap model in §2 assumes a locally flat
   surface at known sensor height. On a gradient the gaps change. Either derive
   the gap from the fitted plane's own slope, or state the flat-ground assumption
   as a limitation. Deriving it is not much harder and is the better answer.
4. **Sensor height provenance.** 1.73 m is the KITTI datasheet figure. It should
   be read from calibration rather than hard-coded, or asserted loudly against
   calibration at startup.
5. **Does `below_plane` need the hull condition?** It is the high-value signal and
   the hull rule will suppress some of it near kerbs — exactly where real
   depressions cluster. Measure both ways on the frozen set before deciding.

---

## 7. Why this ordering

The detector cannot be rewritten before these definitions exist, because every
term in its decision rule — expected ground height, expected return count, cell
ownership — is either undefined or defined wrongly today. Fixing the loop
structure first would make a wrong answer arrive faster: `negative_obs` costs
990 ms median and 1,325 ms worst, but a large share of that work is spent
enumerating sites that §3.2 says should never have been visited. Geometry removes
most of the cost before any optimisation is attempted.


---

## 8. What implementation changed

Three things were wrong or missing in §1-§7. They were found by building it and
are now part of the definition.

### 8.1 A fifth visibility state: BLIND

Between 2.5 m and 3.5 m a KITTI scan carries **6 returns spanning ±11° of
azimuth**. The roof-mounted sensor cannot see the ground beside its own vehicle.
This is self-occlusion, but `_OcclusionIndex` cannot represent it: that index
reasons from cells classified STATIC or DYNAMIC, and the ego vehicle returns no
points to classify. Left unhandled it produced the largest false clusters in the
frame.

BLIND is measured per azimuth as the nearest range ever returned in that
direction, so it adapts to sensor and mounting instead of hard-coding a vehicle
footprint. It accounts for ~7,100 rays per frame.

### 8.2 Predicted landing points need a radial tolerance, not an exact cell

A ground intersection's radius responds to height error with gain **`dr/dz = r/h`**.
At 25 m with a 1.7 m sensor, a 3 cm plane residual — ordinary road roughness —
displaces the predicted landing point by **44 cm**, several cells. Testing the
exact predicted cell therefore manufactures false alarms out of roughness.

Occupancy is tested within `tol = max(cell_size, rms_residual · r / h)`.
Applying this moved 45,583 rays per frame out of EXPECTED-EMPTY and into
OBSERVED. Dilation is isotropic while the true error is radial, which
over-suppresses slightly — the correct direction for false alarms.

This also means the two-pass tile lookup of §3.1 is not enough: at range the
first estimate can land a tile away and the correction land in a third. The
intersection is iterated to a fixed point, and a ray contributes evidence only
if it settles in the tile whose plane produced it.

### 8.3 Clusters need a shape gate, not just a size gate

8-connectivity will chain a sparse scatter of isolated misses into one huge
"candidate". Untuned, the two largest clusters in a frame spanned 7 m at fill
ratios of **0.081 and 0.057**, against 0.25-0.30 for plausible ones. Clusters are
now bounded by `max_extent_m` (1.0 m, from `configs/pipeline.yaml`) and required
to reach `min_fill_ratio` of their bounding box.

### 8.4 Open decisions, resolved

| # | Decision | Outcome |
|---|---|---|
| 1 | `E_min` | Moot. Rays are counted exactly per cell, so the threshold is simply ≥1 ray. |
| 2 | Effective range | **Split per kind: 25 m for missing-return, 30 m for below-plane.** A shared 25 m cap was tried first and removed 1.5-1.9% of missing-return candidates (geometrically marginal — §2 shows 31 unsampled cell rows between beam rings there) but 4.9-8.1% of below-plane ones, which are measured evidence and whose 20-25 m band is the single largest bucket on seq 10. Restoring 30 m for below-plane recovers those 26 + 43 candidates at no latency cost: seq 10 median 78.6 ms vs 77.1 ms shared. A near limit also exists (§8.1). Revisit only against the frozen example set. |
| 3 | Flat-ground assumption | Eliminated. Rays intersect the *fitted* plane, so slope is handled directly. |
| 4 | Sensor height | Measured, not assumed: the plane at the origin gives 1.680 m (seq 09) and 1.737 m (seq 10) against KITTI's 1.73 m datasheet. |
| 5 | Hull rule on `below_plane` | Still open by design — `require_supported` is a flag, measured both ways. |

### 8.5 Measured outcome

24 frames spread across sequences 09 and 10, ground-truth labels.

| | old | new |
|---|---:|---:|
| Candidates per frame | 95,672 | **194** |
| Latency median | 930 ms | **92 ms** |
| Latency p95 / worst | 1,045 / 1,057 ms | 111 / 112 ms |

**Recall is still not scored.** `scripts/extract_examples.py` freezes a 419-example
set under a seed, with empty `verdict` fields; until a person fills them in, the
claim supported by evidence is that the detector is cheaper and better justified,
**not** that it is more accurate. 152 missing-return candidates per frame remains
too many to be real depressions, so a material false-alarm rate is still expected.
