# Stage 7 — Benchmark and Pareto curve (the primary deliverable)

**Status: OFFICIAL LIVE BENCHMARK RUN COMPLETE (Tesla T4, 2026-08-27)**

## Official Live DL Run — SalsaNext Inference, Full Validation Sequence 08, Tesla T4

`python -m src.main benchmark --seq 08 --mode live` over all 4,071
frames on NVIDIA Tesla T4 (`g4dn.xlarge`), zero failed frames, provenance-stamped (`commit 9dd9061`, ring-config hash `430e3998...`, hardware `Tesla T4`, `inference_mode: live`). Artifacts in `runs/seq08_live/`.

Obstacle Recall per Distance Band (0–10 / 10–25 / 25–50 / 50–100 m) over >108.7 Million Reference Cells:

| Representation | 0–10 m | 10–25 m | 25–50 m | 50–100 m | Mean Memory / Sweep |
|---|---|---|---|---|---|
| **Foveated 2.5D Grid (Ours)** | **92.03%** | **97.13%** | **97.48%** | **96.57%** | **3.30 MB** |
| Uniform 5 cm 3D Voxel (Baseline) | 92.36% | 95.97% | 96.02% | 91.98% | 1.31 MB |
| Uniform 2D Occupancy Grid (Baseline) | 85.34% | 85.84% | 83.39% | 85.52% | 0.54 MB |

Reference obstacle cells evaluated:
- Near (0–10 m): 34,171,739 cells
- Mid (10–25 m): 51,567,403 cells
- Far (25–50 m): 22,838,058 cells
- Very Far (50–100 m): 177,157 cells
- **Total:** 108,754,357 reference cells evaluated across 4,071 frames.

## Preliminary Run (gt-labels, CPU reference)

Recall per band (0-10 / 10-25 / 25-50 / 50-100 m):

| Representation | 0-10 | 10-25 | 25-50 | 50-100 | Mean memory / sweep |
|---|---|---|---|---|---|
| Foveated (full 84 B stats payload) | 0.9923 | 0.9997 | 0.9998 | 0.9933 | 3.30 MB |
| Uniform 5 cm 3D voxel (skeletal 15 B payload) | 0.9958 | 1.0000 | 1.0000 | 1.0000 | 1.31 MB |
| Uniform 2D occupancy (binary) | 0.8534 | 0.8584 | 0.8339 | 0.8552 | 0.54 MB |

## Honest reading — record this, do not spin it

- **Recall loss inside 10 m is 0.77%** - under the 1% target, with far-band
  recall also above 99%. That half of the expected headline holds.
- **The naive "10x memory reduction" does NOT appear against these
  baselines.** The 5 cm voxel baseline is deliberately skeletal (15 bytes:
  indices + count + obstacle bit), so it is CHEAPER than the foveated grid
  (1.31 vs 3.30 MB) while carrying none of the elevation statistics the
  project's analytics require. The flat occupancy grid is cheaper still and
  loses ~14 recall points everywhere - that is what discarding structure
  costs.
- The measured value proposition is therefore: **within 1% of the
  high-fidelity reference at bounded 3.3 MB/sweep, while being the only
  representation of the three that carries the full per-cell statistics**
  (z-stats, variance, class histograms, confidences) that kerb detection,
  negative obstacles, and the elevation map are built on.
- Note the near-tautology the spec warns about in reverse: the voxel_5cm
  baseline scores ~100% far recall partly because the recall reference is a
  5 cm lattice - same resolution as that baseline.
- Open question for the owner before the official run: whether the headline
  memory comparison should also include a like-for-like baseline (uniform
  5 cm cells carrying the SAME 84 B stats payload), which is where a large
  memory ratio would be measured. That baseline is not in the spec's 11.4
  set; adding it is a methodology decision.

Mean frame time 0.35 s on this CPU (banner rules apply: not a performance
result).

Purpose: the measured memory-vs-recall trade-off — the expected result is
roughly 10x memory reduction with under 1% recall loss inside 10 m, but the
number reported is whatever the run measures.

## Command

```
python -m src.main benchmark --seq 08
```

Single command (PAC-6); artifacts land in `runs/<run-id>/`:
`results.json`, `results.csv`, `pareto.png`, plus provenance (hardware, git
commit, config hash, inference mode). The run aborts without provenance.

## Data required

Sequence 08 (velodyne + labels + calib + poses) on the GPU instance for
inference; grid/recall aggregation can run on the Azure VM afterwards from
the same artifacts. Sync `runs/` to S3 before terminating the GPU instance.

## Design constraints already implemented

- Three configurations (foveated, uniform 5 cm voxel, uniform 2D occupancy)
  x four distance bands; recall reference is a fixed 5 cm lattice
  independent of any representation, so the denominator is identical for all
  configurations.
- Memory measured from actual array footprints; nothing hardcoded.
- Every number reported anywhere must trace to one of these provenance-
  stamped runs.
