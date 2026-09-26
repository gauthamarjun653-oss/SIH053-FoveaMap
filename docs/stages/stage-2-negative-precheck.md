# Stage 2 — First check 2: negative-obstacle pre-check

**Status: RAN; SPOT-CHECK PENDING** (2026-08-27, on the Azure VM)

Purpose: derive negative-obstacle ground truth on sequence 00 and count
instances, deciding (spec 11.3) whether Module D gets a measured evaluation
(count >= 30), a reframed one (< 30), or a synthetic-only presentation (~0) —
before investing in ray-casting evaluation.

## Command

```
python -m src.main precheck-negatives --seq 00
```

## Data required

The validated stage-1 map (`runs/map_seq00.npy`). Present on the VM.

## Measured results

- Raw count: 197,828 instances (depth >= 0.15 m, extent <= 1 m) —
  numerically the ">= 30" branch ("proceed as specified; AC-D6 is a measured
  result"). Instances in `runs/neg_precheck_seq00.json` (gitignored).
- Phantom guard excluded 10,422 mirror-reflection cells (> 5 m apparent
  depth) — see stage 1.

## The raw count is NOT trustworthy as pothole ground truth

Do not score AC-D6 against it yet:

- 58% of instances are single 10-cm cells (per-cell min-z noise); median
  "depth" 0.34 m, p90 1.53 m — not pothole geometry.
- Only 39.9% of instance centroids lie on labeled road surface (the deriver
  runs over the whole map: vegetation, off-road terrain).
- Even the strictest cut — on-road, >= 4 cells, depth < 0.5 m — leaves
  11,068 instances. Dominant suspected confounder: loop-closure pose drift
  (sequence 00 revisits streets; ground layers misaligned by 10-20 cm read
  as depressions across whole road stretches).

## Owner decisions needed before AC-D6 scoring

1. The spec's mandated manual spot-check of >= 20 instances (visual, human).
2. Whether to rederive ground truth road-masked and drift-aware (per-cell
   robust ground instead of min-z, or per-revisit z-alignment) — a
   methodology change, not to be made unilaterally.
