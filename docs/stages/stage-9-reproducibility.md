# Stage 9 — Reproducibility pass

**Status: UNBLOCKED (stage 7 official run complete); one caveat to clear
first**

Purpose: prove every reported number traces to a provenance-stamped run
anyone can regenerate — clean-machine reproduction from the repo alone, with
metrics within 2% (PAC-7 / T-27, T-22).

## Caveat cleared (2026-08-28)

The 2026-08-27 run's `-dirty` provenance is superseded: the fp16 session
re-ran the full live benchmark from **clean commit `fe332e1`** on a T4.
That run is now the reference. (The 2026-08-27 numbers agree with it to
within 0.02 percentage points, which is itself reproducibility evidence.)

## Reference numbers to reproduce (within 2%)

From `runs/gpu-fp16/seq08_live/results.json` (commit `fe332e1`, live
inference, fp16, Tesla T4):

| Metric | Foveated | voxel_5cm | occupancy_2d |
|---|---|---|---|
| Recall 0-10 m | 0.9204 | 0.9237 | 0.8534 |
| Recall 10-25 m | 0.9713 | 0.9597 | 0.8584 |
| Recall 25-50 m | 0.9749 | 0.9603 | 0.8339 |
| Recall 50-100 m | 0.9659 | 0.9201 | 0.8552 |
| Mean memory / sweep (MB) | 3.30 | 1.31 | 0.54 |

Stage 5 reference: 26.2 ms median / 26.5 ms p90, fp16, Tesla T4 (fp32 was
50.6 / 51.4).

## Procedure

1. Fresh machine (any clean machine; a T4-class GPU instance for the live
   benchmark).
2. Clone the repo at the reference run's commit; follow README setup.
3. Stage data from S3 exactly as `infra/aws/bootstrap_gpu.sh` does
   (sequences 00 + 08, labels, calib, poses, weights) — the reproduction
   must need nothing that is not in the repo or the bucket.
4. Run the test suite (must be green with zero dataset), then
   `python -m src.main benchmark --seq 08 --mode live`.
5. Compare every metric above: PASS requires all within 2% and provenance
   showing a clean commit, live inference, and the T4-class hardware.

Determinism notes: recall and memory are deterministic given identical
weights, data, and code; small recall drift can come only from GPU
nondeterminism in inference (cuDNN autotuning) — if any metric lands
between 1% and 2%, rerun once to confirm stability before judging.
