# Stage 3 — Random-model end-to-end (PAC-1 dry run)

**Status: DONE** (2026-08-27, on the Azure VM)

Purpose: process a complete sequence without manual intervention using a
RANDOM-initialized SalsaNext (AC-B3 / T-06). This proves the grid engine —
the contribution — is independent of segmentation accuracy. Random mode
exists as an explicit flag only; the pipeline never falls back to random
weights silently.

## Command

```
python -m src.main run --seq 00 --mode random
```

## Data required

Sequence 00 velodyne + calib + poses locally. No labels needed in random
mode (labels are only read in `--mode gt-labels`); no weights needed
(random init is explicit).

## Measured results

- All 4,541 frames processed with no manual intervention; failed frames:
  none; exit code 0. This is the PAC-1 / AC-B3 evidence.
- Per-sweep grid stayed at roughly 36k-53k cells and 3.0-4.4 MB throughout —
  far under the 512 MB ceiling; discards and projection collisions were
  counted on every reported frame (never silent).
- Throughput was about 1.0 FPS on this CPU-only VM. The "CPU INFERENCE — FPS
  is not representative of the demo GPU" banner was present on every status
  line, exactly as required; this number is NOT a performance result.
