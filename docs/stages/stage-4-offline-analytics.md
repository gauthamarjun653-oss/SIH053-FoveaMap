# Stage 4 — Offline analytics on real data

**Status: MEASURED (2026-08-27); AC-D2 manual spot-check pending (human)**

Purpose: validate Module D's load-bearing statistics against real scans, not
just synthetic tests.

## Commands

```
python -m src.main run --seq 00 --mode gt-labels    # AC-D1 over the full sequence
python -m src.main analytics-d2 --seq 00 --stride 20 # AC-D2 + occlusion counters
```

Artifact: `runs/stage4_analytics_seq00.json` (gitignored, on the VM).

## Measured results

### AC-D1 — no NaN in any cell field across a full real sequence: PASS

Full gt-labels pass over all 4,541 frames of sequence 00 with the NaN
assertions armed in `compute_stats`: zero failures, zero assertion trips.

### AC-D2 — variance separates flat road from kerb: MEASURED, decisive

228 frames sampled (stride 20), ring-1 (5 cm) cells, minimum 4 points per
cell. Road-interior cells contain only raw road points; kerb cells contain
both raw road and raw sidewalk points (the physical step lives at that
boundary).

| Group | Cells | z-variance p10 | p50 | p90 |
|---|---|---|---|---|
| Road interior | 160,297 | 1.2e-6 | 4.7e-6 | 1.5e-5 |
| Kerb (road+sidewalk) | 3,558 | 4.6e-6 | 5.2e-4 | 1.8e-3 |

- Median kerb variance is **111.8x** median road variance (road noise is at
  millimeter scale; the kerb median corresponds to a ~2.3 cm standard
  deviation — the kerb step).
- 78% of kerb cells sit above the road p90 — a threshold a renderer could
  actually use.
- The artifact contains the top-30 kerb cells (frame, x, y, variance, count)
  as the manual verification list; the spec requires >= 20 manually verified
  cells before AC-D2 is claimed as closed. That verification is the one
  remaining human step.

### AC-D5 behaviour on real geometry — occlusion exclusion is active

Decision counters over the same 228 frames (from `stats_out` in
`negative_obstacle_candidates`):

| Counter | Total |
|---|---|
| below_plane candidates | 23,184 |
| missing-return cells flagged | 18,243,901 |
| suppressed by occlusion (AC-D5 rule) | 16,797,299 |
| skipped: no fitted plane (do not guess) | 25,982,312 |

The occlusion rule suppressed 48% of would-be missing-return candidates —
it is doing real work on real streets, not just passing a unit test. Honest
observation for later stages: the missing-return candidate volume per frame
is large (~80k, at low confidence); AC-D6 scoring of these candidates stays
gated behind the stage-2 ground-truth decision.

## Notes

- min_points is 4, not 8: at HDL-64 beam spacing a 5 cm cell in a single
  sweep rarely collects more than a handful of returns; requiring 8 empties
  both comparison groups.
- The synthetic kerb unit test places the class boundary OFF the 5 cm
  lattice; a boundary exactly on a cell edge produces zero straddling cells.
