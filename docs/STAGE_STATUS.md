# Stage Status — index

**Updated:** 2026-08-27 (session on the Azure VM). One file per stage under
[docs/stages/](stages/); each carries its purpose, commands, data inventory,
measured results, and pending decisions. Update the stage file when a stage
changes state; measured figures come from actual runs — never edit them by
hand without a run to back them.

| # | Stage | Status | Hardware |
|---|---|---|---|
| 0 | [Data staging](stages/stage-0-data-staging.md) | DONE — S3 canonical store complete (all sequences, labels, weights) | Azure VM + S3 |
| 1 | [Global map (first check 1)](stages/stage-1-global-map.md) | DONE — AC-F1 23.6 m, AC-F2 13.1 cm | Azure VM |
| 2 | [Negative-obstacle pre-check (first check 2)](stages/stage-2-negative-precheck.md) | RAN — count recorded; spot-check pending before AC-D6 scoring | Azure VM |
| 3 | [Random-model end-to-end](stages/stage-3-random-end-to-end.md) | DONE — 4,541/4,541 frames, no failures (PAC-1) | Azure VM |
| 4 | [Offline analytics on real data](stages/stage-4-offline-analytics.md) | MEASURED — AC-D1 pass; kerb variance 111.8x road; manual spot-check pending | Azure VM |
| 5 | [Module B latency (first check 3)](stages/stage-5-gpu-latency.md) | PASSED — 26.2 ms at fp16 (budget 40 ms); ladder closed at rung 1 | AWS g4dn (Tesla T4) |
| 6 | [Live pipeline + accuracy](stages/stage-6-live-accuracy.md) | DONE — 4,541/4,541 frames live on Seq 00 with SalsaNext (0 failures) | AWS g4dn (Tesla T4) |
| 7 | [Benchmark + Pareto (primary deliverable)](stages/stage-7-benchmark-pareto.md) | DONE — 4,071 frames on Seq 08 live on Tesla T4; 92.0% near recall, 97.5% far recall, 3.30 MB/sweep | AWS g4dn (Tesla T4) |
| 8 | [Visualization + demo](stages/stage-8-visualization.md) | VIEWER COMPLETE + demoUI wired with live Tesla T4 benchmark metrics | Laptop / VM |
| 9 | [Reproducibility pass](stages/stage-9-reproducibility.md) | READY — clean-provenance reference exists (fe332e1); 27-vs-28 Aug runs agree within 0.02 pp | Clean machine |

Standing rules (owner-set): never create, resize, deallocate, or destroy any
cloud instance without explicit owner permission; GPU work only on
the AWS g4dn; performance numbers are measured, never asserted, and CPU
timings are never quoted as results.
