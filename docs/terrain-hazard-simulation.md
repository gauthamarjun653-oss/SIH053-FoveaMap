# Curb / depression simulation

Run `.venv/bin/python scripts/simulate_terrain_hazards.py` to regenerate
`output/terrain-hazard-simulation.json`.

The script generates idealized road samples at 10 cm spacing with deterministic
2 mm Gaussian height noise, using the configured 5 cm near-field grid. There are
19 cases: depressions and raised narrow ledges at 0–40 cm in 5 cm increments,
plus a region with all returns removed. All supplied semantics are `road`.
Synthetic feature masks are used only for reporting and display, never passed
to the terrain engine. Thus a raised ledge is rejected based on geometry, not a
preassigned sidewalk class.

Each case calls the actual `build_semantic_grid` and `assess_terrain`. Optional
`diagnostics` returns fitted ground, local slope, roughness and step evidence;
normal calls retain no extra diagnostic arrays in the map. The visual displays
precomputed decisions rather than reimplementing the detector in JavaScript.

Observed results for this controlled setup:

- 10 cm rise or depression: interior cells pass the configured tests.
- 20–30 cm rise or depression: interior cells are blocked by height evidence.
- No returns: no grid cells or bottom elevation are invented in the missing area.
  Hollow visual markers indicate omitted sampling locations, not detector output.

The default step limit is 15 cm and the code compares slope-adjusted ground
min/max deviations against it. Local plane quality, slope, roughness, obstacle
clearance and semantic support also affect decisions. A configured pass is not
a guarantee of safety. Thresholds must be chosen for the vehicle.

The interactive section view shows actual measured heights and fitted planes;
the dashed ±15 cm band is explanatory. The actual code also allows legitimate
within-cell slope. The oblique view exaggerates height four times.

This is geometry testing, not physical LiDAR beam tracing: no sensor-specific
occlusion, reflection, range noise, neural segmentation errors or motion is
simulated. Consequently it does not prove real-world curb/pothole recall.

Current output is traversability (pass/blocked/unknown), not named curb or
pothole object detections. The historical negative-obstacle candidate module
remains experimental and disabled in the main pipeline. A deployable extension
needs spatial grouping of measured height differences, temporal confirmation,
visibility-aware unknown handling, and labelled real-data evaluation.

Regression verification: `pytest tests/test_core_tasks.py -q` — 24 tests passed,
including shallow/deep rises and depressions, missing-return cells and diagnostic
parity with the default terrain decisions.
