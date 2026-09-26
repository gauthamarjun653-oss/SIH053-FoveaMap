# FoveaMap file guide

Two descriptive lines for every published file. Datasets, model binaries, environments and secrets are excluded.
The current engine is `src/core`; historical and upstream modules are retained for comparison and tooling.

## `.github/workflows/ci.yml`

Runs automated checks for repository changes in GitHub Actions.  
Keeps the test suite reproducible outside the development laptop.

## `.gitignore`

Excludes datasets, checkpoints, environments, generated sequence exports and private files from Git.  
Prevents local credentials, SSH keys and Terraform state from being added accidentally.

## `FILE_GUIDE.md`

Provides two descriptive lines for every file included in this repository.  
Covers application code, tests, dependencies, documentation and saved demonstration assets.

## `README.md`

Introduces FoveaMap, its current core pipeline and the project layout.  
Provides entry commands and distinguishes current capabilities from historical experiments.

## `REPOSITORY_SETUP.md`

Explains the published repository scope and how to prepare a new machine.  
Lists excluded local resources and clarifies how to interpret demos and stored results.

## `SiH.pdf`

Contains the project report about memory, FPS and proposed engineering improvements.  
Use it as historical design context rather than a guarantee of current performance.

## `configs/core.yaml`

Configures the current grid, terrain assessment and object-motion pipeline.  
Contains vehicle and geometry thresholds that influence traversability and classification.

## `configs/pipeline.yaml`

Sets dataset locations and shared processing, inference and visualization options.  
Update local paths and model settings before running on another machine.

## `configs/rings.yaml`

Defines distance bands and cell resolutions for the historical ring-grid pipeline.  
Loaded and validated to prevent invalid ring boundaries or resolution combinations.

## `configs/semantic-kitti.yaml`

Defines dataset labels and the mappings used by the project semantic pipeline.  
Keeps raw dataset IDs separate from neural-network learning indices.

## `conftest.py`

Configures pytest imports and shared test-session setup.  
Lets tests resolve the project modules consistently from the repository.

## `demoUI/OrbitControls.js`

Provides mouse orbit, pan and zoom controls for the Three.js scene.  
Used by the browser demo to navigate the displayed map.

## `demoUI/README.md`

Documents FOVEA MAP Demo Dashboard (Module E / S-2 shell).  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `demoUI/assets/pareto_live.png`

Stores a saved plot of the memory/accuracy comparison for the dashboard.  
Its results belong to the originating benchmark, not automatically to the current core.

## `demoUI/export_frames.py`

Export REAL KITTI keyframes for the demoUI live replay.  
Key entry points: ring_cells, main.

## `demoUI/export_scene.py`

Export ONE real KITTI scan as binned cells for the demo dashboard.  
Key entry points: bin_cells, main.

## `demoUI/index.html`

Displays the browser-based FoveaMap dashboard and map visualization.  
Real sequence exports are excluded from Git and must be regenerated locally.

## `demoUI/metrics.js`

Exposes saved dashboard metrics as JavaScript data.  
These values are loaded artifacts rather than measurements from a running sensor.

## `demoUI/metrics.json`

Stores the dashboard metrics in machine-readable form.  
Regenerate from a benchmark using wire_metrics.py when demonstrating new results.

## `demoUI/metrics.placeholder.json`

Supplies placeholder dashboard values before a real benchmark is connected.  
Placeholder numbers must not be presented as measured performance.

## `demoUI/pitch.html`

Provides the presentation-oriented browser view of the project.  
Supports explaining the mapping approach during a demo or pitch.

## `demoUI/three.min.js`

Bundles the Three.js browser rendering library.  
Supplies scene, camera, geometry and WebGL rendering primitives for the dashboard.

## `demoUI/wire_metrics.py`

Wire the demo dashboard to real data, or restore the placeholder.  
Key entry points: from_run, write, main.

## `docs/STAGE_STATUS.md`

Documents Stage Status — index.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `docs/core-improvements-2026-09-25.md`

Documents Core improvements and verified limits — 25 September 2026.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `docs/corrections-2026-09-23.md`

Documents Review corrections — 23 September 2026.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `docs/ground-visibility-definition.md`

Documents Current implementation notice — 2026-09-23.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `docs/kitti-vnc.md`

Documents KITTI + FoveaMap VNC demo.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `docs/stages/stage-0-data-staging.md`

Documents Stage 0 — Provision and data staging.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-1-global-map.md`

Documents Stage 1 — First check 1: global map.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-2-negative-precheck.md`

Documents Stage 2 — First check 2: negative-obstacle pre-check.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-3-random-end-to-end.md`

Documents Stage 3 — Random-model end-to-end (PAC-1 dry run).  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-4-offline-analytics.md`

Documents Stage 4 — Offline analytics on real data.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-5-gpu-latency.md`

Documents Stage 5 — First check 3: Module B latency on the GPU.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-5-gpu-requirements.md`

Documents Stage 5 GPU setup — requirements checklist.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-6-live-accuracy.md`

Documents Stage 6 — Live pipeline and accuracy.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-7-benchmark-pareto.md`

Documents Stage 7 — Benchmark and Pareto curve (the primary deliverable).  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-8-visualization.md`

Documents Stage 8 — Visualization and demo.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/stages/stage-9-reproducibility.md`

Documents Stage 9 — Reproducibility pass.  
Records the historical stage workflow, checks and evidence needed to reproduce that part of the project.

## `docs/terrain-hazard-simulation.md`

Documents Curb / depression simulation.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `infra/README.md`

Documents the historical AWS architecture, account migration and infrastructure workflows.  
Read its state and account warnings before adapting any cloud scripts.

## `infra/aws/.terraform.lock.hcl`

Pins Terraform provider versions and integrity checks.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/backend.tf`

Configures the historical remote Terraform state backend.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/backend.tf.example`

Provides an example remote-state backend configuration.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/bootstrap_gpu.sh`

Installs the software environment on a GPU instance.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/gpu_instance.tf`

Defines the GPU compute instance and associated infrastructure.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/launch_gpu_session.sh`

Launches the configured cloud GPU session workflow.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/main.tf`

Defines the base Terraform provider and infrastructure configuration.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/outputs.tf`

Declares values reported by Terraform after provisioning.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/run_fp16_session.sh`

Runs the mixed-precision GPU inference session workflow.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/run_gpu_pipeline.sh`

Runs the processing and measurement workflow on a GPU machine.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/s3.tf`

Preserves legacy S3 infrastructure definitions.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/terraform.tfvars.example`

Provides sample Terraform input values without private local variables.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `infra/aws/variables.tf`

Declares configurable infrastructure inputs and defaults.  
Historical optional cloud setup: review account settings and supply credentials privately before use.

## `output/assets/foveamap-hazard-refinement.png`

Provides the hazard refinement presentation illustration.  
Concept artwork for explaining the project; not a measured result or a sensor screenshot.

## `output/assets/foveamap-map-visual.png`

Provides the map visual presentation illustration.  
Concept artwork for explaining the project; not a measured result or a sensor screenshot.

## `output/assets/foveamap-memory-comparison.png`

Provides the memory comparison presentation illustration.  
Concept artwork for explaining the project; not a measured result or a sensor screenshot.

## `output/canva-preview/approval-brief.md`

Documents FoveaMap Canva approval drafts.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `output/canva-preview/slide-01.png`

Stores the visual preview of presentation slide 01.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-02.png`

Stores the visual preview of presentation slide 02.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-03.png`

Stores the visual preview of presentation slide 03.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-04.png`

Stores the visual preview of presentation slide 04.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-05.png`

Stores the visual preview of presentation slide 05.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-06.png`

Stores the visual preview of presentation slide 06.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-07.png`

Stores the visual preview of presentation slide 07.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/canva-preview/slide-08.png`

Stores the visual preview of presentation slide 08.  
Use as a reference when editing the corresponding Canva presentation slide.

## `output/core-cli-smoke.json`

Stores the saved core cli smoke report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-development09.json`

Stores the saved core development09 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-development10.json`

Stores the saved core development10 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-final09.json`

Stores the saved core final09 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-final10.json`

Stores the saved core final10 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-live-demo.png`

Captures a rendered demonstration of the current core viewer.  
A static screenshot illustrates output but does not measure current live throughput.

## `output/core-memory-comparison.json`

Stores the saved core memory comparison report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-model-gpu09.json`

Stores the saved core model gpu09 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-model-gpu10.json`

Stores the saved core model gpu10 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-model-smoke10.json`

Stores the saved core model smoke10 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/core-smoke10.json`

Stores the saved core smoke10 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/corrections-2026-09-23/cli-seq09.json`

Stores the saved cli seq09 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/corrections-2026-09-23/cli-seq10.json`

Stores the saved cli seq10 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/corrections-2026-09-23/detector-evaluation.json`

Stores the saved detector evaluation report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/corrections-2026-09-23/grid-layouts.json`

Stores the saved grid layouts report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/corrections-2026-09-23/validation.json`

Stores the saved validation report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/curbs-and-potholes-source.html`

Contains the editable source of the curb and depression explanation demo.  
Explains geometry evidence using controlled scenes rather than annotated real potholes.

## `output/curbs-and-potholes.html`

Provides the standalone browser demonstration of curb and depression evidence.  
Shows synthetic cases; successful simulation does not establish real-world detection accuracy.

## `output/eval_negative_obs.json`

Stores the saved eval negative obs report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/eval_r25.json`

Stores the saved eval r25 report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/eval_split.json`

Stores the saved eval split report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/examples/manifest.json`

Lists selected examples intended for manual checking of terrain evidence.  
It is an index for review; referenced sequence data is not bundled.

## `output/grid_layout_comparison.json`

Stores the saved grid layout comparison report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/pdf/FoveaMap_Memory_FPS_and_Next_Steps.pdf`

Summarizes the memory/FPS discussion and recommended engineering work.  
Provides historical context for the measurements, limitations and improvement plan.

## `output/profile_stages.json`

Stores the saved profile stages report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/profile_stages_negobs.json`

Stores the saved profile stages negobs report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/seq07-height-evidence-check.json`

Stores the saved seq07 height evidence check report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/seq07-pedestrian-inventory.json`

Stores the saved seq07 pedestrian inventory report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/seq07-pedestrian-model-check.json`

Stores the saved seq07 pedestrian model check report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `output/slides/feasibility-viability-image-prompt.txt`

Stores the design prompt for the feasibility and viability slide image.  
Preserves the intended visual composition for later slide revisions.

## `output/slides/feasibility-viability-preview.png`

Shows a rendered preview of the feasibility and viability slide.  
This is a presentation asset rather than executable pipeline output.

## `output/terrain-hazard-simulation.json`

Stores the saved terrain hazard simulation report or diagnostic results.  
Historical output for inspection and comparison; regenerate to measure another dataset, machine or code revision.

## `requirements.txt`

Lists Python dependencies for processing, segmentation, visualization and testing.  
Install into a virtual environment; pretrained model weights are obtained separately.

## `resource/SIH26053_Claude_Code_Spec.md`

Documents SIH26053 — Implementation Specification.  
Provides project instructions or design context; check dated claims against the current core and fresh measurements.

## `run_seq07.sh`

Starts full sequence 07 playback from frame 0 through the existing VNC viewer, using model inference by default.  
Supports dataset-label replay, replay interval and optional browser opening; requires local data and dependencies.

## `scripts/compare_core_memory.py`

Honest payload comparison: old/new maps and a minimal sparse 3D reference.  
Key entry points: voxel_reference, main.

## `scripts/compare_grid_layouts.py`

Step 5 measurement — cascade/float64 vs direct/compact.  
Key entry points: rss, peak_rss, stat, layout_worker.

## `scripts/download_kitti_09_10.py`

Fetch only sequences 09/10 from official KITTI ZIPs using HTTP ranges.  
Key entry points: RemoteZip, crc, selected, fetch.

## `scripts/download_kitti_sequence.py`

Download one labelled KITTI sequence, including scans, labels, poses and calibration.  
Key entry points: fetch_labels, main.

## `scripts/eval_negative_obs.py`

Step 4 — accuracy and cost comparison, old detector vs new.  
Key entry points: stat, range_histogram, main.

## `scripts/extract_examples.py`

Build the hand-check example set required by the ground/visibility definition §5.  
Key entry points: cross_section, main.

## `scripts/kitti_vnc.sh`

Manages the SemanticKITTI/FoveaMap viewer and its VNC playback session.  
Accepts a sequence and mode so dataset-label replay can be separated from model predictions.

## `scripts/profile_stages.py`

Step 1 - reproducible per-stage profiling.  
Key entry points: rss_bytes, peak_rss_bytes, summarise, uniform_config.

## `scripts/run_core.py`

Run and measure the three PS tasks. GT replay is explicitly labelled.  
Key entry points: main.

## `scripts/simulate_terrain_hazards.py`

Run the actual terrain assessor on controlled synthetic curb/depression scenes.  
Key entry points: scenario, run_scenarios, main.

## `scripts/start_vnc.sh`

Starts the desktop display and browser-accessible VNC components for visualization.  
Requires the native VNC and display dependencies described in the setup notes.

## `scripts/viz_dual_fovea.py`

Dual view on one VNC window: LEFT = raw 3D scan, RIGHT = the same scan processed through the real pipeline into the 2.5D foveated grid. One shared camera (orbiting one orbits both) and one frame index (n/b step both), so the 3D-to-2.5D transformation is directly.  
Key entry points: depth_colormap, process, main.

## `scripts/viz_semantickitti_fovea.py`

PRBonn SemanticKITTI viewer with an additional FoveaMap panel.  
Key entry points: CurrentScan, FoveaKITTIVis, main.

## `src/__init__.py`

Initializes the src Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/analytics/__init__.py`

Initializes the src.analytics Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/analytics/cell_stats.py`

Module D — cell statistics.  
Key entry points: expected_returns, compute_stats, render_class, kerb_road_variance_cells.

## `src/analytics/ground.py`

Ground estimates with per-plane convex-hull support and fit-quality gates.  
Key entry points: GroundPlane, GroundField, _fit, build_ground_field.

## `src/analytics/negative_obs.py`

Negative-obstacle inference (Module D) and offline ground-truth derivation for the section 11.3 pre-check.  
Key entry points: GroundPlane, NegativeObstacleCandidate, fit_ground_planes, _OcclusionIndex.

## `src/analytics/negative_obs2.py`

Negative-obstacle detection, rebuilt on the ground/visibility definition.  
Key entry points: NegativeObstacle, DetectionResult, _components, _below_plane_candidates.

## `src/analytics/visibility.py`

Modelled beam intersections and evidence-aware visibility.  
Key entry points: beam_elevations, derive_beam_elevations, VisibilityResult, _cell_of.

## `src/benchmark/__init__.py`

Initializes the src.benchmark Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/benchmark/harness.py`

Benchmark harness (spec 11) — regenerates the Pareto curve from a single command with no manual steps (PAC-6).  
Key entry points: run_benchmark.

## `src/benchmark/provenance.py`

Run provenance — required, not optional (spec 11.6).  
Key entry points: _git_commit, _hardware, collect_provenance.

## `src/common/__init__.py`

Initializes the src.common Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/common/config.py`

Configuration loading and validation.  
Key entry points: RingSpec, RingConfig, validate_rings, load_rings.

## `src/common/errors.py`

Project error types.  
Key entry points: ConfigError, DatasetError, WeightsError, MemoryCeilingExceeded.

## `src/core/__init__.py`

Initializes the src.core Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/core/grid.py`

40-byte occupied leaf records on a disjoint, distance-adaptive dyadic grid.  
Key entry points: SemanticGrid, unique_rows, leaf_keys, build_semantic_grid.

## `src/core/motion.py`

Bounded geometric object tracking; no instance IDs or motion GT inputs.  
Key entry points: Track, MotionTracker.

## `src/core/pipeline.py`

Primary PS pipeline. Experimental pothole inference is intentionally excluded.  
Key entry points: CoreResult, CorePipeline, display_colors.

## `src/core/semantics.py`

Detailed learning taxonomy, kept separate from motion and traversability.  
Key entry points: detailed_labels, motion_truth.

## `src/core/terrain.py`

Conservative traversability from separate ground/obstacle layers.  
Key entry points: VehicleLimits, assess_terrain, load_core_settings.

## `src/frame_pipeline.py`

Shared per-scan map/analytics path for the CLI and both viewers.  
Key entry points: FrameResult, process_frame.

## `src/grid/__init__.py`

Initializes the src.grid Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/grid/compact.py`

Step 5 — compact cell records and direct construction.  
Key entry points: CompactCellBlock, CoordinateOverflow, pack_block, build_block_direct.

## `src/grid/foveated_grid.py`

Module C — Foveated grid engine (the contribution).  
Key entry points: Cell, zero_cell, update, combine.

## `src/grid/lattice.py`

Shared lattice indexing.  
Key entry points: cell_index, cell_indices.

## `src/grid/rings.py`

Ring assignment by planar radius.  
Key entry points: assign_rings, assign_ring_scalar.

## `src/io/__init__.py`

Initializes the src.io Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/io/kitti_reader.py`

KITTI odometry / SemanticKITTI I/O.  
Key entry points: read_scan, read_labels, load_calib, load_poses.

## `src/main.py`

SIH26053 command-line entry point.  
Key entry points: _load, cmd_validate_config, cmd_build_map, cmd_precheck_negatives.

## `src/offline/__init__.py`

Initializes the src.offline Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/offline/baselines.py`

Baseline representations for the Pareto comparison (spec 11.4).  
Key entry points: UniformVoxel3D, UniformOccupancy2D.

## `src/offline/build_map.py`

Module F — global map construction (offline, run once; NOT part of the deployed system — keep that separation visible in every diagram).  
Key entry points: velo_to_world, _voxel_keys, build_global_map, assert_z_extent.

## `src/offline/pareto.py`

Recall metric (spec section 10) and the Pareto curve (spec 11.5).  
Key entry points: band_of, reference_obstacle_cells, foveated_obstacle_lookup, foveated_detects.

## `src/projection/__init__.py`

Initializes the src.projection Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/projection/range_projection.py`

Module A — Ingestion and spherical projection.  
Key entry points: ProjectionResult, project, back_project.

## `src/segmentation/__init__.py`

Initializes the src.segmentation Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/segmentation/class_map.py`

Class taxonomy and the namespace trap (spec section 5).  
Key entry points: _raw_to_class3, build_learning_to_class3, build_raw_to_class3.

## `src/segmentation/inference.py`

Module B — semantic segmentation wrapper.  
Key entry points: Segmenter.

## `src/segmentation/salsanext.py`

SalsaNext architecture (Cortinhal et al., 2020).  
Key entry points: ResContextBlock, ResBlock, UpBlock, SalsaNext.

## `src/viz/__init__.py`

Initializes the src.viz Python package.  
Makes the package modules importable; any public exports are defined here.

## `src/viz/rerun_viewer.py`

Module E — rerun.io viewer (S-1 / S-2 / S-3).  
Key entry points: _require_rerun, SequenceEntry, list_sequences, format_sequence_table.

## `src/viz/telemetry.py`

Telemetry: rolling FPS, live memory, and mandatory status banners.  
Key entry points: RollingFPS, Telemetry.

## `start_demo.sh`

Starts the browser demonstration using the project launch workflow.  
Use with the HTML dashboard and its configured local server dependencies.

## `tests/__init__.py`

Initializes the tests Python package.  
Makes the package modules importable; any public exports are defined here.

## `tests/test_cell_safety.py`

T-14 / AC-D4 (safety): 99 ROAD + 1 DYNAMIC surfaces DYNAMIC in both payload and render. Plus AC-D1: no NaN in any stats field.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_class_map.py`

T-04: build_learning_to_class3 derived from YAML and asserted vs literal.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_config_validation.py`

T-09 / AC-C4 / AC-C5: ring table validation at startup.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_core_tasks.py`

Tests compact grid ownership, terrain assessment, semantic handling and motion behavior.  
Exercises the current core pipeline and synthetic terrain-evidence regressions.

## `tests/test_corrected_pipeline.py`

Regression cases from the September review; canonical tests stay intact.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_kitti_io.py`

T-19: scan/pose/label count mismatch aborts naming all three.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_lossless_aggregation.py`

AC-C1 / T-08: lossless aggregation, every ring pair, BIT-IDENTICAL.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_map_transform.py`

T-17 / AC-F1: the inv(Tr) conjugation, and the z-extent assertion that catches its omission — the highest-consequence low-visibility bug in the codebase.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_negative_obstacles.py`

T-15 (occlusion excludes flagging) and T-16 (synthetic depression detected, confidence in [0,1]); plus the offline ground-truth deriver on known geometry.  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_projection_roundtrip.py`

Module A: T-01 (bad .bin size), T-02 (round-trip), T-03 (collision), AC-A2 (accounting), AC-A3 (determinism).  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_provenance.py`

T-20 (benchmark without provenance aborts) and T-21 (cached inference refused).  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_telemetry_and_weights.py`

T-25 / AC-E5 (cached-mode banner is a functional requirement) and T-07 (weights absent -> startup abort, no random fallback; needs torch).  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_viz.py`

Module E: S-1 selector rules, comparison toggle invariants (AC-E4/E6), frame-failure markers, and that log_frame actually runs against the installed rerun SDK (the set_time_sequence -> set_time API break was caught here).  
Regression coverage; run through pytest to check the associated behavior.

## `tests/test_welford.py`

T-10 / AC-C6: Welford combine against hand-computed expected values.  
Regression coverage; run through pytest to check the associated behavior.

## `third_party/semantic-kitti-api/.gitignore`

Excludes upstream-generated files from version control.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/.pep8`

Configures upstream Python style checking.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/Dockerfile`

Defines the upstream container environment.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/LICENSE`

Preserves the license of the vendored SemanticKITTI API.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/README.md`

Documents installation and use of the upstream SemanticKITTI API.  
Explains its dataset tools, evaluation commands and visualization workflows.

## `third_party/semantic-kitti-api/UPSTREAM_VERSION.txt`

Records the provenance of the vendored upstream code.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/SSCDataset.py`

Loads semantic scene-completion samples and voxel targets.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/__init__.py`

Initializes the third_party.semantic-kitti-api.auxiliary Python package.  
Makes the package modules importable; any public exports are defined here.

## `third_party/semantic-kitti-api/auxiliary/camera.py`

Implements camera controls and projection for voxel visualization.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/eval_np.py`

Provides NumPy evaluation routines for semantic and instance predictions.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/filelist2files.py`

Converts lists of dataset paths into inputs used by the API tools.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/glow.py`

Provides OpenGL helpers for shader-driven visualization.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/laserscan.py`

Loads LiDAR scans, semantic labels and range-image projections.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/laserscancomp.py`

Builds synchronized scan comparison views.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/laserscanvis.py`

Renders semantic LiDAR scans with VisPy.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/np_ioueval.py`

Computes confusion matrices and intersection-over-union using NumPy.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/shaders/check_uniforms.vert`

Provides the check uniforms OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/shaders/draw_pose.geom`

Provides the draw pose OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/shaders/draw_voxels.frag`

Provides the draw voxels OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/shaders/draw_voxels.vert`

Provides the draw voxels OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/shaders/empty.frag`

Provides the empty OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/shaders/empty.vert`

Provides the empty OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/shaders/passthrough.frag`

Provides the passthrough OpenGL shader stage.  
Used by the upstream voxel/pose visualization renderer, not the mapping engine.

## `third_party/semantic-kitti-api/auxiliary/torch_ioueval.py`

Computes segmentation evaluation statistics using PyTorch.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/auxiliary/vispy_manager.py`

Coordinates VisPy canvases and interactive viewer events.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/compare.py`

Compares semantic scan visualizations side by side.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/config/semantic-kitti-all.yaml`

Defines the semantic-kitti-all label taxonomy and evaluation mappings.  
Consumed by upstream dataset tools and viewers to interpret semantic IDs.

## `third_party/semantic-kitti-api/config/semantic-kitti-coarse.yaml`

Defines the semantic-kitti-coarse label taxonomy and evaluation mappings.  
Consumed by upstream dataset tools and viewers to interpret semantic IDs.

## `third_party/semantic-kitti-api/config/semantic-kitti-mos.yaml`

Defines the semantic-kitti-mos label taxonomy and evaluation mappings.  
Consumed by upstream dataset tools and viewers to interpret semantic IDs.

## `third_party/semantic-kitti-api/config/semantic-kitti.yaml`

Defines the semantic-kitti label taxonomy and evaluation mappings.  
Consumed by upstream dataset tools and viewers to interpret semantic IDs.

## `third_party/semantic-kitti-api/content.py`

Calculates the distribution of dataset semantic classes.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/count.py`

Counts points and labels in the dataset.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/docker.sh`

Launches the upstream container workflow.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/evaluate_completion.py`

Evaluates semantic scene-completion predictions.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/evaluate_mos.py`

Evaluates moving-object segmentation predictions.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/evaluate_panoptic.py`

Evaluates panoptic segmentation predictions.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/evaluate_semantics.py`

Evaluates semantic segmentation predictions.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/evaluate_semantics_by_distance.py`

Evaluates semantic predictions across distance ranges.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/generate_sequential.py`

Prepares sequential scan inputs for multi-scan processing.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/remap_semantic_labels.py`

Converts semantic IDs between raw and learning label spaces.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/requirements.txt`

Lists dependencies for the upstream API and viewers.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/validate_submission.py`

Checks the format and completeness of benchmark submissions.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/visualize.py`

Starts the semantic point-cloud viewer.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/visualize_mos.py`

Starts the moving-object segmentation viewer.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `third_party/semantic-kitti-api/visualize_voxels.py`

Displays voxel occupancy and semantic scene-completion data.  
Vendored SemanticKITTI API support code; retained with its upstream license and provenance.

## `weights/README.md`

Documents model-weight provenance and the expected SalsaNext checkpoint setup.  
Weights are excluded from Git; obtain compatible files separately before model inference.

## `weights/salsanext-arch.yaml`

Records the SalsaNext architecture and associated model configuration.  
Use it with compatible pretrained weights when preparing neural inference.

## `weights/salsanext-data.yaml`

Stores the semantic dataset configuration associated with SalsaNext.  
Defines label mappings and preprocessing settings needed for compatible predictions.
