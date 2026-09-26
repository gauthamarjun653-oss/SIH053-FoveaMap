# Published project setup

See [FILE_GUIDE.md](FILE_GUIDE.md) for two lines about each published file.
Source, tests, configs, docs, reports, presentation assets and the vendored SemanticKITTI API are included.

## Kept local

Sequence scans, labels, poses, calibration, archives and dataset-derived browser exports are excluded.
Model checkpoints, virtual environments, caches, logs and run directories are excluded.
SSH keys, cloud credentials, environment secrets, Terraform state and private variables are excluded.
No original datasets or private files were deleted by preparing this upload.

## Setup

Create a Python virtual environment and install `requirements.txt`.
Set `data_root` in `configs/pipeline.yaml` to your SemanticKITTI dataset.
Use the download scripts' CLI help to obtain sequences separately from official sources, respecting their license/access terms.
Run `python -m pytest tests/ -q` for tests.
Run `python scripts/run_core.py --sequence 07 --mode gt-labels --frames 30` for dataset-label replay.
GT-label replay evaluates mapping, not neural prediction quality.
For `--mode model`, obtain compatible SalsaNext weights separately following `weights/README.md`.

VNC playback additionally needs VisPy, a supported Qt/OpenGL backend, TigerVNC and noVNC/websockify.
The old machine's `.venv/native/usr` installation is not bundled; adjust viewer executable paths to your installation.
See `docs/kitti-vnc.md`. Serve HTML demos with `python -m http.server 8080`.
Regenerate real sequence browser exports locally using the `demoUI` export scripts.

## Interpreting this repository

`src/core` is the current pipeline. Historical grid/negative-obstacle modules remain for reproducibility.
Saved `output` reports are historical measurements, not fresh results from a new machine.
Presentation concept images are illustrations, not sensor evidence.
Synthetic curb/depression demonstrations do not establish real-world pothole accuracy.
Cloud scripts are optional historical infrastructure: supply private keys locally and adapt account-specific settings before use.
Cloning the repository does not launch cloud resources.
