"""Run provenance — required, not optional (spec 11.6).

Every benchmark run records git commit hash, ring-config hash, hardware spec,
dataset split, and whether inference was live or cached. The run ABORTS without
them: optional metadata is what gets omitted under deadline pressure, which is
exactly when it matters. The harness also refuses to run against a cached
inference path (T-21).
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.common.errors import ProvenanceError

VALID_INFERENCE_MODES = ("live", "random", "gt-labels")


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ProvenanceError(f"Cannot obtain git commit hash: {e}") from e
    if out.returncode != 0:
        raise ProvenanceError(
            "Cannot obtain git commit hash — is this a git repository with at "
            f"least one commit? ({out.stderr.strip()}) Benchmark runs abort "
            "without provenance."
        )
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                           capture_output=True, text=True).stdout.strip()
    commit = out.stdout.strip()
    return commit + ("-dirty" if dirty else "")


def _hardware() -> dict:
    hw = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch
        hw["torch"] = torch.__version__
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        hw["torch"] = None
        hw["cuda_available"] = False
    return hw


def collect_provenance(repo_root: str | Path, ring_config_path: str | Path,
                       dataset_split: str, inference_mode: str) -> dict:
    """Assemble complete provenance or abort. Never returns a partial record."""
    if inference_mode == "cached":
        raise ProvenanceError(
            "Benchmark refused: inference_mode='cached'. Cached-throughput "
            "figures must never be presentable as live perception throughput."
        )
    if inference_mode not in VALID_INFERENCE_MODES:
        raise ProvenanceError(
            f"Unknown inference_mode '{inference_mode}'; expected one of "
            f"{VALID_INFERENCE_MODES}."
        )
    ring_path = Path(ring_config_path)
    if not ring_path.is_file():
        raise ProvenanceError(f"Ring config '{ring_path}' not found; cannot hash it.")
    prov = {
        "git_commit": _git_commit(Path(repo_root)),
        "ring_config_sha256": hashlib.sha256(ring_path.read_bytes()).hexdigest(),
        "hardware": _hardware(),
        "dataset_split": dataset_split,
        "inference_mode": inference_mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    missing = [k for k, v in prov.items() if v in (None, "", {})]
    if missing or not dataset_split:
        raise ProvenanceError(f"Provenance incomplete ({missing or 'dataset_split'}); aborting run.")
    return prov
