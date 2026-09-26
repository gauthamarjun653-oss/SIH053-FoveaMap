"""T-20 (benchmark without provenance aborts) and T-21 (cached inference
refused)."""

import pytest

from src.benchmark.provenance import collect_provenance
from src.common.errors import ProvenanceError

RINGS = "configs/rings.yaml"


def test_t21_cached_inference_refused(tmp_path):
    with pytest.raises(ProvenanceError) as e:
        collect_provenance(tmp_path, RINGS, "sequence 08", "cached")
    assert "cached" in str(e.value)


def test_t20_no_git_repo_aborts(tmp_path):
    with pytest.raises(ProvenanceError):
        collect_provenance(tmp_path, RINGS, "sequence 08", "gt-labels")


def test_missing_ring_config_aborts(tmp_path):
    with pytest.raises(ProvenanceError):
        collect_provenance(tmp_path, tmp_path / "nope.yaml", "sequence 08", "gt-labels")


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ProvenanceError):
        collect_provenance(tmp_path, RINGS, "sequence 08", "warp-speed")
