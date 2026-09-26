"""T-25 / AC-E5 (cached-mode banner is a functional requirement) and
T-07 (weights absent -> startup abort, no random fallback; needs torch)."""

import pytest

from src.viz.telemetry import Telemetry


def test_t25_cached_banner_present_whenever_cache_active():
    tel = Telemetry()
    assert tel.banners() == []
    tel.cached_mode = True
    assert any("CACHED" in b for b in tel.banners())
    tel.cpu_fallback = True
    banners = tel.banners()
    assert any("CACHED" in b for b in banners)
    assert any("CPU" in b for b in banners)


def test_status_line_visible_without_interaction():
    tel = Telemetry()
    tel.frame(12_500_000)
    line = tel.status_line()
    assert "MB" in line and "FPS" in line and "12.5" in line


def test_t07_weights_absent_aborts_no_random_fallback(tmp_path):
    pytest.importorskip("torch")
    from src.common.errors import WeightsError
    from src.segmentation.inference import Segmenter
    with pytest.raises(WeightsError):
        Segmenter(str(tmp_path / "missing.pt"), "configs/semantic-kitti.yaml",
                  device="cpu")


def test_t28_state_dict_load_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from src.segmentation.salsanext import SalsaNext
    from src.segmentation.inference import Segmenter
    path = tmp_path / "sd.pt"
    torch.save(SalsaNext(20).state_dict(), path)
    seg = Segmenter(str(path), "configs/semantic-kitti.yaml", device="cpu")
    assert seg.cpu_fallback  # CPU inference must raise the banner condition


def test_detailed_prediction_keeps_classes_and_abstains():
    torch = pytest.importorskip("torch")
    import numpy as np
    from src.segmentation.inference import Segmenter

    class FixedModel:
        def __call__(self, x):
            scores = torch.zeros((1, 20, 1, 4))
            scores[0, 19, 0, 0] = .9  # high class ID must survive unchanged
            scores[0, :, 0, 1] = .05  # uncertain prediction
            scores[0, 1, 0, 2:] = .9  # invalid input remains unknown
            return scores

    seg = Segmenter.__new__(Segmenter)
    seg._torch = torch
    seg.device = 'cpu'
    seg.fp16 = False
    seg.model = FixedModel()
    scan = np.ones((1, 4, 5), np.float32)
    scan[0, 2, 0] = 0
    scan[0, 3, 2] = np.nan
    ids, confidence = seg.segment_detailed(scan)
    assert ids.tolist() == [[19, 0, 0, 0]]
    assert confidence[0, 0] > .8
    assert confidence[0, 2:].tolist() == [0, 0]
    with pytest.raises(ValueError):
        seg.segment_detailed(scan, confidence_threshold=1.1)


def test_detailed_prediction_rejects_nonfinite_model_output():
    torch = pytest.importorskip("torch")
    import numpy as np
    from src.segmentation.inference import Segmenter
    seg = Segmenter.__new__(Segmenter)
    seg._torch = torch
    seg.device = 'cpu'
    seg.fp16 = False
    seg.model = lambda x: torch.full((1, 20, 1, 1), float('nan'))
    assert seg.segment_detailed(np.ones((1, 1, 5), np.float32)) is None
