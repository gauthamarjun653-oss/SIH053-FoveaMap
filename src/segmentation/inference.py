"""Module B — semantic segmentation wrapper.

segment(range_image, model) -> int8[H, W] 3-class labels.

Failure behaviour (spec section 7):
  * Missing/corrupt weights -> abort at startup (WeightsError). Never fall back
    to random weights silently. A RANDOM model is only available when asked for
    explicitly (random_init=True) — that path exists to prove the pipeline does
    not depend on Module B's accuracy (AC-B3 / T-06).
  * GPU unavailable -> CPU inference with cpu_fallback=True; the viewer shows a
    persistent banner stating FPS is not representative.
  * NaN output -> the frame is marked failed (returns None); the caller emits
    an empty grid and continues (T-23).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.errors import WeightsError
from src.segmentation.class_map import build_learning_to_class3

# Training-set channel statistics from the SalsaNext reference implementation,
# channel order [range, x, y, z, intensity].
SENSOR_MEANS = np.array([12.12, 10.88, 0.23, -1.04, 0.21], dtype=np.float32)
SENSOR_STDS = np.array([12.32, 11.47, 6.91, 0.86, 0.16], dtype=np.float32)


class Segmenter:
    def __init__(self, weights_path: str, semantic_kitti_yaml: str,
                 device: str = "auto", fp16: bool = False, random_init: bool = False):
        import torch  # deferred so data-only workflows never need torch
        from src.segmentation.salsanext import SalsaNext

        self._torch = torch
        self.learning_to_class3 = build_learning_to_class3(semantic_kitti_yaml)
        self.model = SalsaNext(nclasses=20)
        self.random_init = random_init

        if not random_init:
            path = Path(weights_path)
            if not path.is_file():
                raise WeightsError(
                    f"SalsaNext weights not found at '{path}'. Aborting: the pipeline "
                    "never falls back to random weights silently. Use random_init=True "
                    "only for the explicit AC-B3 end-to-end proof."
                )
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)  # T-28
            except Exception as e:
                raise WeightsError(
                    f"Failed to load '{path}' with weights_only=True: {e}. If the "
                    "checkpoint is a whole model object, convert it ONCE offline to a "
                    "state dict and commit that."
                ) from e
            if not isinstance(state, dict):
                raise WeightsError(
                    f"Checkpoint at '{path}' is not a state dict "
                    f"(got {type(state).__name__}). Convert offline; see spec 7.2."
                )
            # Tolerate common wrappers from the reference training harness.
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                     for k, v in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected:
                raise WeightsError(f"Unexpected checkpoint parameters: {unexpected[:3]}")
            if missing:
                raise WeightsError(
                    f"Checkpoint at '{path}' is missing {len(missing)} parameters "
                    f"(first: {missing[:3]}). Refusing to run partially initialised."
                )

        self.cpu_fallback = False
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            import os
            torch.set_num_threads(max(1,int(os.environ.get('FOVEAMAP_TORCH_THREADS','4'))))
            self.cpu_fallback = True  # viewer must show the persistent banner
        self.device = device
        self.fp16 = fp16 and device == "cuda"
        self.model.eval().to(device)
        if self.fp16:
            self.model.half()

    def segment_detailed(self, range_image: np.ndarray, confidence_threshold: float = .5):
        """Return detailed IDs 0..19 and confidence; low-confidence pixels are unknown."""
        if not 0 <= confidence_threshold <= 1:
            raise ValueError('Confidence threshold must be in [0,1]')
        if range_image.ndim!=3 or range_image.shape[2]!=5:
            raise ValueError('Expected a range image [H,W,5]')
        torch = self._torch
        valid = np.isfinite(range_image).all(axis=2) & (range_image[:,:,0] > 0)
        img = (range_image - SENSOR_MEANS) / SENSOR_STDS
        img[~valid] = 0
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))[None])
        x = x.to(self.device, dtype=torch.float16 if self.fp16 else torch.float32)
        with torch.no_grad():
            probs = self.model(x)
        if not bool(torch.isfinite(probs).all()):
            return None
        confidence, learning_idx = probs[0].max(dim=0)
        ids = learning_idx.cpu().numpy().astype(np.uint8)
        confidence = confidence.float().cpu().numpy()
        ids[(confidence < confidence_threshold) | ~valid] = 0
        confidence[~valid] = 0
        return ids, confidence

    def segment(self, range_image: np.ndarray) -> np.ndarray | None:
        """Compatibility API for historical three-class benchmarks."""
        result = self.segment_detailed(range_image, confidence_threshold=0.)
        return None if result is None else self.learning_to_class3[result[0]].astype(np.int8)
