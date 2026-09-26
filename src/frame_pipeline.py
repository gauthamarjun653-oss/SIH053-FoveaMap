"""Shared per-scan map/analytics path for the CLI and both viewers.

Timings begin after segmentation; callers must include inference, input and
rendering separately before claiming end-to-end FPS.
"""
from dataclasses import dataclass
from time import perf_counter
from src.analytics.cell_stats import compute_stats
from src.analytics.negative_obs2 import detect
from src.grid.compact import build_compact_grid
from src.grid.foveated_grid import build_grid

@dataclass
class FrameResult:
    grid: object
    candidates: list
    build_ms: float
    detection_ms: float
    detection: object | None


def process_frame(points_xyz, labels, rings, *, implementation="corrected",
                  image_width=2048, memory_ceiling_bytes=None,
                  depth_threshold=.15, max_extent_m=1., blind_ranges_m=None, experimental_hazards=False):
    if implementation not in ("corrected", "legacy"):
        raise ValueError("Unknown pipeline implementation")
    started = perf_counter()
    if implementation == "corrected":
        grid = build_compact_grid(points_xyz, labels, rings, image_width,
                                  memory_ceiling_bytes)
    else:
        grid = compute_stats(build_grid(points_xyz, labels, rings,
                             memory_ceiling_bytes=memory_ceiling_bytes), image_width)
    built = perf_counter()
    detection = None
    if not experimental_hazards:
        return FrameResult(grid, [], (built-started)*1000, 0., None)
    if implementation == "corrected":
        detection = detect(grid, rings, points_xyz=points_xyz, image_width=image_width,
                           depth_threshold=depth_threshold, max_extent_m=max_extent_m,
                           blind_ranges_m=blind_ranges_m)
        candidates = detection.candidates
    else:
        from src.analytics.negative_obs import negative_obstacle_candidates
        candidates = negative_obstacle_candidates(grid, rings, depth_threshold=depth_threshold)
    return FrameResult(grid, candidates, (built-started)*1000,
                       (perf_counter()-built)*1000, detection)
