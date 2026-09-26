"""Telemetry: rolling FPS, live memory, and mandatory status banners.

Memory (MB) and FPS (rolling 30-frame average) must be visible without
interaction (AC-E4). The cached-mode banner is a FUNCTIONAL requirement
(AC-E5): its silent failure would let cached-throughput figures be presented as
live perception throughput.
"""

from __future__ import annotations

import time
from collections import deque


class RollingFPS:
    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window)
        self._last: float | None = None

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            self.times.append(now - self._last)
        self._last = now

    @property
    def fps(self) -> float:
        if not self.times:
            return 0.0
        mean = sum(self.times) / len(self.times)
        return 1.0 / mean if mean > 0 else 0.0


class Telemetry:
    def __init__(self):
        self.fps = RollingFPS()
        self.memory_mb: float = 0.0
        self.uniform_memory_mb: float | None = None  # comparison grid, when shown
        self.cached_mode: bool = False
        self.cpu_fallback: bool = False
        self.failed_frames: list[int] = []

    def frame(self, memory_bytes: int, uniform_memory_bytes: int | None = None) -> None:
        self.fps.tick()
        self.memory_mb = memory_bytes / 1e6
        if uniform_memory_bytes is not None:
            self.uniform_memory_mb = uniform_memory_bytes / 1e6

    def banners(self) -> list[str]:
        """Persistent banners. AC-E5 asserts the cached banner is present
        whenever cache is active — tested, not assumed."""
        out = []
        if self.cached_mode:
            out.append("CACHED MODE — results are replayed, NOT live perception throughput")
        if self.cpu_fallback:
            out.append("CPU INFERENCE — FPS is not representative of the demo GPU")
        return out

    def status_line(self) -> str:
        # Both grids' memory stays visible even while one layer is toggled off
        # (AC-E4): the comparison is about cost, and cost must not hide with
        # the layer.
        if self.uniform_memory_mb is not None:
            return (f"foveated {self.memory_mb:.1f} MB | "
                    f"uniform {self.uniform_memory_mb:.1f} MB | "
                    f"{self.fps.fps:.1f} FPS")
        return f"{self.memory_mb:.1f} MB | {self.fps.fps:.1f} FPS"
