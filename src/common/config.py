"""Configuration loading and validation.

The ring table is changeable via config alone (AC-C4); an invalid table is
rejected at startup naming the offending pair (AC-C5 / T-09).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml

from src.common.errors import ConfigError


@dataclass(frozen=True)
class RingSpec:
    id: int
    radius_min: float
    radius_max: float
    cell_size: float


@dataclass(frozen=True)
class RingConfig:
    base_cell_size: float
    rings: Tuple[RingSpec, ...]

    @property
    def radius_min(self) -> float:
        return self.rings[0].radius_min

    @property
    def radius_max(self) -> float:
        return self.rings[-1].radius_max

    def ring_by_id(self, ring_id: int) -> RingSpec:
        for r in self.rings:
            if r.id == ring_id:
                return r
        raise KeyError(f"No ring with id {ring_id}")


def validate_rings(rings: List[RingSpec], base: float) -> None:
    """Abort (never warn) on a ring table that cannot satisfy AC-C1.

    Every ring's cell size must be a power-of-two multiple of the base cell so
    that nesting is structural for every ring pair. A 5/10/25/50 table fails at
    R2 -> R3 (25 / 10 = 2.5) and is rejected here, naming the pair.
    """
    if not rings:
        raise ConfigError("Ring table is empty.")
    for r in rings:
        ratio = r.cell_size / base
        if abs(ratio - round(ratio)) > 1e-9 or (round(ratio) & (round(ratio) - 1)) != 0:
            raise ConfigError(
                f"Ring {r.id} cell {r.cell_size} is not a power-of-two multiple "
                f"of base {base}. AC-C1 cannot hold."
            )
        if r.radius_max <= r.radius_min:
            raise ConfigError(f"Ring {r.id} has radius_max <= radius_min.")
    for a, b in zip(rings, rings[1:]):
        if abs(b.radius_min - a.radius_max) > 1e-9:
            raise ConfigError(f"Rings {a.id}/{b.id} are not contiguous.")
        # Adjacent-ring rollup is only provably equivalent when the outer cell
        # is a power-of-two multiple of the inner cell.
        ratio = b.cell_size / a.cell_size
        if abs(ratio - round(ratio)) > 1e-9 or (round(ratio) & (round(ratio) - 1)) != 0:
            raise ConfigError(
                f"Rings {a.id}/{b.id}: cell size {b.cell_size} is not a "
                f"power-of-two multiple of {a.cell_size}. AC-C1 cannot hold "
                f"for this ring pair."
            )


def load_rings(path: str | Path) -> RingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base = float(raw["base_cell_size"])
    rings = [
        RingSpec(
            id=int(r["id"]),
            radius_min=float(r["radius_min"]),
            radius_max=float(r["radius_max"]),
            cell_size=float(r["cell_size"]),
        )
        for r in raw["rings"]
    ]
    rings.sort(key=lambda r: r.radius_min)
    validate_rings(rings, base)
    return RingConfig(base_cell_size=base, rings=tuple(rings))


@dataclass
class ProjectionConfig:
    height: int = 64
    width: int = 2048
    fov_up_deg: float = 3.0
    fov_down_deg: float = -25.0
    range_min: float = 2.5
    range_max: float = 100.0


@dataclass
class PipelineConfig:
    data_root: str
    projection: ProjectionConfig
    weights_path: str
    semantic_kitti_yaml: str
    device: str = "auto"
    fp16: bool = False
    memory_ceiling_mb: float = 512.0
    min_class_count: int = 1
    neg_depth_m: float = 0.15
    neg_max_extent_m: float = 1.0
    neg_reframed_depth_m: float = 0.08
    map_voxel_m: float = 0.10
    z_extent_min_m: float = 10.0
    z_extent_max_m: float = 40.0
    bands_m: tuple = ((0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0))
    reference_cell_m: float = 0.05
    output_dir: str = "runs"
    train_sequences: tuple = ("00", "02", "05")
    validate_sequences: tuple = ("08",)


def load_pipeline(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    proj = raw.get("projection", {})
    seg = raw.get("segmentation", {})
    grid = raw.get("grid", {})
    stats = raw.get("stats", {})
    neg = raw.get("negative_obstacles", {})
    off = raw.get("offline", {})
    bench = raw.get("benchmark", {})
    seqs = raw.get("sequences", {})
    return PipelineConfig(
        data_root=str(Path(raw["data_root"]).expanduser()),
        projection=ProjectionConfig(
            height=int(proj.get("height", 64)),
            width=int(proj.get("width", 2048)),
            fov_up_deg=float(proj.get("fov_up_deg", 3.0)),
            fov_down_deg=float(proj.get("fov_down_deg", -25.0)),
            range_min=float(proj.get("range_min", 2.5)),
            range_max=float(proj.get("range_max", 100.0)),
        ),
        weights_path=seg.get("weights_path", ""),
        semantic_kitti_yaml=seg.get("semantic_kitti_yaml", "configs/semantic-kitti.yaml"),
        device=seg.get("device", "auto"),
        fp16=bool(seg.get("fp16", False)),
        memory_ceiling_mb=float(grid.get("memory_ceiling_mb", 512.0)),
        min_class_count=int(stats.get("min_class_count", 1)),
        neg_depth_m=float(neg.get("depth_m", 0.15)),
        neg_max_extent_m=float(neg.get("max_extent_m", 1.0)),
        neg_reframed_depth_m=float(neg.get("reframed_depth_m", 0.08)),
        map_voxel_m=float(off.get("map_voxel_m", 0.10)),
        z_extent_min_m=float(off.get("z_extent_min_m", 10.0)),
        z_extent_max_m=float(off.get("z_extent_max_m", 40.0)),
        bands_m=tuple(tuple(b) for b in bench.get("bands_m", [[0, 10], [10, 25], [25, 50], [50, 100]])),
        reference_cell_m=float(bench.get("reference_cell_m", 0.05)),
        output_dir=bench.get("output_dir", "runs"),
        train_sequences=tuple(seqs.get("train", ["00", "02", "05"])),
        validate_sequences=tuple(seqs.get("validate", ["08"])),
    )
