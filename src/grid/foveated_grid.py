"""Module C — Foveated grid engine (the contribution).

Storage is a sparse structure-of-arrays keyed by (ring_id, cell_x, cell_y):
only occupied cells are allocated, so an empty scene costs ~zero (AC-C2), and
per-object Python overhead never contaminates the measured footprint.

Lossless aggregation (AC-C1, bit-identical) is achieved structurally:

  * Every cell at size base * 2^k is DEFINED as the canonical dyadic fold of
    its base-lattice cells: base cells are computed once, deterministically,
    then combined 2x2 -> parent through each intermediate power-of-two level,
    children folded in fixed (cx, cy) order.
  * `aggregate()` runs the very same doubling. Building natively at a coarse
    size and aggregating a fine grid to that size therefore execute the
    identical floating-point computation DAG, and the results are bit-identical
    — not merely close.

Variance uses Welford/Chan (count, mean, M2). Variance is NOT additive;
averaging child variances is silently wrong and looks correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.common.config import RingConfig
from src.common.errors import MemoryCeilingExceeded
from src.grid.lattice import cell_indices
from src.grid.rings import assign_rings, OUT_OF_RANGE

N_CLASSES = 4  # IGNORE, ROAD, STATIC, DYNAMIC


# ---------------------------------------------------------------------------
# Scalar reference cells (spec pseudocode) — used by unit tests (T-10) and as
# the executable definition of `combine`.
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0
    z_min: float = math.inf
    z_max: float = -math.inf
    hist: np.ndarray = field(default_factory=lambda: np.zeros(N_CLASSES, dtype=np.uint32))


def zero_cell() -> Cell:
    return Cell()


def update(cell: Cell, z: float, cls: int) -> None:
    """Streaming Welford update."""
    cell.count += 1
    delta = z - cell.mean
    cell.mean += delta / cell.count
    cell.M2 += delta * (z - cell.mean)
    cell.hist[cls] += 1
    cell.z_min = min(cell.z_min, z)
    cell.z_max = max(cell.z_max, z)


def combine(a: Cell, b: Cell) -> Cell:
    """Parallel (Chan) combination — the only sanctioned way to merge cells."""
    n = a.count + b.count
    if n == 0:
        return zero_cell()
    delta = b.mean - a.mean
    return Cell(
        count=n,
        mean=a.mean + delta * b.count / n,
        M2=a.M2 + b.M2 + delta * delta * a.count * b.count / n,
        z_min=min(a.z_min, b.z_min),
        z_max=max(a.z_max, b.z_max),
        hist=(a.hist + b.hist).astype(np.uint32),
    )


def variance(cell: Cell) -> float:
    return cell.M2 / cell.count if cell.count > 1 else 0.0  # never NaN


# ---------------------------------------------------------------------------
# Structure-of-arrays cell block
# ---------------------------------------------------------------------------

class CellBlock:
    """Occupied cells of one lattice level, sorted by (cx, cy)."""

    __slots__ = ("cell_size", "cx", "cy", "count", "mean", "m2",
                 "z_min", "z_max", "hist",
                 "z_var", "occupancy_conf", "neg_obstacle_conf")

    def __init__(self, cell_size: float, cx, cy, count, mean, m2, z_min, z_max, hist):
        self.cell_size = float(cell_size)
        self.cx = cx
        self.cy = cy
        self.count = count
        self.mean = mean
        self.m2 = m2
        self.z_min = z_min
        self.z_max = z_max
        self.hist = hist
        # Derived stats (Module D fills these in place)
        self.z_var = None
        self.occupancy_conf = None
        self.neg_obstacle_conf = None

    def __len__(self) -> int:
        return int(self.cx.shape[0])

    @classmethod
    def empty(cls, cell_size: float) -> "CellBlock":
        z = np.zeros(0)
        return cls(cell_size,
                   np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                   np.zeros(0, dtype=np.int64), z.copy(), z.copy(), z.copy(), z.copy(),
                   np.zeros((0, N_CLASSES), dtype=np.uint32))

    def memory_bytes(self) -> int:
        """Measured footprint of the payload arrays. Measure, never assert."""
        total = 0
        for name in ("cx", "cy", "count", "mean", "m2", "z_min", "z_max", "hist",
                     "z_var", "occupancy_conf", "neg_obstacle_conf"):
            arr = getattr(self, name)
            if arr is not None:
                total += arr.nbytes
        return total


def base_block_from_points(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           labels: np.ndarray, base_size: float) -> CellBlock:
    """Deterministic base-lattice accumulation.

    Points are grouped by cell with a stable sort (original point order kept
    within each cell), then per-cell statistics are computed with sequential
    reduceat folds. This function is the single shared root of every coarser
    cell's computation DAG — both native builds and aggregations start here,
    which is what makes AC-C1 bit-identical.
    """
    n = x.shape[0]
    if n == 0:
        return CellBlock.empty(base_size)

    cx, cy = cell_indices(x, y, base_size)
    # lexsort: last key is primary -> sort by cx, then cy, then original index.
    order = np.lexsort((np.arange(n), cy, cx))
    cx_s, cy_s = cx[order], cy[order]
    z_s = z[order].astype(np.float64)
    lab_s = labels[order]

    new_cell = np.empty(n, dtype=bool)
    new_cell[0] = True
    new_cell[1:] = (cx_s[1:] != cx_s[:-1]) | (cy_s[1:] != cy_s[:-1])
    starts = np.flatnonzero(new_cell)
    cell_of_point = np.cumsum(new_cell) - 1

    counts = np.diff(np.append(starts, n)).astype(np.int64)
    sums = np.add.reduceat(z_s, starts)
    means = sums / counts
    centered = z_s - means[cell_of_point]
    m2 = np.add.reduceat(centered * centered, starts)
    z_min = np.minimum.reduceat(z_s, starts)
    z_max = np.maximum.reduceat(z_s, starts)

    hist = np.zeros((starts.shape[0], N_CLASSES), dtype=np.uint32)
    np.add.at(hist, (cell_of_point, lab_s.astype(np.int64)), 1)

    return CellBlock(base_size, cx_s[starts], cy_s[starts], counts, means, m2,
                     z_min, z_max, hist)


def _combine_rows(acc: dict, rows: dict, dst: np.ndarray, src: np.ndarray) -> None:
    """Vectorised Chan combine of rows[src] into acc[dst]."""
    ca, cb = acc["count"][dst], rows["count"][src]
    n = ca + cb
    delta = rows["mean"][src] - acc["mean"][dst]
    acc["mean"][dst] = acc["mean"][dst] + delta * cb / n
    acc["m2"][dst] = acc["m2"][dst] + rows["m2"][src] + delta * delta * ca * cb / n
    acc["count"][dst] = n
    acc["z_min"][dst] = np.minimum(acc["z_min"][dst], rows["z_min"][src])
    acc["z_max"][dst] = np.maximum(acc["z_max"][dst], rows["z_max"][src])
    acc["hist"][dst] += rows["hist"][src]


def double_block(block: CellBlock) -> CellBlock:
    """One canonical doubling step: cell size 2x, children folded in (cx, cy) order."""
    n = len(block)
    parent_size = block.cell_size * 2.0
    if n == 0:
        return CellBlock.empty(parent_size)

    pcx = np.floor_divide(block.cx, 2)
    pcy = np.floor_divide(block.cy, 2)
    # Group by parent; children ordered by (cx, cy) within each parent.
    order = np.lexsort((block.cy, block.cx, pcy, pcx))
    pcx_s, pcy_s = pcx[order], pcy[order]

    new_parent = np.empty(n, dtype=bool)
    new_parent[0] = True
    new_parent[1:] = (pcx_s[1:] != pcx_s[:-1]) | (pcy_s[1:] != pcy_s[:-1])
    starts = np.flatnonzero(new_parent)
    parent_of_child = np.cumsum(new_parent) - 1
    pos_in_group = np.arange(n) - starts[parent_of_child]
    n_parents = starts.shape[0]

    rows = {
        "count": block.count[order], "mean": block.mean[order], "m2": block.m2[order],
        "z_min": block.z_min[order], "z_max": block.z_max[order], "hist": block.hist[order],
    }

    first = starts  # position of each parent's first child in sorted order
    acc = {
        "count": rows["count"][first].copy(),
        "mean": rows["mean"][first].copy(),
        "m2": rows["m2"][first].copy(),
        "z_min": rows["z_min"][first].copy(),
        "z_max": rows["z_max"][first].copy(),
        "hist": rows["hist"][first].copy(),
    }
    # Fold remaining children (at most 3 more) in canonical order, vectorised
    # across parents. Sequential in j => identical DAG on every path.
    for j in range(1, 4):
        src = np.flatnonzero(pos_in_group == j)
        if src.size == 0:
            break
        _combine_rows(acc, rows, parent_of_child[src], src)

    return CellBlock(parent_size, pcx_s[first], pcy_s[first],
                     acc["count"], acc["mean"], acc["m2"],
                     acc["z_min"], acc["z_max"], acc["hist"])


def block_to_size(block: CellBlock, target_size: float) -> CellBlock:
    """Aggregate a block to a coarser (or equal) power-of-two lattice size."""
    ratio = target_size / block.cell_size
    k = round(math.log2(ratio)) if ratio > 0 else -1
    if k < 0 or abs(ratio - 2 ** k) > 1e-9:
        raise ValueError(
            f"Cannot aggregate cell size {block.cell_size} to {target_size}: "
            "not a power-of-two multiple.")
    for _ in range(k):
        block = double_block(block)
    return block


def build_block(points_xyz: np.ndarray, labels: np.ndarray,
                cell_size: float, base_size: float) -> CellBlock:
    """Canonical build: base lattice -> target size via dyadic doubling."""
    base = base_block_from_points(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2],
                                  labels, base_size)
    return block_to_size(base, cell_size)


# ---------------------------------------------------------------------------
# Foveated grid: one block per ring
# ---------------------------------------------------------------------------

class FoveatedGrid:
    def __init__(self, ring_config: RingConfig):
        self.ring_config = ring_config
        self.blocks: dict[int, CellBlock] = {}
        self.assigned_points = 0
        self.discarded_points = 0   # AC-C3: never dropped silently

    @property
    def n_cells(self) -> int:
        return sum(len(b) for b in self.blocks.values())

    def memory_bytes(self) -> int:
        return sum(b.memory_bytes() for b in self.blocks.values())


def build_grid(points_xyz: np.ndarray, labels: np.ndarray, ring_config: RingConfig,
               memory_ceiling_bytes: int | None = None) -> FoveatedGrid:
    """Build the variable-resolution grid for one sweep.

    points_xyz: float array [N, 3] in sensor frame. labels: int [N] in {0..3}.
    Points outside all rings are counted as discarded, never silently dropped.
    """
    grid = FoveatedGrid(ring_config)
    if points_xyz.shape[0] == 0:
        return grid

    ring_ids = assign_rings(points_xyz[:, 0], points_xyz[:, 1], ring_config)
    grid.discarded_points = int(np.count_nonzero(ring_ids == OUT_OF_RANGE))
    grid.assigned_points = int(points_xyz.shape[0] - grid.discarded_points)

    for ring in ring_config.rings:
        mask = ring_ids == ring.id
        if not np.any(mask):
            continue
        block = build_block(points_xyz[mask], labels[mask],
                            ring.cell_size, ring_config.base_cell_size)
        grid.blocks[ring.id] = block

    if memory_ceiling_bytes is not None:
        used = grid.memory_bytes()
        if used > memory_ceiling_bytes:
            # Abort loudly. Never silently reduce resolution (spec section 8).
            raise MemoryCeilingExceeded(grid.n_cells, used, memory_ceiling_bytes)
    return grid


def aggregate(grid: FoveatedGrid, to_ring_id: int) -> dict[int, CellBlock]:
    """Aggregate every finer ring block up to the target ring's cell size.

    Returns {ring_id: aggregated block}. Uses the same canonical doubling as
    the native build, so results are bit-identical to building natively at the
    target size (AC-C1).
    """
    target = grid.ring_config.ring_by_id(to_ring_id)
    out = {}
    for ring_id, block in grid.blocks.items():
        if block.cell_size <= target.cell_size:
            out[ring_id] = block_to_size(block, target.cell_size)
    return out


def cells_bit_identical(a: CellBlock, b: CellBlock) -> bool:
    """Bit-identical comparison used by AC-C1."""
    if len(a) != len(b) or a.cell_size != b.cell_size:
        return False
    return (np.array_equal(a.cx, b.cx) and np.array_equal(a.cy, b.cy)
            and np.array_equal(a.count, b.count)
            and np.array_equal(a.mean, b.mean)
            and np.array_equal(a.m2, b.m2)
            and np.array_equal(a.z_min, b.z_min)
            and np.array_equal(a.z_max, b.z_max)
            and np.array_equal(a.hist, b.hist))
