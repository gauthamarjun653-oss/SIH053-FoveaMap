"""Step 5 — compact cell records and direct construction.

Two independent changes that the priority list treated as one. Keeping them
apart matters, because they break different guarantees:

  * **Compact records** (84 -> 56 bytes) change how many bits a finished value
    is stored in. Counts stay exact; heights lose precision.
  * **Direct construction** changes the *order* the same values are summed in.
    It breaks bit-identity at any precision, float32 or float64, because
    floating-point addition is not associative. The cascade's bit-identity was
    never a precision property — it was an ordering property.

So `cells_bit_identical` cannot be the contract for direct construction, and
weakening it to a tolerance would quietly weaken it for the cascade too. The
contract here is stated per field instead:

    count    exact      integer addition
    hist     exact      integer addition
    z_min    exact      min() is associative and exact
    z_max    exact      max() is associative and exact
    mean     bounded    float32 storage + reordered summation
    m2/var   bounded    same, and variance is the sensitive one

`compare_blocks` measures the bounded fields rather than asserting them, so a
regression shows up as a number.

Accumulation stays float64 throughout and narrows only at the end. Smaller
stored records do not mean smaller working memory — that is why the comparison
script reports stored bytes and peak RSS separately and never adds them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import RingConfig
from src.common.errors import MemoryCeilingExceeded
from src.grid.foveated_grid import (N_CLASSES, CellBlock, FoveatedGrid,
                                    base_block_from_points)
from src.grid.rings import OUT_OF_RANGE, assign_rings

# int32 lattice indices. At the 0.05 m base cell this allows +-107 km, so the
# bound is never reached in practice — but it is checked, not assumed, because
# silent wraparound would corrupt cell identity rather than raise.
_COORD_MIN, _COORD_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max


class CompactCellBlock:
    """56 bytes per occupied cell.

    cx, cy, count            3 x 4 = 12
    mean, m2, z_min, z_max   4 x 4 = 16
    hist                     4 x 4 = 16
    z_var, occ, neg          3 x 4 = 12
                                     56
    """

    __slots__ = ("cell_size", "cx", "cy", "count", "mean", "m2",
                 "z_min", "z_max", "hist",
                 "z_var", "occupancy_conf", "neg_obstacle_conf")

    BYTES_PER_CELL = 56

    def __init__(self, cell_size, cx, cy, count, mean, m2, z_min, z_max, hist,
                 z_var=None, occupancy_conf=None, neg_obstacle_conf=None):
        self.cell_size = float(cell_size)
        self.cx = cx
        self.cy = cy
        self.count = count
        self.mean = mean
        self.m2 = m2
        self.z_min = z_min
        self.z_max = z_max
        self.hist = hist
        self.z_var = z_var
        self.occupancy_conf = occupancy_conf
        self.neg_obstacle_conf = neg_obstacle_conf

    def __len__(self) -> int:
        return int(self.cx.shape[0])

    def memory_bytes(self) -> int:
        total = 0
        for name in self.__slots__[1:]:
            arr = getattr(self, name)
            if arr is not None:
                total += arr.nbytes
        return total


class CoordinateOverflow(ValueError):
    """Lattice index outside int32. Raised, never wrapped silently."""


def pack_block(block: CellBlock, fill_derived: bool = True) -> CompactCellBlock:
    """Narrow a finished float64 block to the 56-byte layout.

    Call after `compute_stats`, so derived layers exist and the packed block is
    the finished record rather than the 72-byte intermediate the historical
    3.30 MB benchmark actually measured.
    """
    if len(block):
        lo = int(min(block.cx.min(), block.cy.min()))
        hi = int(max(block.cx.max(), block.cy.max()))
        if lo < _COORD_MIN or hi > _COORD_MAX:
            raise CoordinateOverflow(
                f"Lattice index range [{lo}, {hi}] exceeds int32 at cell size "
                f"{block.cell_size}.")

    for field in (block.count, block.hist):
        if field.size and (np.any(field < 0) or np.any(field > np.iinfo(np.uint32).max)):
            raise OverflowError("Compact counts/histograms must fit uint32")
    n = len(block)
    z = (lambda a, dt: (a.astype(dt) if a is not None
                        else np.zeros(n, dtype=dt) if fill_derived else None))
    return CompactCellBlock(
        block.cell_size,
        block.cx.astype(np.int32), block.cy.astype(np.int32),
        block.count.astype(np.uint32),
        block.mean.astype(np.float32), block.m2.astype(np.float32),
        block.z_min.astype(np.float32), block.z_max.astype(np.float32),
        block.hist.astype(np.uint32),
        z(block.z_var, np.float32),
        z(block.occupancy_conf, np.float32),
        z(block.neg_obstacle_conf, np.float32),
    )


def build_block_direct(points_xyz: np.ndarray, labels: np.ndarray,
                       cell_size: float) -> CellBlock:
    """Accumulate straight onto the target lattice — no dyadic cascade.

    `base_block_from_points` is already generic over cell size, so the cascade
    was a choice rather than a necessity: a point at 70 m currently passes
    through 5, 10 and 20 cm intermediate grids to reach its 40 cm cell. This
    skips them. Measured cost of the cascade is real — the foveated build is
    slower than a uniform 5 cm build at every percentile, because the uniform
    build does no doubling.
    """
    return base_block_from_points(points_xyz[:, 0], points_xyz[:, 1],
                                  points_xyz[:, 2], labels, cell_size)


def build_grid_direct(points_xyz: np.ndarray, labels: np.ndarray,
                      ring_config: RingConfig) -> FoveatedGrid:
    """Build every ring natively at its own cell size."""
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
        grid.blocks[ring.id] = build_block_direct(points_xyz[mask], labels[mask],
                                                  ring.cell_size)
    return grid


@dataclass
class BlockComparison:
    """Per-field verdict. Exact fields are booleans; approximate ones carry
    the measured error, so a tolerance can be chosen from evidence."""
    same_cells: bool
    count_exact: bool
    hist_exact: bool
    z_min_exact: bool
    z_max_exact: bool
    max_abs_mean_err: float
    max_abs_var_err: float
    max_rel_var_err: float
    n_cells: int

    @property
    def counts_preserved(self) -> bool:
        """The non-negotiable part: no return may be lost or duplicated."""
        return self.count_exact and self.hist_exact

    def within(self, mean_tol: float, var_tol: float) -> bool:
        return (self.same_cells and self.counts_preserved
                and self.z_min_exact and self.z_max_exact
                and self.max_abs_mean_err <= mean_tol
                and self.max_abs_var_err <= var_tol)


def _variance(count, m2):
    c = np.asarray(count, dtype=np.float64)
    return np.where(c > 1, np.asarray(m2, dtype=np.float64) / np.maximum(c, 1), 0.0)


def compare_blocks(a, b) -> BlockComparison:
    """Compare two blocks of the same cell size, matching cells by (cx, cy).

    Order-independent: the two builders emit cells in the same sorted order
    today, but a comparison that assumed so would silently pass if one changed.
    """
    ka = np.stack([np.asarray(a.cx, np.int64), np.asarray(a.cy, np.int64)], 1)
    kb = np.stack([np.asarray(b.cx, np.int64), np.asarray(b.cy, np.int64)], 1)
    oa = np.lexsort((ka[:, 1], ka[:, 0]))
    ob = np.lexsort((kb[:, 1], kb[:, 0]))
    same_cells = a.cell_size == b.cell_size and np.array_equal(ka[oa], kb[ob])
    if not same_cells:
        return BlockComparison(False, False, False, False, False,
                               float("inf"), float("inf"), float("inf"),
                               int(len(ka)))

    if not len(ka):
        return BlockComparison(True, True, True, True, True, 0., 0., 0., 0)

    va = _variance(np.asarray(a.count)[oa], np.asarray(a.m2)[oa])
    vb = _variance(np.asarray(b.count)[ob], np.asarray(b.m2)[ob])
    denom = np.maximum(np.abs(va), 1e-12)
    return BlockComparison(
        same_cells=True,
        count_exact=np.array_equal(np.asarray(a.count)[oa].astype(np.int64),
                                   np.asarray(b.count)[ob].astype(np.int64)),
        hist_exact=np.array_equal(np.asarray(a.hist)[oa].astype(np.int64),
                                  np.asarray(b.hist)[ob].astype(np.int64)),
        z_min_exact=np.array_equal(np.asarray(a.z_min, np.float32)[oa],
                                   np.asarray(b.z_min, np.float32)[ob]),
        z_max_exact=np.array_equal(np.asarray(a.z_max, np.float32)[oa],
                                   np.asarray(b.z_max, np.float32)[ob]),
        max_abs_mean_err=float(np.max(np.abs(
            np.asarray(a.mean, np.float64)[oa] - np.asarray(b.mean, np.float64)[ob]))),
        max_abs_var_err=float(np.max(np.abs(va - vb))),
        max_rel_var_err=float(np.max(np.abs(va - vb) / denom)),
        n_cells=int(len(ka)),
    )


def build_compact_grid(points_xyz, labels, ring_config, image_width=2048,
                       memory_ceiling_bytes=None):
    """Build and pack one ring at a time, releasing wide intermediates.

    The ceiling bounds retained map payload, not process working memory.
    Canonical build_grid remains available with its bit-identity contract.
    """
    from src.analytics.cell_stats import compute_stats
    points_xyz = np.asarray(points_xyz)
    labels = np.asarray(labels)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3 or labels.shape != (len(points_xyz),):
        raise ValueError("Expected Nx3 points and N labels")
    if not np.isfinite(points_xyz).all():
        raise ValueError("Grid points must be finite")
    grid = FoveatedGrid(ring_config)
    ids = assign_rings(points_xyz[:, 0], points_xyz[:, 1], ring_config)
    grid.discarded_points = int(np.count_nonzero(ids == OUT_OF_RANGE))
    grid.assigned_points = len(points_xyz) - grid.discarded_points
    for ring in ring_config.rings:
        mask = ids == ring.id
        if not mask.any():
            continue
        working = FoveatedGrid(ring_config)
        working.blocks[ring.id] = build_block_direct(points_xyz[mask], labels[mask], ring.cell_size)
        compute_stats(working, image_width)
        grid.blocks[ring.id] = pack_block(working.blocks[ring.id])
        del working
        if memory_ceiling_bytes is not None and grid.memory_bytes() > memory_ceiling_bytes:
            raise MemoryCeilingExceeded(grid.n_cells, grid.memory_bytes(), memory_ceiling_bytes)
    return grid
