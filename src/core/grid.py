"""40-byte occupied leaf records on a disjoint, distance-adaptive dyadic grid.

Conservatively split an entire parent if any part touches a finer radius.
Thus no coarse leaf can overlap a finer leaf, even at circular ring boundaries.
Input counts and semantic presence are retained. Within-cell XY geometry and
per-class frequency histograms are deliberately compressed, not 'lossless 3D'.
"""
from dataclasses import dataclass
import numpy as np
from src.common.errors import MemoryCeilingExceeded
from src.core.semantics import GROUND, MOVABLE, MOVING, STATIONARY

CELL_DTYPE = np.dtype([
 ('ix','<i2'),('iy','<i2'),('level','u1'),('semantic','u1'),('motion','u1'),('traversability','u1'),
 ('count','<u2'),('ground_count','<u2'),('ground_z','<f4'),('ground_m2','<f4'),
 ('ground_min','<f4'),('ground_max','<f4'),('obstacle_min','<f4'),('obstacle_max','<f4'),
 ('semantic_mask','<u4')])
assert CELL_DTYPE.itemsize == 40

@dataclass
class SemanticGrid:
    cells: np.ndarray
    base: float
    assigned_points: int
    discarded_points: int

    @property
    def n_cells(self): return len(self.cells)
    def memory_bytes(self): return self.cells.nbytes
    def sizes(self): return self.base * (2. ** self.cells['level'])
    def xy(self):
        half = self.sizes()*.5
        return np.column_stack([self.cells['ix']*self.base+half,self.cells['iy']*self.base+half])
    def xyz(self):
        z=np.where(self.cells['ground_count']>0,self.cells['ground_z'],self.cells['obstacle_min'])
        return np.column_stack([self.xy(),z])
    def variance(self):
        c=self.cells
        return c['ground_m2']/np.maximum(c['ground_count'],1)


def unique_rows(values):
    """Integer rows via lexsort, avoiding NumPy's slower structured comparison sort."""
    if not len(values): return values.copy(), np.empty(0,np.int64)
    order=np.lexsort(tuple(values[:,j] for j in range(values.shape[1]-1,-1,-1)))
    sorted_values=values[order]
    first=np.r_[True,np.any(sorted_values[1:]!=sorted_values[:-1],axis=1)]
    inverse=np.empty(len(values),np.int64)
    inverse[order]=np.cumsum(first)-1
    return sorted_values[first],inverse


def leaf_keys(xy, rings, uniform=False):
    base=rings.base_cell_size
    fine=np.floor(np.asarray(xy,dtype=np.float64)/base).astype(np.int64)
    max_level=int(round(np.log2(rings.rings[-1].cell_size/base)))
    if max_level>15: raise ValueError('Unsupported grid depth')
    level=np.full(len(fine),0 if uniform else max_level,dtype=np.uint8)
    # The partition depends only on position/configuration, never point density.
    for k in range(max_level,0,-1):
        size=base*2**k
        lower=(fine//(2**k))*(2**k)*base
        upper=lower+size
        closest=np.maximum(np.maximum(lower,-upper),0.)
        distance=np.linalg.norm(closest,axis=1)
        # Radius where resolution must be finer than this level.
        thresholds=[r.radius_max for r in rings.rings if r.cell_size<size-1e-9]
        threshold=max(thresholds,default=-1.)
        split=(level==k)&(distance<=threshold)
        level[split]-=1
    scale=2**level.astype(np.int64)
    origin=(fine//scale[:,None])*scale[:,None]
    if origin.size and (origin.min()<np.iinfo(np.int16).min or origin.max()>np.iinfo(np.int16).max):
        raise OverflowError('Local grid coordinates exceed int16; change bounds or use wider records')
    return np.column_stack([level,origin])


def build_semantic_grid(points, semantic, rings, motion=None, *, uniform=False, memory_ceiling_bytes=8_000_000):
    points=np.asarray(points); semantic=np.asarray(semantic)
    if points.ndim!=2 or points.shape[1]!=3 or semantic.shape!=(len(points),):
        raise ValueError('Expected points [N,3] and semantic [N]')
    if np.any((semantic<0)|(semantic>=20)) or np.any(semantic!=semantic.astype(int)):
        raise ValueError('Expected detailed semantic IDs 0..19')
    if motion is None: motion=np.zeros(len(points),np.uint8)
    motion=np.asarray(motion)
    if motion.shape!=(len(points),) or np.any((motion<0)|(motion>2)):
        raise ValueError('Expected motion [N] IDs 0..2')
    valid=np.isfinite(points).all(axis=1)
    radius=np.hypot(points[:,0],points[:,1])
    valid&=(radius>=rings.radius_min)&(radius<=rings.radius_max)
    p=points[valid].astype(np.float64); sem=semantic[valid].astype(np.uint8); mot=motion[valid]
    if not len(p): return SemanticGrid(np.empty(0,CELL_DTYPE),rings.base_cell_size,0,len(points))
    keys=leaf_keys(p[:,:2],rings,uniform)
    keys,inverse=unique_rows(keys)
    n=len(keys)
    if n*CELL_DTYPE.itemsize>memory_ceiling_bytes:
        raise MemoryCeilingExceeded(n, n*CELL_DTYPE.itemsize, memory_ceiling_bytes)
    count=np.bincount(inverse,minlength=n)
    if count.max()>65535: raise OverflowError('Per-cell count exceeds uint16; widen records instead of wrapping')
    cells=np.zeros(n,CELL_DTYPE)
    cells['level'],cells['ix'],cells['iy']=keys.T
    cells['count']=count
    # Count only observed (cell, class) pairs. Preserve minority classes in a mask.
    pairs,pair_count=np.unique(inverse*20+sem,return_counts=True)
    pair_cell=pairs//20
    max_count=np.zeros(n,np.int64)
    np.maximum.at(max_count,pair_cell,pair_count)
    tied=pair_count==max_count[pair_cell]
    cells['semantic'].fill(19)
    np.minimum.at(cells['semantic'],pair_cell[tied],(pairs[tied]%20).astype(np.uint8))
    np.bitwise_or.at(cells['semantic_mask'],inverse,np.left_shift(np.uint32(1),sem.astype(np.uint32)))
    ground=np.isin(sem,GROUND)
    gi=inverse[ground]; z=p[ground,2]
    gc=np.bincount(gi,minlength=n)
    cells['ground_count']=gc
    cells['ground_z']=np.nan
    means=np.bincount(gi,weights=z,minlength=n)/np.maximum(gc,1)
    cells['ground_z'][gc>0]=means[gc>0]
    cells['ground_m2']=np.bincount(gi,weights=(z-means[gi])**2,minlength=n)
    for field,values,idx,fn,init in [
      ('ground_min',z,gi,np.minimum,np.inf),('ground_max',z,gi,np.maximum,-np.inf),
      ('obstacle_min',p[~ground,2],inverse[~ground],np.minimum,np.inf),
      ('obstacle_max',p[~ground,2],inverse[~ground],np.maximum,-np.inf)]:
        cells[field].fill(init); fn.at(cells[field],idx,values.astype(np.float32))
    movable=np.isin(sem,MOVABLE)
    movcount=np.bincount(inverse[movable],minlength=n)
    stationary=np.bincount(inverse[movable&(mot==STATIONARY)],minlength=n)
    moving=np.bincount(inverse[movable&(mot==MOVING)],minlength=n)
    cells['motion'][(movcount>0)&(stationary==movcount)]=STATIONARY
    cells['motion'][moving>0]=MOVING
    return SemanticGrid(cells,rings.base_cell_size,int(valid.sum()),int((~valid).sum()))
