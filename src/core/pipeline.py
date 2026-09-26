"""Primary PS pipeline. Experimental pothole inference is intentionally excluded."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from src.core.grid import build_semantic_grid
from src.core.motion import MotionTracker
from src.core.terrain import VehicleLimits, assess_terrain
from src.core.semantics import PALETTE, MOVABLE_BITS, MOVING, STATIONARY, DRIVABLE, BLOCKED

@dataclass
class CoreResult:
    grid: object
    terrain: dict
    motion: dict
    point_motion: np.ndarray
    timings: dict
    semantic_source: str

class CorePipeline:
    def __init__(self,rings,*,limits=VehicleLimits(),memory_ceiling_bytes=8_000_000):
        self.rings=rings; self.limits=limits; self.ceiling=memory_ceiling_bytes
        self.tracker=MotionTracker()

    def process(self,points,semantic,*,pose=None,timestamp=None,semantic_source='unknown'):
        points=np.asarray(points); semantic=np.asarray(semantic)
        if points.ndim!=2 or points.shape[1]!=3 or semantic.shape!=(len(points),):
            raise ValueError('Expected points [N,3], semantic [N]')
        if np.any((semantic<0)|(semantic>=20)): raise ValueError('Expected detailed IDs 0..19')
        start=perf_counter()
        radius=np.hypot(points[:,0],points[:,1])
        keep=np.isfinite(points).all(axis=1)&(radius>=self.rings.radius_min)&(radius<=self.rings.radius_max)
        motion,mreport=self.tracker.update(points[keep],semantic[keep],pose,timestamp)
        point_motion=np.zeros(len(points),np.uint8); point_motion[keep]=motion
        tracked=perf_counter()
        grid=build_semantic_grid(points,semantic,self.rings,point_motion,memory_ceiling_bytes=self.ceiling)
        built=perf_counter()
        terrain=assess_terrain(grid,points[keep],semantic[keep],self.limits)
        end=perf_counter()
        return CoreResult(grid,terrain,mreport,point_motion,
                    dict(motion_ms=(tracked-start)*1000,grid_ms=(built-tracked)*1000,
                         terrain_ms=(end-built)*1000,core_ms=(end-start)*1000),semantic_source)


def display_colors(grid,mode='terrain'):
    c=grid.cells
    colors=PALETTE[c['semantic']].copy()
    if mode=='terrain':
        colors[:]=(.4,.43,.48)
        colors[c['traversability']==DRIVABLE]=(.18,.55,.95)
        colors[c['traversability']==BLOCKED]=(.95,.66,.19)
    elif mode!='semantic': raise ValueError('Unknown view mode')
    movable=(c['semantic_mask']&np.uint32(MOVABLE_BITS))!=0
    colors[movable&(c['motion']==0)]=(.72,.60,.88)
    colors[movable&(c['motion']==STATIONARY)]=(.20,.82,.90)
    colors[movable&(c['motion']==MOVING)]=(1.,.32,.26)
    return np.column_stack([colors,np.ones(len(c))])
