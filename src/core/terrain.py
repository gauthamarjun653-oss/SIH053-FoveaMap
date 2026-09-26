"""Conservative traversability from separate ground/obstacle layers.

Drivable means observed surface passes configured geometric tests, not a safety
certification. Absence of a return never proves clearance or free space.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from src.core.grid import unique_rows
from src.core.semantics import ROAD_BITS, GROUND_BITS, UNASSESSED, DRIVABLE, BLOCKED

@dataclass(frozen=True)
class VehicleLimits:
    max_slope_deg: float = 10.
    max_roughness_m: float = .08
    max_step_m: float = .15
    height_m: float = 1.5
    width_m: float = 1.8
    lateral_margin_m: float = .2
    clearance_margin_m: float = .2
    fit_radius_m: float = 2.5
    tile_size_m: float = 2.
    min_ground_points: int = 8
    min_xy_spread_m: float = .15

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in asdict(self).values()):
            raise ValueError('Vehicle/terrain limits must be finite and positive')
        if self.max_slope_deg>=90 or self.min_ground_points<3:
            raise ValueError('Invalid slope or support count')


def assess_terrain(grid, points, semantics, limits=VehicleLimits(), *, diagnostics=None):
    c=grid.cells; n=len(c)
    # These working arrays are released by the caller; only the decision byte persists.
    slope=np.full(n,np.nan); residual=np.full(n,np.nan); height=np.full(n,np.nan)
    supported=np.zeros(n,bool)
    xy=grid.xy()
    points=np.asarray(points); semantics=np.asarray(semantics)
    road=np.isin(semantics,(9,10))&np.isfinite(points).all(axis=1)
    samples=points[road].astype(np.float64)
    if len(samples)>=limits.min_ground_points:
        # Voxel representatives bound local fit cost and reduce sampling imbalance.
        _,idx=np.unique(np.floor(samples[:,:2]/.15).astype(np.int64),axis=0,return_index=True)
        samples=samples[idx]
        tree=cKDTree(samples[:,:2])
        tiles,inv=unique_rows(np.floor(xy/limits.tile_size_m).astype(np.int64))
        tile_centers=(tiles+.5)*limits.tile_size_m
        order=np.argsort(inv); starts=np.searchsorted(inv[order],np.arange(len(tiles)+1))
        neighbours=tree.query_ball_point(tile_centers,limits.fit_radius_m)
        for j,ids in enumerate(neighbours):
            if len(ids)<limits.min_ground_points: continue
            a=samples[ids]
            origin=tile_centers[j]
            local=a[:,:2]-origin
            if np.linalg.eigvalsh(np.cov(local.T)).min()<limits.min_xy_spread_m**2: continue
            A=np.column_stack([local,np.ones(len(a))]); keep=np.ones(len(a),bool)
            for _ in range(3):
                coef=np.linalg.lstsq(A[keep],a[keep,2],rcond=None)[0]
                err=a[:,2]-A@coef
                scale=max(.025,1.4826*np.median(np.abs(err[keep]-np.median(err[keep]))))
                new=np.abs(err)<=3*scale
                if new.sum()<limits.min_ground_points: break
                keep=new
            coef=np.linalg.lstsq(A[keep],a[keep,2],rcond=None)[0]
            error=a[keep,2]-A[keep]@coef
            rms=float(np.sqrt(np.mean(error**2)))
            # A bad fit is unknown, never assumed flat.
            if rms>.20: continue
            try: hull=ConvexHull(a[keep,:2])
            except QhullError: continue
            rows=order[starts[j]:starts[j+1]]
            inside=np.all(xy[rows]@hull.equations[:,:2].T+hull.equations[:,2]<=1e-7,axis=1)
            rows=rows[inside]
            supported[rows]=True
            slope[rows]=np.degrees(np.arctan(np.hypot(coef[0],coef[1])))
            residual[rows]=rms
            height[rows]=(xy[rows]-origin)@coef[:2]+coef[2]
    c['traversability']=UNASSESSED
    ground=c['ground_count']>0
    mask=c['semantic_mask']
    allowed=(mask&np.uint32(ROAD_BITS))>0
    other_surface=(mask&np.uint32(GROUND_BITS ^ ROAD_BITS))>0
    unknown=(mask&1)>0
    obstacles=np.isfinite(c['obstacle_min'])
    clearance=c['obstacle_min']-height
    roughness=np.maximum(np.sqrt(grid.variance()),residual)
    # Account for legitimate slope inside a cell before classifying a height jump.
    allowance=np.tan(np.radians(slope))*grid.sizes()*np.sqrt(2)/2
    deviation=np.maximum(np.abs(c['ground_min']-height),np.abs(c['ground_max']-height))
    step=np.maximum(deviation-allowance,0)
    geometric=(slope<=limits.max_slope_deg)&(roughness<=limits.max_roughness_m)&(step<=limits.max_step_m)
    obstacle_clear=(~obstacles)|(clearance>=limits.height_m+limits.clearance_margin_m)
    eligible=allowed&ground&supported&~unknown&~other_surface
    c['traversability'][eligible&geometric&obstacle_clear]=DRIVABLE
    known_obstacle=(mask & np.uint32(((1<<20)-1) ^ (GROUND_BITS|1)))!=0
    c['traversability'][other_surface|(known_obstacle&~ground)]=BLOCKED
    c['traversability'][eligible&(~geometric|~obstacle_clear)]=BLOCKED
    # Inflate observed collision evidence by half vehicle width plus a margin.
    # Unsupported elevated-only cells are conservatively treated as obstacles.
    collision=known_obstacle & ((~ground) | (supported & ~obstacle_clear))
    if collision.any():
        distance=cKDTree(xy[collision]).query(xy,k=1)[0]
        footprint_block=(distance<=limits.width_m/2+limits.lateral_margin_m)
        c['traversability'][footprint_block & (c['traversability']==DRIVABLE)]=BLOCKED
    else:
        footprint_block=np.zeros(n,bool)
    # Signed observed height evidence, not object identities or confirmed potholes.
    # Reuse the same support, semantics and slope allowance as traversability.
    rise_evidence=eligible & ((c['ground_max']-height-allowance)>limits.max_step_m)
    depression_evidence=eligible & ((height-c['ground_min']-allowance)>limits.max_step_m)
    # Opt-in diagnostics for explainers/tests; never persist these in normal map records.
    if diagnostics is not None:
        diagnostics.update(expected_ground_z=height, supported=supported,
                           slope_deg=slope, roughness_m=roughness, step_m=step,
                           clearance_m=clearance, step_exceeded=step>limits.max_step_m,
                           slope_exceeded=slope>limits.max_slope_deg,
                           roughness_exceeded=roughness>limits.max_roughness_m)
    # Unknown evidence takes precedence over a 'safe' colour, but not a known obstacle.
    return dict(drivable=int(np.count_nonzero(c['traversability']==DRIVABLE)),
                blocked=int(np.count_nonzero(c['traversability']==BLOCKED)),
                unknown=int(np.count_nonzero(c['traversability']==UNASSESSED)),
                observed_rise_cells=int(rise_evidence.sum()),
                observed_depression_cells=int(depression_evidence.sum()),
                height_evidence_meaning='supported road-cell height deviations, not curb/pothole object detections',
                supported_cells=int(supported.sum()), footprint_blocked_cells=int(footprint_block.sum()), limits=asdict(limits),
                meaning='observed-surface assessment; unobserved space remains unknown')


def load_core_settings(path):
    import yaml
    with open(path) as stream: config=yaml.safe_load(stream)
    limits=VehicleLimits(**config.get('vehicle',{}))
    budget=float(config.get('map_memory_mb',8.))*1e6
    if not np.isfinite(budget) or budget<=0: raise ValueError('Invalid map memory budget')
    return limits,int(budget)
