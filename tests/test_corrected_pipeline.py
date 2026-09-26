"""Regression cases from the September review; canonical tests stay intact."""
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from src.analytics.ground import (GroundField, GroundPlane, build_ground_field,
                                  SUPPORTED, EXTRAPOLATED)
from src.analytics.visibility import (compute_visibility, _BlindZone,
                                      _OccupancyProximity, EXPECTED_EMPTY)
from src.analytics.negative_obs2 import detect, _below_plane_candidates, _components
from src.analytics.cell_stats import compute_stats
from src.common.config import RingConfig, RingSpec, load_rings
from src.common.errors import MemoryCeilingExceeded
from src.grid.compact import build_compact_grid, compare_blocks, pack_block
from src.grid.foveated_grid import build_grid, CellBlock
from src.segmentation.class_map import ROAD, STATIC
from src.frame_pipeline import process_frame

RINGS = load_rings('configs/rings.yaml')
SINGLE = RingConfig(.05, (RingSpec(1, 0., 30., .05),))

def supported_plane(tile, z):
    x, y = np.asarray(tile)*5.
    eq = ConvexHull([[x,y],[x+5,y],[x,y+5],[x+5,y+5]]).equations
    return GroundPlane(tile, 0., 0., z, 10, 0., eq)

def test_oscillating_ray_is_rejected():
    ground = GroundField({(0,0): supported_plane((0,0), -1.2),
                          (1,0): supported_plane((1,0), -.8)}, 5., (0.,0.,-.8))
    grid = build_grid(np.empty((0,3)), np.empty(0, dtype=int), SINGLE)
    result = compute_visibility(grid, ground, SINGLE, None,
                  elevations=np.array([np.arctan(-.2)]), image_width=1,
                  max_refine_iters=4)
    assert result.x.size == 0
    assert result.inconsistent == 1


def test_settled_ray_matches_its_plane():
    ground = GroundField({(0,0): supported_plane((0,0), -.8)}, 5., (0.,0.,-.8))
    grid = build_grid(np.empty((0,3)), np.empty(0, dtype=int), SINGLE)
    v = compute_visibility(grid, ground, SINGLE, None,
                           elevations=np.array([np.arctan(-.2)]), image_width=1)
    np.testing.assert_allclose(v.z_ground, -.2*v.x)
    assert v.inconsistent == 0 and v.state.tolist() == [EXPECTED_EMPTY]


def test_blind_zone_needs_independent_mask():
    assert not _BlindZone().blind(np.array([4.]), np.array([0.])).any()
    assert _BlindZone(np.full(360, 2.)).blind(np.array([1., 4.]), np.array([0.,0.])).tolist() == [True, False]
    with pytest.raises(ValueError):
        _BlindZone(np.array([-1.]))


def test_proximity_uses_distance_not_bucket_membership():
    grid = build_grid(np.array([[8.01,.01,-1.7]]), np.array([ROAD]), SINGLE)
    prox = _OccupancyProximity(grid, extent=30)
    assert not prox.occupied_near(np.array([8.24]), np.array([.14]), .05).any()
    assert prox.occupied_near(np.array([8.03]), np.array([.03]), .05).all()


def test_raw_return_height_and_distance_required():
    pts = np.array([[8.,0.,1.], [8.03,0.,-1.7]])
    grid = build_grid(pts, np.full(2, ROAD), SINGLE)
    prox = _OccupancyProximity(grid, 30, points_xyz=pts)
    assert not prox.occupied_near(np.array([8.]), np.array([0.]), .01, [-1.7], .03).any()
    assert prox.occupied_near(np.array([8.]), np.array([0.]), .05, [-1.7], .03).all()


def patch_points():
    x,y = np.meshgrid(np.arange(5.01, 7.01, .1), np.arange(.01, 2.01, .1))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -1.7)])


def test_ground_rejects_collinear_samples():
    pts=np.array([[5.01+i*.1,.01,-1.7] for i in range(10)])
    ground=build_ground_field(build_grid(pts, np.full(len(pts),ROAD), SINGLE))
    assert not ground.planes


def test_ground_support_stops_at_own_hull():
    pts=patch_points()
    ground=build_ground_field(build_grid(pts, np.full(len(pts),ROAD), SINGLE))
    _, state=ground.height_and_state(np.array([6., 8.]), np.array([1.,1.]))
    assert state.tolist() == [SUPPORTED, EXTRAPOLATED]


def test_adjacent_tile_support_does_not_leak():
    p0=supported_plane((0,0), -1.7)
    p1=GroundPlane((1,0), 0.,0.,-1.7,10,0.,
                   ConvexHull([[8,1],[9,1],[8,2],[9,2]]).equations)
    ground=GroundField({(0,0):p0,(1,0):p1},5.,(0.,0.,-1.7))
    _,state=ground.height_and_state(np.array([4.9,5.1,8.5]),np.array([1.5]*3))
    assert state.tolist()==[SUPPORTED,EXTRAPOLATED,SUPPORTED]


def test_mixed_obstacle_cells_do_not_fit_ground():
    pts=patch_points()
    grid=build_grid(np.concatenate([pts,pts]),
                    np.r_[np.full(len(pts),ROAD),np.full(len(pts),STATIC)],SINGLE)
    assert not build_ground_field(grid).planes


def test_compact_below_plane_keys_do_not_overflow():
    pts=np.array([[4.01,.01,-2.]])
    grid=build_compact_grid(pts,np.array([ROAD]),SINGLE)
    ground=GroundField({(0,0):supported_plane((0,0),-1.7)},5.,(0.,0.,-1.7))
    candidates, _, _ = _below_plane_candidates(grid,ground,.15,True,30)
    assert len(candidates)==1 and candidates[0].depth > .29
    labels=_components(np.array([0,1]),np.array([80,81],dtype=np.int32),np.array([0,0],dtype=np.int32))
    assert labels.tolist()==[0,0]


def test_compact_preserves_counts_and_numeric_bounds():
    rng=np.random.default_rng(4)
    xyz=rng.uniform([-80,-80,-3],[80,80,3],size=(10000,3)).astype(np.float32)
    labels=rng.integers(0,4,len(xyz))
    wide=compute_stats(build_grid(xyz,labels,RINGS))
    packed=build_compact_grid(xyz,labels,RINGS)
    assert packed.memory_bytes()==56*packed.n_cells
    assert set(wide.blocks)==set(packed.blocks)
    for rid,b in wide.blocks.items():
        assert compare_blocks(b,packed.blocks[rid]).within(1e-3,1e-4)


def test_empty_compact_comparison_and_cell_size_mismatch():
    a=CellBlock.empty(.05)
    b=pack_block(a)
    assert compare_blocks(a,b).within(0,0)
    b.cell_size=.1
    assert not compare_blocks(a,b).same_cells


def test_packing_rejects_count_overflow():
    g=build_grid(np.array([[4.,0.,-1.7]]),np.array([ROAD]),SINGLE)
    b=g.blocks[1]
    b.count[0]=2**32
    with pytest.raises(OverflowError):
        pack_block(b)


def test_compact_ceiling_includes_derived_arrays():
    with pytest.raises(MemoryCeilingExceeded):
        build_compact_grid(np.array([[4.,0.,-1.7]]),np.array([ROAD]),SINGLE,
                            memory_ceiling_bytes=55)


def test_repeat_detection_clears_old_confidence():
    pts=patch_points()
    grid=build_compact_grid(pts,np.full(len(pts),ROAD),SINGLE)
    for b in grid.blocks.values():
        b.neg_obstacle_conf.fill(1.)
    detect(grid,SINGLE,depth_threshold=10.,image_width=32,points_xyz=pts)
    assert all(not b.neg_obstacle_conf.any() for b in grid.blocks.values())


def test_combined_pipeline_and_viewer_geometry():
    from src.viz.rerun_viewer import _grid_rects
    pts=patch_points()
    result=process_frame(pts,np.full(len(pts),ROAD),SINGLE,image_width=32,experimental_hazards=True)
    assert result.grid.memory_bytes()==56*result.grid.n_cells
    assert result.build_ms>0 and result.detection_ms>0
    assert _grid_rects(result.grid,1)[0].shape==(result.grid.n_cells,2)


def test_viewer_clears_hazards_on_empty_next_frame(monkeypatch):
    import src.viz.rerun_viewer as viewer
    from src.viz.telemetry import Telemetry
    class RR:
        def __init__(self): self.cleared=[]
        def set_time(self,*a,**k): pass
        def TextDocument(self,*a): return None
        def Clear(self,**kw): return 'clear'
        def log(self,path,value):
            if value=='clear': self.cleared.append(path)
    rr=RR()
    monkeypatch.setattr(viewer,'_require_rerun',lambda:rr)
    grid=build_compact_grid(np.empty((0,3)),np.empty(0,dtype=int),SINGLE)
    viewer.log_frame(grid,Telemetry(),1,SINGLE,show_rings=False,neg_candidates=[])
    assert len(rr.cleared)==len(viewer.NEG_CONF_TIERS)
