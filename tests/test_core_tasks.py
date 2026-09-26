import numpy as np
import pytest
from src.common.config import RingConfig, RingSpec, load_rings
from src.common.errors import MemoryCeilingExceeded
from src.core.grid import build_semantic_grid, CELL_DTYPE
from src.core.semantics import detailed_labels, MOVING, STATIONARY, DRIVABLE, BLOCKED
from src.core.terrain import assess_terrain, VehicleLimits
from src.core.motion import MotionTracker
from src.core.pipeline import CorePipeline

RINGS=load_rings('configs/rings.yaml')
SINGLE=RingConfig(.05,(RingSpec(1,0.,100.,.05),))

def patch(slope=0.,height=-1.7,label=9):
    x,y=np.meshgrid(np.arange(4.01,7.,.08),np.arange(.01,3.,.08))
    p=np.column_stack([x.ravel(),y.ravel(),height+slope*(x.ravel()-5.5)])
    return p,np.full(len(p),label,np.uint8)

def test_preserve_semantics_without_motion_leakage():
    assert detailed_labels([10,252,40,48,72,71,80]).tolist()==[1,1,9,11,17,16,18]

def test_sparse_40bytes_and_ground_obstacle_separation():
    p=np.array([[4.01,.01,-1.7],[4.02,.02,-1.6],[4.01,.01,1.]])
    g=build_semantic_grid(p,np.array([9,9,13]),SINGLE)
    c=g.cells[0]
    assert g.memory_bytes()==40 and c['count']==3 and c['ground_count']==2
    assert c['ground_z']==pytest.approx(-1.65)
    assert c['obstacle_min']==1.
    assert c['semantic_mask']==(1<<9)|(1<<13)
    assert g.variance()[0]==pytest.approx(.0025)

def test_no_overlapping_leaves_at_radial_boundaries():
    rng=np.random.default_rng(42)
    angle=rng.uniform(-np.pi,np.pi,30000)
    radius=rng.choice([10.,25.,50.],len(angle))+rng.uniform(-.8,.8,len(angle))
    p=np.column_stack([radius*np.cos(angle),radius*np.sin(angle),np.zeros(len(angle))])
    g=build_semantic_grid(p,np.full(len(p),9),RINGS)
    keys={(int(c['level']),int(c['ix']),int(c['iy'])) for c in g.cells}
    for c in g.cells:
        for parent in range(int(c['level'])+1,4):
            scale=2**parent
            assert (parent,int(c['ix'])//scale*scale,int(c['iy'])//scale*scale) not in keys
    assert int(g.cells['count'].sum())==len(p)

def test_invalid_points_and_unknown_are_accounted():
    g=build_semantic_grid(np.array([[4,0,0],[np.nan,0,0],[110,0,0],[1,0,0]]),np.array([0,9,9,9]),RINGS)
    assert g.assigned_points==1 and g.discarded_points==3 and g.cells[0]['semantic_mask']==1

def test_budget_and_counter_overflow_fail_explicitly():
    with pytest.raises(MemoryCeilingExceeded):build_semantic_grid(np.array([[4,0,0]]),np.array([9]),RINGS,memory_ceiling_bytes=39)
    with pytest.raises(OverflowError):build_semantic_grid(np.tile([4,0,0],(65536,1)),np.full(65536,9),RINGS)

def test_minority_obstacle_preserved():
    p=np.tile([4.01,.01,0.],(100,1));sem=np.full(100,9);sem[-1]=6
    g=build_semantic_grid(p,sem,SINGLE)
    assert g.cells[0]['semantic']==9
    assert g.cells[0]['semantic_mask']&(1<<6)

def test_road_drivable_sidewalk_blocked_and_unknown_unknown():
    p,s=patch();g=build_semantic_grid(p,s,SINGLE)
    result=assess_terrain(g,p,s)
    assert result['drivable']>len(g.cells)*.5
    side=build_semantic_grid(p,np.full(len(p),11),SINGLE)
    assert assess_terrain(side,p,np.full(len(p),11))['blocked']==len(side.cells)
    unknown=build_semantic_grid(p,np.zeros(len(p),int),SINGLE)
    # Unknown returns must not acquire a drivable label.
    assert assess_terrain(unknown,p,np.zeros(len(p),int))['drivable']==0

def test_steep_road_is_not_drivable():
    p,s=patch(slope=.5);g=build_semantic_grid(p,s,SINGLE)
    result=assess_terrain(g,p,s)
    assert result['drivable']==0 and result['blocked']>0

def test_low_overhang_blocks_but_high_overhang_retains_ground():
    p,s=patch()
    for height,expect_drivable in [(-.3,False),(1.,True)]:
        q=p.copy();q[:,2]=height
        pts=np.r_[p,q];sem=np.r_[s,np.full(len(q),15)]
        g=build_semantic_grid(pts,sem,SINGLE);r=assess_terrain(g,pts,sem)
        assert (r['drivable']>len(g.cells)*.4)==expect_drivable
        np.testing.assert_allclose(g.cells['ground_z'],-1.7)

def test_lone_ground_point_not_drivable():
    p=np.array([[5.,1.,-1.7]]);s=np.array([9]);g=build_semantic_grid(p,s,SINGLE)
    assert assess_terrain(g,p,s)['unknown']==1

def test_depression_step_is_blocked():
    p,s=patch();hole=(np.abs(p[:,0]-5.5)<.25)&(np.abs(p[:,1]-1.5)<.25)
    p[hole,2]-=.35
    g=build_semantic_grid(p,s,SINGLE);assess_terrain(g,p,s)
    low=g.cells['ground_z']<-1.9
    assert np.all(g.cells['traversability'][low]!=DRIVABLE)

@pytest.mark.parametrize('speed,expected',[(0.,STATIONARY),(2.,MOVING)])
def test_motion_compensates_ego_and_tracks_actual_motion(speed,expected):
    rng=np.random.default_rng(8); shape=rng.uniform([-.4,-.3,-.4],[.4,.3,.4],(150,3))
    tracker=MotionTracker()
    for i in range(10):
        t=i*.1; world=shape+[8+speed*t,2,0]
        pose=np.eye(4);pose[:3,3]=[3*t,0,0]
        a=.1*t;pose[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
        local=(world-pose[:3,3])@pose[:3,:3]
        labels,report=tracker.update(local,np.ones(len(local),np.uint8),pose,t)
        if i==0:assert not labels.any()
    assert np.mean(labels==expected)>.95
    assert len(tracker.tracks)<=tracker.max_tracks

def test_tracker_resets_on_seek_and_missing_pose():
    p=np.random.default_rng(3).normal(0,.1,(30,3))+[5,0,0];s=np.ones(30,int)
    tr=MotionTracker()
    for i in range(8):tr.update(p,s,np.eye(4),i*.1)
    out,_=tr.update(p,s,np.eye(4),.1);assert not out.any()
    out,_=tr.update(p,s,None,None);assert not out.any() and not tr.tracks

def test_pipeline_has_no_pothole_dependency():
    p,s=patch();r=CorePipeline(SINGLE).process(p,s,semantic_source='synthetic')
    assert r.semantic_source=='synthetic' and 'hazard_ms' not in r.timings
    assert r.grid.memory_bytes()==40*r.grid.n_cells

def test_motion_expiry_and_capacity_bound():
    rng=np.random.default_rng(2)
    shape=rng.uniform(-.1,.1,(20,3))
    p=np.concatenate([shape+[i*2,5,0] for i in range(12)])
    s=np.ones(len(p),int);tr=MotionTracker(max_tracks=4)
    tr.update(p,s,np.eye(4),0.)
    assert len(tr.tracks)==4
    tr.update(np.empty((0,3)),np.empty(0,int),np.eye(4),2.)
    assert len(tr.tracks)==0

def test_unknown_semantic_evidence_prevents_drivable_decision():
    p,s=patch();q=p.copy();q[:,2]+=2.5
    g=build_semantic_grid(np.r_[p,q],np.r_[s,np.zeros(len(q),int)],SINGLE)
    report=assess_terrain(g,np.r_[p,q],np.r_[s,np.zeros(len(q),int)])
    assert report['drivable']==0 and report['unknown']==g.n_cells

def test_label_order_does_not_change_cells():
    rng=np.random.default_rng(7)
    p=rng.uniform([4,0,-2],[6,2,2],(500,3));s=rng.integers(0,20,len(p))
    a=build_semantic_grid(p,s,RINGS);order=rng.permutation(len(p));b=build_semantic_grid(p[order],s[order],RINGS)
    for key in ('ix','iy','level','count','semantic','semantic_mask','ground_count'):
        np.testing.assert_array_equal(a.cells[key],b.cells[key])

def test_disabled_hazard_path_does_not_call_detector(monkeypatch):
    import src.frame_pipeline as old
    def fail(*args,**kwargs):raise AssertionError('Experimental detector called')
    monkeypatch.setattr(old,'detect',fail)
    p,s=patch();result=old.process_frame(p,np.ones(len(p),int),SINGLE)
    assert result.candidates==[] and result.detection_ms==0 and result.detection is None


def test_vehicle_width_blocks_ground_near_pole():
    p,s=patch();pts=np.r_[p,[[5.53,1.53,-.2]]];sem=np.r_[s,[18]]
    g=build_semantic_grid(pts,sem,SINGLE)
    assess_terrain(g,pts,sem)
    near=np.linalg.norm(g.xy()-[5.53,1.53],axis=1)<.8
    assert not np.any(g.cells['traversability'][near]==DRIVABLE)

@pytest.mark.parametrize('kind',['curb','depression'])
def test_synthetic_height_examples_follow_configured_step_limit(kind):
    from scripts.simulate_terrain_hazards import scenario
    for depth in (10,30):
        p=scenario(kind,depth);sem=np.full(len(p),9,np.uint8)
        grid=build_semantic_grid(p,sem,RINGS)
        diag={};assess_terrain(grid,p,sem,diagnostics=diag)
        affected=np.abs(grid.cells['ground_z']+1.7)>depth/100-.02
        assert affected.any()
        assert np.all(grid.cells['traversability'][affected]==(DRIVABLE if depth==10 else BLOCKED))
        assert grid.memory_bytes()==40*grid.n_cells


def test_terrain_diagnostics_do_not_change_default_decisions():
    p,s=patch();a=build_semantic_grid(p,s,SINGLE);b=build_semantic_grid(p,s,SINGLE)
    plain=assess_terrain(a,p,s);diag={};observed=assess_terrain(b,p,s,diagnostics=diag)
    assert plain==observed
    np.testing.assert_array_equal(a.cells,b.cells)
    assert diag['expected_ground_z'].shape==(a.n_cells,)


def test_missing_return_gap_has_no_invented_ground_cells():
    from scripts.simulate_terrain_hazards import scenario
    p=scenario('gap',0);sem=np.full(len(p),9,np.uint8)
    grid=build_semantic_grid(p,sem,RINGS);assess_terrain(grid,p,sem)
    xy=grid.xy()
    assert not np.any((xy[:,0]-6.525)**2+xy[:,1]**2<.60**2)


@pytest.mark.parametrize('kind,key,other',[
    ('curb','observed_rise_cells','observed_depression_cells'),
    ('depression','observed_depression_cells','observed_rise_cells')])
def test_main_pipeline_reports_signed_height_evidence(kind,key,other):
    from scripts.simulate_terrain_hazards import scenario
    p=scenario(kind,30)
    result=CorePipeline(RINGS).process(p,np.full(len(p),9,np.uint8),semantic_source='synthetic-road')
    assert result.terrain[key]>0
    assert result.terrain[other]==0
    assert result.terrain[key]<=result.terrain['blocked']
    assert result.grid.memory_bytes()==40*result.grid.n_cells


def test_main_pipeline_does_not_call_missing_returns_a_depression():
    from scripts.simulate_terrain_hazards import scenario
    p=scenario('gap',0)
    result=CorePipeline(RINGS).process(p,np.full(len(p),9,np.uint8),semantic_source='synthetic-road')
    assert result.terrain['observed_depression_cells']==0
    assert result.terrain['observed_rise_cells']==0
