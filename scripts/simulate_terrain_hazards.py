#!/usr/bin/env python3
"""Run the actual terrain assessor on controlled synthetic curb/depression scenes.

Scene truth never enters the detector: all samples are supplied as road class.
These idealized point samples are not a physical LiDAR ray/occlusion simulator.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common.config import load_rings
from src.core.grid import build_semantic_grid
from src.core.terrain import assess_terrain,load_core_settings


def scenario(kind, amplitude_cm):
    x,y=np.meshgrid(np.arange(3.025,10.,.10),np.arange(-1.975,2.,.10))
    x=x.ravel();y=y.ravel()
    affected=(x-6.525)**2+y*y<.65**2
    if kind=='curb': affected=(x>6.2)&(x<6.8)&(np.abs(y)<1.25)
    z=np.full(len(x),-1.7)+np.random.default_rng(42).normal(0,.002,len(x))
    if kind in ('depression','curb'):
        z[affected]+=amplitude_cm/100*(-1 if kind=='depression' else 1)
    keep=~affected if kind=='gap' else np.ones(len(x),bool)
    return np.column_stack([x[keep],y[keep],z[keep]])


def run_scenarios():
    rings=load_rings(ROOT/'configs/rings.yaml')
    limits,budget=load_core_settings(ROOT/'configs/core.yaml')
    scenarios=[];base_xy=None
    configs=[('depression',d) for d in range(0,41,5)]+[('curb',d) for d in range(0,41,5)]+[('gap',0)]
    for kind,depth in configs:
        points=scenario(kind,depth);sem=np.full(len(points),9,np.uint8)
        grid=build_semantic_grid(points,sem,rings,memory_ceiling_bytes=budget)
        diag={};report=assess_terrain(grid,points,sem,limits,diagnostics=diag)
        xy=grid.xy()
        if base_xy is None:base_xy=xy.copy()
        index={(round(x,4),round(y,4)):i for i,(x,y) in enumerate(base_xy)}
        ids=np.array([index[(round(x,4),round(y,4))] for x,y in xy])
        state=np.full(len(base_xy),3,dtype=np.uint8) # 3 = absent, display only, no allocated cell
        state[ids]=grid.cells['traversability']
        z=np.full(len(base_xy),np.nan);z[ids]=grid.cells['ground_z']+1.7
        affected=(xy[:,0]-6.525)**2+xy[:,1]**2<.60**2
        if kind=='curb':affected=(xy[:,0]>6.25)&(xy[:,0]<6.75)&(np.abs(xy[:,1])<1.15)
        counts={name:int(np.count_nonzero(affected&(grid.cells['traversability']==value))) for value,name in enumerate(['unknown','drivable','blocked'])}
        profile_mask=np.abs(xy[:,1]-.025)<.02
        profile=[]
        for i in np.flatnonzero(profile_mask):
            plane=diag['expected_ground_z'][i]+1.7
            profile.append([round(float(xy[i,0]),3),round(float(grid.cells['ground_z'][i]+1.7),3),
                            round(float(plane),3) if np.isfinite(plane) else None,
                            int(grid.cells['traversability'][i])])
        profile.sort()
        scenarios.append(dict(kind=kind,amplitude_cm=depth,states=''.join(str(v) for v in state),
          z=[round(float(v),3) if np.isfinite(v) else None for v in z],profile=profile,
          affected=counts,missing_cells=int((state==3).sum()),
          max_step_cm=round(float(np.nanmax(diag['step_m']))*100,1),
          observed_cells=grid.n_cells,terrain=report))
    return dict(source='Actual src/core/terrain.py outputs, synthetic idealized road samples; no neural inference or physical ray simulation',
        max_step_cm=limits.max_step_m*100,xy=np.round(base_xy,3).tolist(),scenarios=scenarios)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default=str(ROOT/'output/terrain-hazard-simulation.json'))
    args=ap.parse_args();data=run_scenarios()
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    for case in data['scenarios']:
        if case['amplitude_cm'] in (0,10,20,30):
            print(case['kind'],case['amplitude_cm'],case['affected'],'absent',case['missing_cells'],'max_step_cm',case['max_step_cm'])
    print(path)
if __name__=='__main__':main()
