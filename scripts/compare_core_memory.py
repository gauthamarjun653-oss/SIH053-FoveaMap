#!/usr/bin/env python3
"""Honest payload comparison: old/new maps and a minimal sparse 3D reference.

3D reference fields differ: voxel coordinates, count and semantic presence.
No ground/clearance summaries. Never treat it as an equivalent-field 2.5D map.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common.config import load_rings
from src.io.kitti_reader import SequenceReader
from src.core.grid import build_semantic_grid, unique_rows
from src.core.semantics import detailed_labels
from src.grid.compact import build_compact_grid
from src.segmentation.class_map import build_raw_to_class3

VOXEL=np.dtype([('x','<i4'),('y','<i4'),('z','<i4'),('count','<u4'),('semantic_mask','<u4')])
def voxel_reference(points,labels,cell=.05):
    keys,inv=unique_rows(np.floor(points/cell).astype(np.int64))
    if keys.size and (keys.min()<np.iinfo(np.int32).min or keys.max()>np.iinfo(np.int32).max): raise OverflowError('Voxel coordinate overflow')
    out=np.zeros(len(keys),VOXEL)
    out['x'],out['y'],out['z']=keys.T
    counts=np.bincount(inv,minlength=len(keys))
    if counts.size and counts.max()>np.iinfo(np.uint32).max:raise OverflowError('Voxel count overflow')
    out['count']=counts
    np.bitwise_or.at(out['semantic_mask'],inv,np.left_shift(np.uint32(1),labels.astype(np.uint32)))
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frames',type=int,default=10)
    ap.add_argument('--out',default=str(ROOT/'output/core-memory-comparison.json'));args=ap.parse_args()
    rings=load_rings(ROOT/'configs/rings.yaml');lut=build_raw_to_class3();rows=[]
    for seq in ('09','10'):
        reader=SequenceReader(ROOT/'data/semantickitti',seq)
        for i in range(min(args.frames,len(reader))):
            points=reader.scan(i);raw=reader.labels(i);labels=detailed_labels(raw)
            radius=np.hypot(points[:,0],points[:,1]);keep=np.isfinite(points).all(axis=1)&(radius>=rings.radius_min)&(radius<=rings.radius_max)
            p=points[keep,:3];s=labels[keep]
            old=build_compact_grid(p,lut[raw[keep]],rings)
            new=build_semantic_grid(p,s,rings)
            uniform=build_semantic_grid(p,s,rings,uniform=True,memory_ceiling_bytes=512_000_000)
            voxel=voxel_reference(p,s)
            rows.append(dict(sequence=seq,frame=i,points=len(p),old_map_bytes=old.memory_bytes(),
              new_map_bytes=new.memory_bytes(),uniform_25d_bytes=uniform.memory_bytes(),minimal_sparse_3d_bytes=voxel.nbytes,
              new_vs_old_saving_pct=100*(1-new.memory_bytes()/old.memory_bytes()),
              new_vs_uniform_saving_pct=100*(1-new.memory_bytes()/uniform.memory_bytes()),
              retained_count_exact=int(new.cells['count'].sum())==len(p)))
    report=dict(point_policy='Same finite points in planar 2.5–100 m, including unknown labels.',
      retained_fields='New map: 40 bytes per cell, detailed presence mask and ground/obstacle separation. Old map: 56 bytes with 4-class counts and derived arrays.',
      voxel_fields='Sparse 5 cm 3D: 20 bytes/voxel, XYZ int32, uint32 count and semantic bitmask. Different representation/fields.',
      scope='Array payload only. Not total RAM or GPU memory. No latency claims.',frames=rows)
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(report,indent=2)+'\n')
    for seq in ('09','10'):
        r=[x for x in rows if x['sequence']==seq]
        print(seq,{key:round(float(np.median([x[key] for x in r])),3) for key in ('old_map_bytes','new_map_bytes','uniform_25d_bytes','minimal_sparse_3d_bytes','new_vs_old_saving_pct','new_vs_uniform_saving_pct')})
if __name__=='__main__': main()
