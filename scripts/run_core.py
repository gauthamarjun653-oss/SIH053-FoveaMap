#!/usr/bin/env python3
"""Run and measure the three PS tasks. GT replay is explicitly labelled."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common.config import load_rings, load_pipeline
from src.io.kitti_reader import SequenceReader
from src.core.pipeline import CorePipeline
from src.core.terrain import VehicleLimits, load_core_settings
from dataclasses import replace
from src.core.semantics import detailed_labels, motion_truth, MOVABLE, NAMES
from src.core.grid import build_semantic_grid

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',default='10')
    ap.add_argument('--dataset',default=str(ROOT/'data/semantickitti'))
    ap.add_argument('--mode',choices=['gt-labels','model','random'],default='gt-labels')
    ap.add_argument('--frames',type=int,default=30)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--budget-mb',type=float,default=None)
    ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--rings',default=str(ROOT/'configs/rings.yaml'))
    ap.add_argument('--pipeline',default=str(ROOT/'configs/pipeline.yaml'))
    ap.add_argument('--core-config',default=str(ROOT/'configs/core.yaml'))
    ap.add_argument('--vehicle-height',type=float,default=None)
    ap.add_argument('--evaluate-labels',action='store_true',help='Evaluate against dataset labels after timed processing, never feed them into model mode')
    ap.add_argument('--compare',action='store_true',help='Separately build a uniform 2.5D reference after each timed frame')
    ap.add_argument('--out',default=str(ROOT/'output/core-report.json'))
    args=ap.parse_args(argv)
    limits,budget=load_core_settings(args.core_config)
    if args.budget_mb is not None: budget=int(args.budget_mb*1e6)
    if args.vehicle_height is not None: limits=replace(limits,height_m=args.vehicle_height)
    if args.frames<=0 or args.start<0 or budget<=0 or args.stride<=0: ap.error('Invalid frame/budget options')
    reader=SequenceReader(args.dataset,args.sequence,require_labels=args.mode=='gt-labels' or args.evaluate_labels)
    if args.start>=len(reader):ap.error('Start is beyond the sequence')
    times=np.loadtxt(reader.dir/'times.txt')
    if len(times)!=len(reader):raise ValueError('Timestamp/scan count mismatch')
    rings=load_rings(args.rings)
    pipe=CorePipeline(rings,limits=limits,memory_ceiling_bytes=budget)
    tr=reader.calib; invtr=np.linalg.inv(tr)
    segmenter=None
    if args.mode!='gt-labels':
        from src.segmentation.inference import Segmenter
        from src.projection.range_projection import project, back_project
        cfg=load_pipeline(args.pipeline)
        segmenter=Segmenter(str(ROOT/cfg.weights_path),str(ROOT/cfg.semantic_kitti_yaml),device=cfg.device,fp16=cfg.fp16,random_init=args.mode=='random')
    if segmenter and segmenter.device == 'cuda':
        segmenter._torch.cuda.reset_peak_memory_stats()
    frames=[]; confusion=np.zeros((3,3),np.int64)
    semantic_confusion={name:np.zeros((20,20),np.int64) for name in ['all','2.5-10m','10-25m','25-50m','50-100m']}
    bands=[(2.5,10.),(10.,25.),(25.,50.),(50.,100.)]
    for index in range(args.start,min(args.start+args.frames,len(reader)),args.stride):
        started=time.perf_counter(); scan=reader.scan(index)
        if segmenter:
            proj=project(scan,cfg.projection)
            prediction=segmenter.segment_detailed(proj.range_image)
            if prediction is None: raise RuntimeError(f'Model produced non-finite values on frame {index}')
            ids=back_project(prediction[0],proj.pixel_map)
            ids=np.where(proj.pixel_map[:,0]>=0,ids,0).astype(np.uint8)
        else:
            raw=reader.labels(index); ids=detailed_labels(raw)
        pose=invtr@reader.pose(index)@tr
        result=pipe.process(scan[:,:3],ids,pose=pose,timestamp=float(times[index]),semantic_source=args.mode)
        frame_ms=(time.perf_counter()-started)*1000
        row=dict(frame=index,processing_ms=frame_ms,**result.timings,map_bytes=result.grid.memory_bytes(),
                 cells=result.grid.n_cells,raw_bytes=scan.nbytes,terrain=result.terrain,motion=result.motion,
                 assigned_points=result.grid.assigned_points,discarded_points=result.grid.discarded_points)
        if not segmenter or args.evaluate_labels:
            if segmenter: raw=reader.labels(index)
            semantic_truth=detailed_labels(raw)
            truth=motion_truth(raw); selected=np.isin(semantic_truth,MOVABLE)
            r=np.hypot(scan[:,0],scan[:,1]);selected&=(r>=rings.radius_min)&(r<=rings.radius_max)
            confusion+=np.bincount(truth[selected]*3+result.point_motion[selected],minlength=9).reshape(3,3)
            if segmenter and args.evaluate_labels:
                valid=(semantic_truth>0)&np.isfinite(scan[:,:3]).all(axis=1)&(r>=rings.radius_min)&(r<=rings.radius_max)
                semantic_confusion['all']+=np.bincount(semantic_truth[valid].astype(np.int64)*20+ids[valid],minlength=400).reshape(20,20)
                for name,(lo,hi) in zip(list(semantic_confusion)[1:],bands):
                    band=valid&(r>lo)&(r<=hi)
                    semantic_confusion[name]+=np.bincount(semantic_truth[band].astype(np.int64)*20+ids[band],minlength=400).reshape(20,20)
        if args.compare:
            uniform=build_semantic_grid(scan[:,:3],ids,rings,result.point_motion,uniform=True,memory_ceiling_bytes=512_000_000)
            row['uniform_map_bytes']=uniform.memory_bytes()
            row['savings_vs_uniform_pct']=100*(1-row['map_bytes']/max(uniform.memory_bytes(),1))
            del uniform
        frames.append(row)
    # Kernel high-water mark covers processing and optional reference construction, but no GUI.
    peak=next((int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')),None)
    durations=np.array([f['processing_ms'] for f in frames])
    summary=dict(sequence=args.sequence,mode=args.mode,frames=len(frames),bytes_per_cell=40,
                 processing_ms_median=float(np.median(durations)),processing_ms_p95=float(np.percentile(durations,95)),
                 processing_throughput_fps=float(1000/durations.mean()),
                 median_map_bytes=float(np.median([f['map_bytes'] for f in frames])),
                 process_peak_rss_bytes=peak,reference_in_peak=args.compare,rendering_included=False,
                 inference_device=segmenter.device if segmenter else None,
                 inference_fp16=segmenter.fp16 if segmenter else False,
                 gpu_name=segmenter._torch.cuda.get_device_name() if segmenter and segmenter.device=='cuda' else None,
                 gpu_peak_allocated_bytes=segmenter._torch.cuda.max_memory_allocated() if segmenter and segmenter.device=='cuda' else None,
                 gpu_peak_reserved_bytes=segmenter._torch.cuda.max_memory_reserved() if segmenter and segmenter.device=='cuda' else None,
                 semantics_are_predictions=bool(segmenter),random_weights=args.mode=='random',tracker_numeric_bytes=pipe.tracker.payload_bytes(),
                 motion_confusion_rows_truth_cols_pred_unknown_stationary_moving=confusion.tolist(),
                 caveats=['GT motion benchmark is diagnostic on development sequences, not held-out accuracy.',
                          'Dataset moving labels are not necessarily instantaneous velocity.',
                          'Traversability uses configured vehicle limits, not verified terrain ground truth.',
                          'Pothole detection disabled; no pothole claims.',
                          'Map payload excludes working arrays/model/GPU memory.'],per_frame=frames)
    total_motion=int(confusion[1:].sum()); known=int(confusion[1:,1:].sum())
    summary['motion_diagnostic']=dict(evaluated_points=total_motion,decided_points=known,
        decided_coverage=known/total_motion if total_motion else None,
        accuracy_among_decided=float((confusion[1,1]+confusion[2,2])/known) if known else None)
    if segmenter and args.evaluate_labels:
        evaluation={}
        for name,matrix in semantic_confusion.items():
            tp=np.diag(matrix);union=matrix.sum(axis=0)+matrix.sum(axis=1)-tp
            scores={NAMES[i]:float(tp[i]/union[i]) if union[i] else None for i in range(1,20)}
            present=[v for v in scores.values() if v is not None]
            evaluation[name]=dict(points=int(matrix.sum()),iou=scores,mean_present_class_iou=float(np.mean(present)) if present else None)
        summary['semantic_diagnostic']=evaluation
        summary['caveats'].append('09/10 are training-split sequences for the pretrained model; these are smoke diagnostics, not generalization scores.')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_frame'},indent=2))
    print(f'Report: {out}')
if __name__=='__main__':main()
