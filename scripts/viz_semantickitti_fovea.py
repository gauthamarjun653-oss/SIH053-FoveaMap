#!/usr/bin/env python3
"""PRBonn SemanticKITTI viewer with an additional FoveaMap panel.

Upstream LaserScanVis remains unmodified; this subclass adds our grid,
measurement overlays and playback controls. Semantic colours are GT labels.
"""
import argparse
from collections import deque
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'third_party'/'semantic-kitti-api'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))
from vispy import app, scene
from auxiliary.laserscan import SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
from src.common.config import load_pipeline, load_rings
from src.segmentation.class_map import build_raw_to_class3
from src.core.pipeline import CorePipeline, display_colors
from src.core.grid import build_semantic_grid
from src.core.terrain import load_core_settings
from src.core.semantics import detailed_labels
from src.io.kitti_reader import SequenceReader


class CurrentScan:
    def __init__(self, scan):
        self.current = scan
    def scan(self, _index):
        return np.column_stack([self.current.points, self.current.remissions])
    def labels(self, _index):
        return self.current.sem_label


class FoveaKITTIVis(LaserScanVis):
    def __init__(self, *args, reader, playing=False, interval=.1, compare=False, segmenter=None, **kwargs):
        self.cfg = load_pipeline(ROOT/'configs/pipeline.yaml')
        self.rings = load_rings(ROOT/'configs/rings.yaml')
        self.reader = reader
        self.times = np.loadtxt(reader.dir/'times.txt')
        self.transform_inverse = np.linalg.inv(reader.calib)
        limits,budget=load_core_settings(ROOT/'configs/core.yaml')
        self.core = CorePipeline(self.rings,limits=limits,memory_ceiling_bytes=budget)
        self.view_mode = 'terrain'
        self.compare = compare
        self.segmenter = segmenter
        self.result = None
        self.playing = playing
        self.frame_times = deque(maxlen=30)
        self.last_drawn = None
        self.measurements = None
        super().__init__(*args, **kwargs)
        self.timer = app.Timer(interval=interval, connect=self.advance, start=True)

    def reset(self):
        super().reset()
        self.canvas.size = (1800, 1000)
        self.canvas.bgcolor = '#0b0e12'
        self.fovea_view = self.grid.add_view(row=0, col=2, border_color='white')
        self.fovea_view.camera = scene.TurntableCamera(up='+z', distance=55,
                                      elevation=32, azimuth=-90, fov=45)
        for view in [self.scan_view, self.sem_view]:
            # Upstream markers have no data during reset; avoid auto-ranging them.
            view.camera.set_range(x=(-30,30), y=(-30,30), z=(-4,6))
            view.camera.up = '+z'
            view.camera.fov = 45
            view.camera.distance = 55
            view.camera.elevation = 32
            view.camera.azimuth = -90
            self.fovea_view.camera.link(view.camera)
        self.grid_vis = scene.visuals.Markers(parent=self.fovea_view.scene)
        self.grid_vis.scaling = 'scene'
        self.grid_vis.antialias = 0
        self.obstacles_vis = scene.visuals.Markers(parent=self.fovea_view.scene)
        self.cards, self.lines = [], []
        for col in range(3):
            card = scene.visuals.Rectangle(center=(1,1), width=1, height=1,
                        color=(.035,.065,.10,.94), border_color='#456078', parent=self.canvas.scene)
            card.order=100
            card.set_gl_state('translucent', depth_test=False)
            self.cards.append(card)
            lines=[]
            for row in range(6):
                text=scene.visuals.Text('', parent=self.canvas.scene, anchor_x='left', anchor_y='top',
                        font_size=14 if row==0 else 12,
                        color='#82d6b2' if col==2 else '#e5edf6')
                text.order=101
                text.set_gl_state('translucent', depth_test=False)
                lines.append(text)
            self.lines.append(lines)
        self.footer=scene.visuals.Text('',parent=self.canvas.scene,anchor_x='left',anchor_y='top',font_size=11,color='white')
        self.footer.order=101
        self.footer.set_gl_state('translucent',depth_test=False)
        self.canvas.events.resize.connect(lambda _event: self.update_overlay())

    def update_scan(self):
        # Original PRBonn scan loader, semantic colours, and 3D point rendering.
        super().update_scan()
        started=perf_counter()
        points=self.scan.points
        source='GT semantics'
        if self.segmenter is None:
            ids=detailed_labels(self.scan.sem_label)
        else:
            from src.projection.range_projection import project, back_project
            proj=project(np.column_stack([points,self.scan.remissions]),self.cfg.projection)
            prediction=self.segmenter.segment_detailed(proj.range_image)
            if prediction is None: raise RuntimeError('Non-finite model output')
            ids=back_project(prediction[0],proj.pixel_map).astype(np.uint8)
            source='SalsaNext predictions ('+self.segmenter.device.upper()+')'
        pose=self.transform_inverse@self.reader.pose(self.offset)@self.reader.calib
        result=self.core.process(points,ids,pose=pose,timestamp=float(self.times[self.offset]),semantic_source=source)
        self.result=result
        grid=result.grid
        self.grid_vis.set_data(pos=grid.xyz(),size=grid.sizes(),face_color=display_colors(grid,self.view_mode),edge_width=0,symbol='square')
        mixed=(grid.cells['ground_count']>0)&np.isfinite(grid.cells['obstacle_min'])
        obs=np.column_stack([grid.xy()[mixed],grid.cells['obstacle_min'][mixed]])
        self.obstacles_vis.set_data(pos=obs,size=4,face_color=(1.,.7,.2,1.),edge_width=0)
        m=dict(raw_scan_bytes=points.nbytes+self.scan.remissions.nbytes,raw_points=len(points),
               grid_bytes=grid.memory_bytes(),processing_ms=(perf_counter()-started)*1000,
               **result.timings,source=source,terrain=result.terrain,motion=result.motion,
               tracker_bytes=self.core.tracker.payload_bytes())
        m['semantic_bytes']=m['raw_scan_bytes']+self.scan.sem_label.nbytes
        m['uniform_bytes']=None
        if self.compare:
            uniform=build_semantic_grid(points,ids,self.rings,result.point_motion,uniform=True,memory_ceiling_bytes=512_000_000)
            m['uniform_bytes']=uniform.memory_bytes()
            m['payload_saved_pct']=100*(1-grid.memory_bytes()/max(uniform.memory_bytes(),1))
        self.measurements=m
        self.canvas.title=f'FoveaMap core | seq {self.reader.sequence} | scan {self.offset} | {source}'
        self.update_overlay()

    def update_overlay(self):
        if not hasattr(self,'lines') or self.measurements is None:
            return
        w,h=self.canvas.size
        m=self.measurements
        rate=(len(self.frame_times)-1)/(self.frame_times[-1]-self.frame_times[0]) if len(self.frame_times)>1 else 0.
        fps=f'{rate:.2f} FPS' if self.playing else 'paused'
        rows=[['PRBonn: RAW 3D LiDAR',f"Scan memory: {m['raw_scan_bytes']/1e6:.2f} MB",f'Playback: {fps}',
               f"{m['raw_points']:,} original returns",'Original official range colouring','Payload only; excludes working/GPU memory'],
              ['PRBonn: SEMANTIC 3D',f"XYZI + labels: {m['semantic_bytes']/1e6:.2f} MB",f'Playback: {fps}',
               'Official SemanticKITTI class colours','Ground-truth labels, not model predictions','Same scan and linked camera'],
              ['FOVEAMAP: TERRAIN + OBJECT MOTION',f"Map: {m['grid_bytes']/1e6:.2f} MB | 40 bytes/cell",f'Playback: {fps}',
               f"Core: {m['processing_ms']:.1f} ms | grid: {m['grid_ms']:.1f} ms",
               (f"Uniform: {m['uniform_bytes']/1e6:.2f} MB | saving {m['payload_saved_pct']:.1f}%" if m['uniform_bytes'] else 'Uniform reference OFF (C to compare)'),
               f"{m['source']} | motion estimated from scans"]]

        for col in range(3):
            left=col*w/3+12; width=w/3-24
            self.cards[col].center=(left+width/2,103)
            self.cards[col].width=width;self.cards[col].height=180
            for row,text in enumerate(self.lines[col]):
                text.pos=(left+12,40+row*28)
                text.text=rows[col][row]
        self.footer.pos=(18,h-55)
        terrain=m['terrain']; motion=m['motion']
        self.footer.text=(f"Seq {self.reader.sequence} / scan {self.offset}/{self.total-1} | Space: play/pause | N/B: step | T: terrain/semantic | C: compare | Q: quit\n"
            f"Blue: drivable surface | amber: blocked | grey: unknown | cyan: stationary object | coral: moving | lavender: unknown motion\n"
            f"Cells D/B/U: {terrain['drivable']}/{terrain['blocked']}/{terrain['unknown']} | tracks: {motion.get('tracks',0)} | rise/dip cells: {terrain.get('observed_rise_cells',0)}/{terrain.get('observed_depression_cells',0)} (unvalidated) | payload only")

        self.canvas.update()

    def draw(self,event):
        super().draw(event)
        if self.last_drawn!=self.offset:
            self.last_drawn=self.offset
            if self.playing:self.frame_times.append(perf_counter())
            self.update_overlay()

    def key_press(self,event):
        if event.key=='T':
            self.view_mode='semantic' if self.view_mode=='terrain' else 'terrain'
            if self.result:
                self.grid_vis.set_data(pos=self.result.grid.xyz(),size=self.result.grid.sizes(),face_color=display_colors(self.result.grid,self.view_mode),edge_width=0,symbol='square')
            self.canvas.update()
        elif event.key=='C':
            self.compare=not self.compare
            # Apply next frame; do not feed the same timestamp into the tracker twice.
            self.update_overlay()
        elif event.key=='Space':
            self.playing=not self.playing
            self.frame_times.clear()
            self.update_overlay()
        else:
            super().key_press(event)

    def advance(self,_event):
        if self.playing:
            if self.offset+1>=self.total:
                self.playing=False
                self.update_overlay()
            else:
                self.offset+=1
                self.update_scan()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',default='09')
    parser.add_argument('--dataset',default=str(ROOT/'data/semantickitti'))
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--play',action='store_true')
    parser.add_argument('--interval',type=float,default=.1)
    parser.add_argument('--mode',choices=['gt-labels','model'],default='gt-labels')
    parser.add_argument('--compare',action='store_true')
    args=parser.parse_args()
    directory=Path(args.dataset)/'sequences'/args.sequence
    scans=sorted((directory/'velodyne').glob('*.bin'))
    labels=[directory/'labels'/(scan.stem+'.label') for scan in scans]
    if not scans or not all(p.is_file() for p in labels):
        parser.error('A complete matching set of scans and labels is required')
    if not 0<=args.start<len(scans) or args.interval<=0:
        parser.error('Invalid start frame or interval')
    config=yaml.safe_load((UPSTREAM/'config/semantic-kitti.yaml').read_text())
    scan=SemLaserScan(config['color_map'],project=True)
    reader=SequenceReader(args.dataset,args.sequence)
    segmenter=None
    if args.mode=='model':
        from src.segmentation.inference import Segmenter
        cfg=load_pipeline(ROOT/'configs/pipeline.yaml')
        segmenter=Segmenter(str(ROOT/cfg.weights_path),str(ROOT/cfg.semantic_kitti_yaml),device=cfg.device,fp16=cfg.fp16)
    viewer=FoveaKITTIVis(scan,[str(p) for p in scans],[str(p) for p in labels],
                        offset=args.start,semantics=True,images=False,link=False,
                        reader=reader,playing=args.play,interval=args.interval,compare=args.compare,segmenter=segmenter)
    viewer.run()

if __name__=='__main__':
    main()
