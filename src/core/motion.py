"""Bounded geometric object tracking; no instance IDs or motion GT inputs.

Cluster movable semantic types, transform observations to world coordinates,
associate conservative matches, and classify sustained centroid displacement.
Partial views, ambiguous associations and short tracks yield UNKNOWN.
"""
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.core.semantics import MOVABLE, UNKNOWN, STATIONARY, MOVING

@dataclass
class Track:
    semantic: int
    center: np.ndarray
    size: np.ndarray
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    motion: int = UNKNOWN
    speed: float = 0.

class MotionTracker:
    def __init__(self, *, max_tracks=256, expiry_s=.8, cluster_radius=.8,
                 min_points=8, moving_speed=1., stationary_speed=.25,
                 min_duration=.4, max_voxels=12000):
        self.max_tracks=max_tracks; self.expiry=expiry_s
        self.radius=cluster_radius; self.min_points=min_points
        self.moving_speed=moving_speed; self.stationary_speed=stationary_speed
        self.min_duration=min_duration; self.max_voxels=max_voxels
        self.tracks=[]; self.last_time=None

    def reset(self): self.tracks=[]; self.last_time=None

    def update(self,points,semantics,pose,timestamp):
        points=np.asarray(points); semantics=np.asarray(semantics)
        result=np.zeros(len(points),np.uint8)
        if pose is None or timestamp is None:
            self.reset(); return result,dict(source='unavailable: pose/time missing',tracks=0)
        pose=np.asarray(pose,dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):
            raise ValueError('Expected finite homogeneous LiDAR-to-world transform')
        if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-3) or np.linalg.det(pose[:3,:3])<0:
            raise ValueError('Pose rotation must be orthonormal and right-handed')
        if not np.isfinite(timestamp): raise ValueError('Invalid timestamp')
        if self.last_time is not None and (timestamp<=self.last_time or timestamp-self.last_time>self.expiry): self.reset()
        self.last_time=timestamp
        self.tracks=[t for t in self.tracks if timestamp-t.history[-1][0]<=self.expiry]
        observations=[]; skipped=0
        for cls in MOVABLE:
            ids=np.flatnonzero((semantics==cls)&np.isfinite(points).all(axis=1))
            if len(ids)<self.min_points: continue
            p=points[ids]
            # Geometry, not raw SemanticKITTI instance IDs, defines the clusters.
            _,first,inverse=np.unique(np.floor(p/.3).astype(np.int64),axis=0,return_index=True,return_inverse=True)
            v=p[first]
            if len(v)>self.max_voxels: skipped+=len(ids); continue
            pairs=cKDTree(v).query_pairs(self.radius,output_type='ndarray')
            edges=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(v),len(v)))
            _,lab=connected_components(edges,directed=False)
            labels=lab[inverse]
            for group in np.unique(labels):
                selected=ids[labels==group]
                if len(selected)<self.min_points: continue
                world=points[selected]@pose[:3,:3].T+pose[:3,3]
                center=np.median(world,axis=0)
                size=np.ptp(world,axis=0)
                observations.append((cls,selected,center,size))
        # Mutual nearest matching, gated by time, size stability, and ambiguity.
        distance=np.full((len(observations),len(self.tracks)),np.inf)
        for i,(cls,ids,center,size) in enumerate(observations):
            for j,t in enumerate(self.tracks):
                dt=timestamp-t.history[-1][0]
                d=np.linalg.norm(center-t.center)
                stable=np.all(np.abs(size-t.size)<=.5+.35*np.maximum(size,t.size))
                if cls==t.semantic and stable and d<=min(4.,.6+25*dt): distance[i,j]=d
        used=set(); updated=[]
        for i,(cls,ids,center,size) in enumerate(observations):
            match=None
            if len(self.tracks):
                j=int(np.argmin(distance[i])); d=distance[i,j]
                row=np.sort(distance[i]); column=np.sort(distance[:,j])
                ambiguous=np.isfinite(d) and ((len(row)>1 and row[1]-d<.3) or (len(column)>1 and column[1]-d<.3))
                if np.isfinite(d) and not ambiguous and np.argmin(distance[:,j])==i and j not in used:
                    match=self.tracks[j]; used.add(j)
            if match is None: match=Track(cls,center,size)
            match.center=center; match.size=size; match.history.append((timestamp,center.copy()))
            match.motion=UNKNOWN
            elapsed=timestamp-match.history[0][0]
            if len(match.history)>=4 and elapsed>=self.min_duration:
                times=np.array([x[0] for x in match.history]); xyz=np.array([x[1] for x in match.history])
                dt=times-times.mean(); velocity=(dt[:,None]*(xyz-xyz.mean(axis=0))).sum(axis=0)/(dt@dt)
                speed=float(np.linalg.norm(velocity[:2])); match.speed=speed
                predicted=xyz.mean(axis=0)+dt[:,None]*velocity
                error=float(np.sqrt(np.mean(np.sum((xyz-predicted)**2,axis=1))))
                displacement=np.linalg.norm(xyz[-1,:2]-xyz[0,:2])
                if error<.18:
                    if speed>=self.moving_speed and displacement>=.4: match.motion=MOVING
                    elif speed<=self.stationary_speed and displacement<=.15: match.motion=STATIONARY
            result[ids]=match.motion
            updated.append(match)
        self.tracks=(updated+[t for j,t in enumerate(self.tracks) if j not in used])[:self.max_tracks]
        return result,dict(source='geometric tracking with ego-motion compensation',tracks=len(self.tracks),
                           stationary_points=int(np.count_nonzero(result==STATIONARY)),
                           moving_points=int(np.count_nonzero(result==MOVING)),
                           unknown_movable_points=int(np.count_nonzero(np.isin(semantics,MOVABLE)&(result==UNKNOWN))),
                           capacity_skipped_points=skipped)

    def payload_bytes(self):
        """Retained numeric buffers only; Python overhead belongs in RSS."""
        return sum(t.center.nbytes+t.size.nbytes+sum(p.nbytes+8 for _,p in t.history) for t in self.tracks)
