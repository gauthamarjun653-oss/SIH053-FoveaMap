"""Detailed learning taxonomy, kept separate from motion and traversability."""
import numpy as np

NAMES = ('unknown', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
         'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
         'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
         'pole', 'traffic-sign')
RAW = {0:0,1:0,10:1,11:2,15:3,18:4,13:5,16:5,20:5,30:6,31:7,32:8,
       40:9,60:9,44:10,48:11,49:12,50:13,51:14,70:15,71:16,72:17,80:18,81:19,
       252:1,253:7,254:6,255:8,256:5,257:5,258:4,259:5}
GROUND = (9,10,11,12,17)
MOVABLE = tuple(range(1,9))
UNKNOWN, STATIONARY, MOVING = 0,1,2
UNASSESSED, DRIVABLE, BLOCKED = 0,1,2
GROUND_BITS = sum(1 << c for c in GROUND)
ROAD_BITS = (1 << 9) | (1 << 10)
MOVABLE_BITS = sum(1 << c for c in MOVABLE)

def detailed_labels(raw):
    raw = np.asarray(raw)
    out = np.zeros(raw.shape, np.uint8)
    for key, value in RAW.items():
        out[raw == key] = value
    return out

def motion_truth(raw):
    """Evaluation only. Raw motion annotations must never enter the tracker."""
    raw = np.asarray(raw)
    result = np.zeros(raw.shape, np.uint8)
    result[np.isin(raw, (10,11,13,15,16,18,20,30,31,32))] = STATIONARY
    result[(raw >= 252) & (raw <= 259)] = MOVING
    return result

COLORS = np.array(['#555B65','#38C9E8','#40E0B0','#CE93D8','#6085FF','#9061E8',
 '#FF80AB','#A3F7D0','#E0BBFF','#526D82','#8299AD','#D6BC8B','#A89B70',
 '#AEB8C4','#B39DDB','#45B85C','#A97142','#9AAE50','#F4D35E','#FF922B'])
PALETTE = np.array([[int(s[i:i+2],16)/255 for i in (1,3,5)] for s in COLORS])
