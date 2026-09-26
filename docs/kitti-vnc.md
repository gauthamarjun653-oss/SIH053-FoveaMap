# KITTI + FoveaMap VNC demo

Run from the project folder. The launcher locates the project relative to itself, so it also works using its absolute path. It uses the existing `.venv`, bundled Xvfb/x11vnc/noVNC, official SemanticKITTI viewer, and downloaded scans/labels. It does not install dependencies or download data.

```bash
cd /home/harishgautham/sih26053-foveamap-main
./scripts/kitti_vnc.sh start 10       # Start sequence 10 playback in the background
./scripts/kitti_vnc.sh restart 07 --mode model  # Sequence 07 with actual inference
./scripts/kitti_vnc.sh restart 09     # Switch to sequence 09, or restart a finished replay
./scripts/kitti_vnc.sh restart 10     # Switch back to sequence 10
./scripts/kitti_vnc.sh status         # Check all four services and print the URL
./scripts/kitti_vnc.sh connect        # Open native TigerVNC on your desktop (optional)
./scripts/kitti_vnc.sh logs           # Show the last 20 lines of each service log
./scripts/kitti_vnc.sh stop           # Stop this launcher's demo and optional client
./scripts/kitti_vnc.sh --help
```

Browser: http://127.0.0.1:6081/vnc.html?autoconnect=true&resize=scale

Native TigerVNC, if you prefer to connect manually:

```bash
vncviewer 127.0.0.1::5901
```

## What the three panels show

| Panel | Display | Measurements |
| --- | --- | --- |
| Left | Original raw 3D LiDAR, coloured by range | XYZI scan payload memory and shared playback FPS |
| Middle | Official SemanticKITTI semantic colours | Scan plus ground-truth label payload and shared playback FPS |
| Right | Current terrain, semantics and estimated motion | 40-byte/cell map payload, core/grid times; optional uniform-map reference |

The middle panel uses ground-truth labels, not neural-network predictions. Experimental pothole candidate markers are disabled in the right panel. Pink points in the middle panel belong to the official semantic palette. The right panel explicitly identifies its source: dataset labels or SalsaNext predictions (CPU/CUDA).

Memory figures exclude process overhead, temporary arrays, and GPU/rendering memory. The uniform comparison uses matching map fields; it does not claim savings versus raw points. All panels share one playback rate. The default replay timer requests a frame every 0.1 seconds (a requested 10 FPS maximum); actual playback may be slower. Grid-build time alone is not end-to-end latency. Reference-map construction and rendering also take time. The uniform reference is OFF by default; press C to enable it on the next scan.

## Controls and replay options

Click inside the viewer before keyboard input. Space toggles play/pause; N/B steps forward/backward; T switches terrain/semantic colours; C toggles the uniform comparison; drag rotates; scroll zooms; Q closes the viewer. Pause before manually stepping. Playback stops at the final scan; `restart` begins again. Closing the browser tab does not stop the server; use `stop` to release resources.

```bash
./scripts/kitti_vnc.sh restart 10 --start-frame 100 --interval 0.5
```

A shorter interval requests faster replay but does not speed up processing.

## Services and troubleshooting

The script launches Xvfb on display `:96` at 1800×1000, the official-viewer adapter with `--sequence ... --play`, x11vnc on `127.0.0.1:5901`, and websockify/noVNC on `127.0.0.1:6081`. Both network listeners are local. No password is needed on this local connection.

Logs and PID/start-time records are stored under `/tmp/foveamap-vnc-<user-id>/`. Repeating `start` with the same options leaves an existing healthy demo running. Use `restart` to change sequence or options. Shutdown targets recorded process identities; unrelated VNC sessions are left running. Runtime files disappear on some reboots; run `start` again.

For occupied ports/displays, inspect `status` and `logs`. To run a separate instance, use a separate runtime directory and free ports/display, and use that same runtime directory for subsequent commands:

```bash
KITTI_VNC_RUNTIME=/tmp/foveamap-second KITTI_VNC_DISPLAY=:98 \
KITTI_VNC_PORT=5902 KITTI_WEB_PORT=6082 ./scripts/kitti_vnc.sh start 10
KITTI_VNC_RUNTIME=/tmp/foveamap-second ./scripts/kitti_vnc.sh status
KITTI_VNC_RUNTIME=/tmp/foveamap-second ./scripts/kitti_vnc.sh stop
```

Its browser URL uses port 6082. Supply the same port/display overrides when restarting that instance.


## Model mode and colour meaning

```bash
./scripts/kitti_vnc.sh restart 10 --mode model
./scripts/kitti_vnc.sh restart 10 --mode gt-labels
```

`model` performs actual SalsaNext inference. `gt-labels` bypasses inference and
uses dataset semantics; it is useful for isolating the mapping algorithm. The
middle panel always remains a labelled ground-truth reference. All three panels
share a single replay rate; separate independent FPS values would be misleading.

Terrain view: blue = observed surface passes configured vehicle limits;
amber = blocked; grey = unknown. Movable objects override those colours:
cyan = estimated stationary, coral = estimated moving, lavender = unknown motion.
The tracker uses geometric clusters, timestamps and KITTI ego poses, not instance
IDs or moving-object ground-truth labels. Occlusion, merged objects and pose
errors can still cause incorrect motion estimates.

The map retains separate ground heights and obstacle bounds. Vehicle limits and
map payload budget are in `configs/core.yaml`; they are demonstration defaults,
not certified navigation parameters. The map is replaced each frame; only bounded
track histories persist. There is no accumulated global navigation map here.
