#!/usr/bin/env bash
# Local VNC stack for the dual raw-3D / 2.5D viewer.
#
# Uses the Xvfb, x11vnc and noVNC bundled under .venv/native, so nothing needs
# installing. Replaces the retired start_vnc_96/97.sh and start_all_viz.sh,
# which hard-coded paths on a cloud VM that no longer exists.
#
#   scripts/start_vnc.sh --sequence 09            # start
#   scripts/start_vnc.sh --sequence 09 --play     # start and auto-play
#   scripts/start_vnc.sh --stop                   # tear down
#
# Then open http://localhost:6080/vnc.html?autoconnect=true
# or point a VNC client at localhost:5900 (localhost-only, no password).
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB="$R/.venv/native/usr/bin"
PY="$R/.venv/bin/python"
SEQ=09; DISP=97; VNC_PORT=5900; WEB_PORT=6080; START=0; IMPL=corrected; PLAY=""; STOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sequence) SEQ=$2; shift 2;;   --display) DISP=$2; shift 2;;
    --vnc-port) VNC_PORT=$2; shift 2;; --web-port) WEB_PORT=$2; shift 2;;
    --start) START=$2; shift 2;;    --implementation) IMPL=$2; shift 2;;
    --play) PLAY="--play"; shift;;  --stop) STOP=1; shift;;
    -h|--help) sed -n 2,14p "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
# tmux kill-session only HUPs the shell; Xvfb and x11vnc survive it and the
# next Xvfb then dies with "server already running". Kill by display, then
# wait for the socket to go away before starting anything.
for s in novnc$DISP vnc$DISP viz$DISP xvfb$DISP; do tmux kill-session -t "$s" 2>/dev/null || true; done
pkill -f "[X]vfb :$DISP( |$)" 2>/dev/null || true
pkill -f "[x]11vnc -display :$DISP( |$)" 2>/dev/null || true
pkill -f "[v]iz_dual_fovea.py.*" 2>/dev/null || true
pkill -f "[w]ebsockify .*localhost:$WEB_PORT" 2>/dev/null || true
for _ in $(seq 1 20); do [ -e "/tmp/.X11-unix/X$DISP" ] || break; sleep 0.25; done
rm -f "/tmp/.X${DISP}-lock" "/tmp/.X11-unix/X$DISP"
[ "$STOP" = 1 ] && { echo "stopped"; exit 0; }
for b in Xvfb x11vnc; do [ -x "$NB/$b" ] || { echo "missing $NB/$b" >&2; exit 1; }; done
[ -d "$R/data/semantickitti/sequences/$SEQ" ] || { echo "sequence $SEQ not under data/semantickitti" >&2; exit 1; }
LOG="$R/logs"; mkdir -p "$LOG"
tmux new-session -d -s "xvfb$DISP" "$NB/Xvfb :$DISP -screen 0 1600x900x24 > $LOG/xvfb$DISP.log 2>&1"
sleep 2
tmux new-session -d -s "viz$DISP" "cd $R && DISPLAY=:$DISP $PY scripts/viz_dual_fovea.py --sequence $SEQ --dataset $R/data/semantickitti --start $START --implementation $IMPL $PLAY > $LOG/viz$DISP.log 2>&1"
sleep 6
tmux new-session -d -s "vnc$DISP" "$NB/x11vnc -display :$DISP -rfbport $VNC_PORT -localhost -forever -shared -nopw -quiet > $LOG/x11vnc$DISP.log 2>&1"
sleep 2
# python -m: the .venv/bin/websockify entry point breaks when the venv is moved.
tmux new-session -d -s "novnc$DISP" "cd $R && $PY -m websockify --web=$R/.venv/native/usr/share/novnc localhost:$WEB_PORT localhost:$VNC_PORT > $LOG/novnc$DISP.log 2>&1"
sleep 2
echo "sequence $SEQ on display :$DISP"
echo "  noVNC : http://localhost:$WEB_PORT/vnc.html?autoconnect=true"
echo "  VNC   : localhost:$VNC_PORT"
echo "  logs  : $LOG/*$DISP.log   stop: $0 --stop --display $DISP"
