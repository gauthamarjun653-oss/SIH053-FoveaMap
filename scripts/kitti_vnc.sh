#!/usr/bin/env bash
# Local PRBonn SemanticKITTI + FoveaMap viewer. Run --help for commands.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "$PROJECT_ROOT" "$@" <<'PY'
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

root = Path(sys.argv[1])
parser = argparse.ArgumentParser(prog='kitti_vnc.sh', description='Run the three-panel KITTI/FoveaMap VNC demo.', epilog='Examples: scripts/kitti_vnc.sh start 10 | restart 09 | status | stop | logs | connect')
parser.add_argument('action', nargs='?', default='start', choices=['start', 'restart', 'stop', 'status', 'logs', 'connect'])
parser.add_argument('sequence', nargs='?', default='10', choices=['07', '09', '10'])
parser.add_argument('--start-frame', type=int, default=0)
parser.add_argument('--interval', type=float, default=.1, help='Replay timer in seconds (default .1); actual FPS may be lower.')
parser.add_argument('--mode', choices=['gt-labels','model'], default='gt-labels')
args = parser.parse_args(sys.argv[2:])
run = Path(os.environ.get('KITTI_VNC_RUNTIME', f'/tmp/foveamap-vnc-{os.getuid()}'))
run.mkdir(mode=0o700, parents=True, exist_ok=True)
lock = (run/'lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
state_path = run/'state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {}
display = os.environ.get('KITTI_VNC_DISPLAY', ':96')
vnc_port = int(os.environ.get('KITTI_VNC_PORT', '5901'))
web_port = int(os.environ.get('KITTI_WEB_PORT', '6081'))
url = f'http://127.0.0.1:{web_port}/vnc.html?autoconnect=true&resize=scale'
py = root/'.venv/bin/python'
native = root/'.venv/native/usr'
required = ('display', 'viewer', 'vnc', 'browser')

def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'started': stat[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None

def alive(record):
    return bool(record) and identity(record['pid']) == record

def save():
    temporary = run/'state.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(state_path)

def stop():
    # Match PID and kernel start time: never kill an unrelated reused PID.
    for name in ('client', 'viewer', 'browser', 'vnc', 'display'):
        record = state.get('processes', {}).get(name)
        if not alive(record):
            continue
        try:
            os.kill(record['pid'], signal.SIGCONT)
            os.kill(record['pid'], signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + 3
        while alive(record) and time.monotonic() < deadline:
            time.sleep(.1)
        if alive(record):
            os.kill(record['pid'], signal.SIGKILL)
    state.clear()
    save()

def describe():
    print(f"Sequence {state.get('sequence', args.sequence)} | {state.get('url', url)}")
    print('LEFT: raw 3D LiDAR, coloured by range; original XYZI scan bytes.')
    print('MIDDLE: official SemanticKITTI ground-truth colours, not model predictions.')
    print('RIGHT: terrain assessment and estimated motion; experimental pothole detector is OFF.')
    print('Overlays: 40-byte map payload, playback FPS and core/grid times. C toggles uniform reference comparison.')
    print('Payload excludes process/GPU memory. This is dataset replay; FPS includes display work.')
    print('Controls: Space play/pause | N/B step | T terrain/semantic | C comparison | drag rotate | scroll zoom | Q close viewer.')
    print(f'Logs/state: {run}')

def launch(name, command, env):
    with (run/f'{name}.log').open('ab') as log:
        process = subprocess.Popen([str(x) for x in command], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    record = identity(process.pid)
    if not record:
        raise RuntimeError(f'{name} exited at startup; inspect {run}/{name}.log')
    state.setdefault('processes', {})[name] = record
    save()
    return process

try:
    if args.action == 'stop':
        stop()
        print('Stopped this script’s VNC demo services.')
    elif args.action == 'status':
        for name in required:
            print(f"{name}: {'running' if alive(state.get('processes', {}).get(name)) else 'stopped'}")
        if state:
            describe()
        sys.exit(0 if all(alive(state.get('processes', {}).get(n)) for n in required) else 1)
    elif args.action == 'logs':
        for path in sorted(run.glob('*.log')):
            print(f'\n{path.name}:')
            print('\n'.join(path.read_text(errors='replace').splitlines()[-20:]))
    elif args.action == 'connect':
        if not alive(state.get('processes', {}).get('vnc')):
            raise RuntimeError('Start the demo first: scripts/kitti_vnc.sh start 10')
        if not os.environ.get('DISPLAY'):
            raise RuntimeError('No desktop DISPLAY. Open the browser URL shown by status instead.')
        if not alive(state.get('processes', {}).get('client')):
            launch('client', ['/usr/bin/vncviewer', '-Shared', '-SecurityTypes', 'None',
                             '-AcceptClipboard=0', '-SendClipboard=0', f"127.0.0.1::{state['vnc_port']}"], os.environ.copy())
        describe()
    else:
        # Validate before stopping a working demo.
        scans = sorted((root/f'data/semantickitti/sequences/{args.sequence}/velodyne').glob('*.bin'))
        if not scans or not 0 <= args.start_frame < len(scans) or args.interval <= 0:
            raise RuntimeError('Missing scans, invalid start frame, or invalid replay interval.')
        if not all((p.parent.parent/'labels'/f'{p.stem}.label').is_file() for p in scans):
            raise RuntimeError('Some sequence labels are missing.')
        for path in (py, native/'bin/Xvfb', native/'bin/x11vnc', native/'share/novnc/vnc.html',
                     root/'scripts/viz_semantickitti_fovea.py'):
            if not path.exists():
                raise RuntimeError(f'Missing dependency: {path}')
        running = all(alive(state.get('processes', {}).get(n)) for n in required)
        if args.action == 'start' and running:
            if (state.get('sequence'), state.get('start_frame'), state.get('interval'), state.get('mode','gt-labels')) != (args.sequence, args.start_frame, args.interval, args.mode):
                raise RuntimeError('Demo already running with different options. Use restart with your new options.')
            print('Already running.')
            describe()
            sys.exit(0)
        if args.action == 'restart' or state:
            stop()
        if not display.startswith(':') or not display[1:].isdigit():
            raise RuntimeError('KITTI_VNC_DISPLAY must look like :96')
        if Path(f'/tmp/.X11-unix/X{display[1:]}').exists() or Path(f'/tmp/.X{display[1:]}-lock').exists():
            raise RuntimeError(f'Display {display} is occupied. Use KITTI_VNC_DISPLAY=:98; no existing display was stopped.')
        for port in (vnc_port, web_port):
            with socket.socket() as sock:
                # Permit a restart while old connections are in TIME_WAIT.
                # An active listener still prevents this bind.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(('127.0.0.1', port))
                except OSError as error:
                    raise RuntimeError(f'Port {port} unavailable. Stop its existing service or choose KITTI_VNC_PORT/KITTI_WEB_PORT.') from error
        state.update(sequence=args.sequence, start_frame=args.start_frame, interval=args.interval, mode=args.mode,
                     url=url, vnc_port=vnc_port, display=display, processes={})
        env = os.environ.copy()
        env.pop('WAYLAND_DISPLAY', None)
        env.update(DISPLAY=display, XDG_SESSION_TYPE='x11', QT_QPA_PLATFORM='xcb', VISPY_APP='pyqt6',
                   PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(run/'matplotlib'))
        try:
            launch('display', [native/'bin/Xvfb', display, '-screen', '0', '1800x1000x24', '-nolisten', 'tcp'], env)
            deadline = time.monotonic() + 5
            while not Path(f'/tmp/.X11-unix/X{display[1:]}').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Xvfb startup timed out')
                time.sleep(.1)
            launch('viewer', [py, '-u', root/'scripts/viz_semantickitti_fovea.py', '--sequence', args.sequence,
                              '--start', str(args.start_frame), '--interval', str(args.interval), '--mode', args.mode, '--play'], env)
            launch('vnc', [native/'bin/x11vnc', '-display', display, '-rfbport', str(vnc_port),
                           '-listen', '127.0.0.1', '-localhost', '-forever', '-shared', '-nopw', '-noxdamage', '-quiet'], env)
            launch('browser', [py, '-m', 'websockify', '--web', native/'share/novnc',
                               f'127.0.0.1:{web_port}', f'127.0.0.1:{vnc_port}'], env)
            time.sleep(5)
            if not all(alive(state['processes'][n]) for n in required):
                raise RuntimeError('A demo service exited; use logs to inspect the error.')
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError('Browser endpoint failed its startup check')
        except BaseException:
            stop()
            raise
        print('VNC demo started (localhost only; background playback).')
        describe()
except (RuntimeError, OSError, ValueError) as error:
    print(f'Error: {error}', file=sys.stderr)
    sys.exit(1)
PY
