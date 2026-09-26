#!/usr/bin/env bash
# Replay all of SemanticKITTI sequence 07 through the FoveaMap VNC viewer.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode=model
interval=0.1
open_browser=1

usage() {
    cat <<'HELP'
Usage: ./run_seq07.sh [--mode model|gt-labels] [--interval SECONDS] [--no-browser]
Plays sequence 07 from frame 0 to its final scan using the existing VNC launcher.
Default: actual SalsaNext inference; the middle panel remains ground-truth labels.
  --mode gt-labels  Replay dataset labels without neural inference.
  --interval 0.1    Requested replay interval; actual FPS depends on processing.
  --no-browser      Start playback and print the URL without opening a browser.
  -h, --help        Show this help without starting anything.

Requires local sequence 07 scans/labels, the project .venv, VNC dependencies,
and SalsaNext weights for model mode. Nothing is downloaded or installed.
Restarts only the demo managed by scripts/kitti_vnc.sh.
Stop: bash scripts/kitti_vnc.sh stop
HELP
}

while (($#)); do
    case "$1" in
        --mode|--interval)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --mode ]]; then mode="$2"; else interval="$2"; fi
            shift 2 ;;
        --no-browser) open_browser=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$mode" in
    model|gt-labels) ;;
    *) printf 'Mode must be model or gt-labels.\n' >&2; exit 2 ;;
esac

printf 'Starting full sequence 07 from frame 0 (mode: %s).\n' "$mode"
bash "$PROJECT_ROOT/scripts/kitti_vnc.sh" restart 07 \
    --start-frame 0 --mode "$mode" --interval "$interval"

url="http://127.0.0.1:${KITTI_WEB_PORT:-6081}/vnc.html?autoconnect=true&resize=scale"
printf '\nOpen: %s\nSpace: pause/play. Playback stops at the final scan.\n' "$url"
printf 'Stop services: bash "%s/scripts/kitti_vnc.sh" stop\n' "$PROJECT_ROOT"

if ((open_browser)) && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
    # A desktop browser is optional; failure must not undo a working VNC session.
    (xdg-open "$url" >/dev/null 2>&1 || true) </dev/null &
fi
