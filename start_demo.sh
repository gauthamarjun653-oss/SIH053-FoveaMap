#!/usr/bin/env bash
# ==============================================================================
# FoveaMap — 1-Click SIH Live Demo & Presentation Launcher
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8080

echo "=================================================================="
echo "          FOVEAMAP — SIH26053 LIVE SOFTWARE DEMO SUITE           "
echo "=================================================================="
echo " Starting local demo web server on port ${PORT}..."
echo ""
echo " Available Endpoints:"
echo "   1. Interactive 3D / 2.5D Split Demo: http://localhost:${PORT}/"
echo "   2. Official SIH Pitch Deck:         http://localhost:${PORT}/pitch.html"
echo ""
echo " Optional 3D Rerun Visualizer Command:"
echo "   rerun ${DIR}/runs/viz_seq00_gt.rrd"
echo ""
echo " Press Ctrl+C to stop the demo server."
echo "=================================================================="

cd "${DIR}/demoUI"
python3 -m http.server ${PORT}
