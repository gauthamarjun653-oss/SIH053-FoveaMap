#!/bin/bash
# ==============================================================================
# FoveaMap fp16 follow-up session - run on the GPU node after bootstrap_gpu.sh.
# Purpose (stage 5 fallback ladder + stage 9 provenance repair, ~15 min):
#   1. Stage 5 came in at 50.6 ms median vs the 40 ms budget with fp32.
#      The spec's ordered response is fp16 FIRST, then width 1024, then
#      pruning - one step at a time, re-measuring after each.
#   2. The 2026-08-27 official benchmark ran from a dirty tree
#      (provenance commit 9dd9061-dirty). Re-running from a clean checkout
#      stamps clean provenance for the stage-9 reproducibility pass.
# The instance itself is created by the owner (terraform apply); this script
# never provisions anything.
# ==============================================================================
set -euo pipefail

REPO_DIR="$HOME/foveamap"
BUCKET="s3://sih-perception-dataset-882040517559"
cd "$REPO_DIR"
source .venv/bin/activate

echo "=== [1/5] Clean-tree check (stage-9 provenance) ==="
git fetch origin -q && git checkout main -q && git pull --ff-only -q
if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is DIRTY - provenance would be stamped -dirty again."
    git status --short
    exit 1
fi
echo "Tree clean at $(git rev-parse --short HEAD)."

echo "=== [2/5] Verify fp16 is on (committed as fallback rung 1 - editing it"
echo "here would dirty the tree and ruin the provenance this session exists to fix) ==="
grep -q "fp16: true" configs/pipeline.yaml || { echo "fp16 is not enabled in the committed config - fix in the repo, not here."; exit 1; }
grep -n "fp16" configs/pipeline.yaml | head -1

echo "=== [3/5] Stage 5 re-measurement (fp16) ==="
mkdir -p runs   # gitignored, absent on a fresh clone - a failing tee here
                # killed the 2026-08-28 session via set -e before shutdown
python -m src.main latency-b | tee runs/stage5_latency_fp16.txt
MED=$(grep -oP 'median \K[0-9.]+' runs/stage5_latency_fp16.txt | head -1)
echo "fp16 median: ${MED} ms (budget 40)"
if python3 -c "import sys; sys.exit(0 if float('${MED}') <= 40.0 else 1)"; then
    echo "UNDER BUDGET - ladder stops here. Keep fp16: true and COMMIT the config change."
else
    echo "STILL OVER BUDGET - next rung is projection width 1024 (configs/pipeline.yaml)."
    echo "Do not proceed to the benchmark until a rung passes; record this result regardless."
fi

echo "=== [4/5] Official benchmark from the clean tree ==="
python -m src.main benchmark --seq 08 --mode live 2>&1 | tail -5

echo "=== [5/5] Sync artifacts, then shut down ==="
aws s3 sync runs/ "$BUCKET/runs/gpu-fp16/" --only-show-errors
echo "Synced. Powering off (STOPS billing for compute; owner must still"
echo "TERMINATE the instance afterwards - a stopped instance keeps billing EBS)."
sudo shutdown -h now
