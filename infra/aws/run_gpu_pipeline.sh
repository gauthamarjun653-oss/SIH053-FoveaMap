#!/bin/bash
# ==============================================================================
# FoveaMap End-to-End GPU Pipeline Runner & Auto-Shutdown
# Project FoveaMap (SIH26053)
#
# Runs Stage 5 (latency-b), Stage 6 (live seq 00), Stage 7 (benchmark seq 08),
# syncs artifacts to S3, and automatically shuts down/terminates the instance
# to eliminate idle costs.
#
# Usage:
#   bash infra/aws/run_gpu_pipeline.sh              # Runs all stages + auto-shutdown
#   bash infra/aws/run_gpu_pipeline.sh --no-shutdown # Runs all stages but keeps VM up
# ==============================================================================
set -euo pipefail

AUTO_SHUTDOWN=true
for arg in "$@"; do
    case "$arg" in
        --no-shutdown|--keep-alive)
            AUTO_SHUTDOWN=false
            shift
            ;;
        *)
            ;;
    esac
done

REPO_DIR="$HOME/foveamap"
BUCKET="s3://sih-perception-dataset-882040517559"
RUNS_DIR="$REPO_DIR/runs"

cd "$REPO_DIR"
mkdir -p "$RUNS_DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found. Run bootstrap_gpu.sh first."
    exit 1
fi

source .venv/bin/activate

# Safety trap to ensure artifacts are synced to S3 even if a stage fails
cleanup_and_sync() {
    local exit_code=$?
    echo ""
    echo "=== Syncing artifacts to S3 canonical store ==="
    if [ -d "$RUNS_DIR" ]; then
        aws s3 sync "$RUNS_DIR" "$BUCKET/runs/gpu/" --only-show-errors || echo "Warning: S3 sync encountered an error."
        echo "Artifacts synced to $BUCKET/runs/gpu/"
    fi

    if [ "$AUTO_SHUTDOWN" = true ]; then
        echo ""
        echo "================================================================="
        echo " Auto-shutdown is enabled. Instance will power off in 30 seconds."
        echo " Press Ctrl+C within 30s to cancel shutdown."
        echo "================================================================="
        sleep 30
        echo "Initiating shutdown now..."
        sudo shutdown -h now
    else
        echo "Auto-shutdown disabled (--no-shutdown). Instance remains running."
    fi
    exit $exit_code
}
trap cleanup_and_sync EXIT

echo "================================================================="
echo " Starting FoveaMap GPU Stages 5 -> 6 -> 7"
echo " Time: $(date -u)"
echo " Auto-shutdown on finish: $AUTO_SHUTDOWN"
echo "================================================================="

# --- Stage 5: Module B latency ---
echo ""
echo "=== [1/3] STAGE 5: Module B Latency Measurement (latency-b) ==="
echo "Measuring 50 iterations with pretrained weights..."
python -m src.main latency-b --iters 50 | tee "$RUNS_DIR/stage5_latency.txt"

# --- Stage 6: Live pipeline accuracy ---
echo ""
echo "=== [2/3] STAGE 6: Live Pipeline Execution (Seq 00) ==="
echo "Running live pipeline with SalsaNext on sequence 00..."
python -m src.main run --seq 00 --mode live | tee "$RUNS_DIR/stage6_live_seq00.log"

# --- Stage 7: Official Benchmark & Pareto ---
echo ""
echo "=== [3/3] STAGE 7: Official Live Benchmark (Seq 08) ==="
echo "Running official benchmark on validation sequence 08..."
python -m src.main benchmark --seq 08 --mode live | tee "$RUNS_DIR/stage7_benchmark_seq08.log"

# Find latest benchmark run directory and wire metrics into demoUI
LATEST_BENCHMARK=$(find "$RUNS_DIR" -maxdepth 1 -type d -name "seq08_live_*" | sort -r | head -n 1)
if [ -n "$LATEST_BENCHMARK" ] && [ -f "$LATEST_BENCHMARK/results.json" ]; then
    echo "Wiring latest benchmark results ($LATEST_BENCHMARK) into demoUI..."
    python demoUI/wire_metrics.py "$LATEST_BENCHMARK" || true
fi

echo ""
echo "=== All GPU Stages 5-7 Completed Successfully ==="
