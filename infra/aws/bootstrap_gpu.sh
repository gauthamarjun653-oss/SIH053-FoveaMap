#!/bin/bash
# ==============================================================================
# FoveaMap AWS GPU node bootstrap — idempotent, run as ubuntu on the g4dn.
# Prereq: the instance was created by infra/aws/gpu_instance.tf, so S3 access
# comes from the instance IAM role (no credentials file needed). Cloning the
# repo needs the GitHub deploy key at ~/.ssh/id_foveamap — copy it from an
# authorized machine first (never commit it).
#
# Stage-5 rule: never create/resize/destroy the instance itself without
# explicit owner permission. This script only prepares an ALREADY-approved,
# already-running instance.
# ==============================================================================
set -euo pipefail

REPO_SSH="git@github.com:pravinraj213/sih26053-foveamap.git"
REPO_DIR="$HOME/foveamap"
DATA_ROOT="$HOME/kitti"
BUCKET="s3://sih-perception-dataset-882040517559"

echo "=== [1/6] GPU sanity ==="
nvidia-smi || { echo "No NVIDIA driver — wrong AMI or driver still initializing."; exit 1; }

# The GPU lives in the OLD account but the data lives in the NEW account's
# bucket (decision 2026-08-27), so the instance role is NOT enough: this node
# needs the new account's credentials copied on before bootstrapping.
ACCT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo none)
if [ "$ACCT" != "882040517559" ]; then
    echo "S3 access is '$ACCT', expected the new data account 882040517559."
    echo "Copy the credentials first:  scp ~/.aws/credentials ubuntu@<this-host>:~/.aws/credentials"
    exit 1
fi

echo "=== [2/6] System packages ==="
sudo apt-get update -y -qq
sudo apt-get install -y -qq git python3-pip python3-venv unzip htop tmux
if ! command -v aws >/dev/null; then
    cd /tmp && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
    unzip -q awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip
fi

echo "=== [3/6] GitHub SSH config + repo ==="
mkdir -p ~/.ssh && chmod 700 ~/.ssh
grep -q "^github.com" ~/.ssh/known_hosts 2>/dev/null \
    || ssh-keyscan -t ed25519,rsa github.com 2>/dev/null >> ~/.ssh/known_hosts
if ! grep -q "Host github.com" ~/.ssh/config 2>/dev/null; then
    printf "Host github.com\n  IdentityFile ~/.ssh/id_foveamap\n  IdentitiesOnly yes\n" >> ~/.ssh/config
    chmod 600 ~/.ssh/config
fi
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
elif [ -d "$REPO_DIR" ]; then
    git clone "$REPO_SSH" /tmp/foveamap_clone
    cp -rn /tmp/foveamap_clone/. "$REPO_DIR/"
    rm -rf /tmp/foveamap_clone
else
    git clone "$REPO_SSH" "$REPO_DIR"
fi


echo "=== [4/6] Python environment (CUDA torch) ==="
cd "$REPO_DIR"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/pip install -q --upgrade pip
# Default PyPI wheel ships CUDA support — no CPU-only index override here
# (that override was the retired CPU VM's trick; using it here would waste the GPU).
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch sees no CUDA device — check driver/AMI"
print(f"torch {torch.__version__}, CUDA OK: {torch.cuda.get_device_name(0)}")
PY

echo "=== [5/6] Data from S3 (in-region, free transfer) ==="
mkdir -p "$DATA_ROOT/sequences"
for seq in 00 08; do
    mkdir -p "$DATA_ROOT/sequences/$seq"
    [ -f "$DATA_ROOT/sequences/$seq/calib.txt" ] || aws s3 cp "$BUCKET/sequences/$seq/calib.txt" "$DATA_ROOT/sequences/$seq/calib.txt"
    # Archive layout ships poses at top level; spec layout wants sequences/XX/poses.txt.
    [ -f "$DATA_ROOT/sequences/$seq/poses.txt" ] || aws s3 cp "$BUCKET/poses/$seq.txt" "$DATA_ROOT/sequences/$seq/poses.txt"
    aws s3 sync "$BUCKET/sequences/$seq/velodyne/" "$DATA_ROOT/sequences/$seq/velodyne/" --only-show-errors
    aws s3 sync "$BUCKET/sequences/$seq/labels/"   "$DATA_ROOT/sequences/$seq/labels/"   --only-show-errors
done
mkdir -p "$REPO_DIR/weights"
[ -f "$REPO_DIR/weights/salsanext_state_dict.pt" ] \
    || aws s3 cp "$BUCKET/weights/salsanext_state_dict.pt" "$REPO_DIR/weights/salsanext_state_dict.pt"

echo "=== [6/6] Verify ==="
.venv/bin/python -m pytest tests/ -q
echo ""
echo "========================================================================"
echo "Bootstrap complete. You can run all stages (5 -> 6 -> 7) end-to-end"
echo "with automated S3 sync and auto-shutdown:"
echo "  bash infra/aws/run_gpu_pipeline.sh"
echo ""
echo "Or run individual stages interactively:"
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  python -m src.main latency-b"
echo "  python -m src.main run --seq 00 --mode live"
echo "  python -m src.main benchmark --seq 08 --mode live"
echo "  aws s3 sync runs/ $BUCKET/runs/gpu/"
echo "  sudo shutdown -h now"
echo "========================================================================"

