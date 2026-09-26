#!/bin/bash
# ==============================================================================
# FoveaMap zero-waste GPU session launcher - run from any machine with the AWS profile configured.
#
#   bash infra/aws/launch_gpu_session.sh "python -m src.main latency-b"
#   bash infra/aws/launch_gpu_session.sh "bash infra/aws/run_fp16_session.sh"
#
# Boots the golden AMI (foveamap-gpu-ready-v1: CUDA venv + dataset + weights +
# repo already on disk, no secrets), copies credentials, refreshes the repo,
# and runs the given command - the GPU is billed for boot plus the actual
# workload and nothing else. The instance TERMINATES ITSELF (and deletes its
# volume) when the command finishes, because it is launched with
# instance-initiated-shutdown-behavior=terminate and the wrapper ends in
# shutdown.
#
# Standing rule: run this only with explicit owner approval per session -
# it creates a billed instance.
# ==============================================================================
set -euo pipefail

CMD="${1:?usage: launch_gpu_session.sh \"<command to run in ~/foveamap>\"}"
PROFILE="${AWS_PROFILE_OLD:-oldaccount}"
REGION="us-east-1"
KEY="$HOME/foveamap/infra/keys/foveamap_key"
SG_NAME="foveamap-fp16-sg"

AMI=$(aws ec2 describe-images --profile "$PROFILE" --region "$REGION" \
      --owners self --filters "Name=name,Values=foveamap-gpu-ready-v*" \
      --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text)
[ "$AMI" != "None" ] || { echo "No golden AMI found - build one first."; exit 1; }
SG=$(aws ec2 describe-security-groups --profile "$PROFILE" --region "$REGION" \
     --filters "Name=group-name,Values=$SG_NAME" --query "SecurityGroups[0].GroupId" --output text)
MYIP=$(curl -s ifconfig.me)
aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" \
    --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" 2>/dev/null || true

echo "Launching $AMI (terminate-on-shutdown)..."
IID=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
  --image-id "$AMI" --instance-type g4dn.xlarge \
  --key-name foveamap-fp16-key --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=foveamap-gpu-session}]' \
  --query "Instances[0].InstanceId" --output text)
aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$IID" \
     --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "Instance $IID at $IP - waiting for SSH..."
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$IP"
until $SSH "true" 2>/dev/null; do sleep 5; done

# Only the secrets travel at launch time - everything else is baked in.
$SSH "mkdir -p ~/.aws ~/.ssh"
scp -i "$KEY" -o StrictHostKeyChecking=no ~/.ssh/id_foveamap "ubuntu@$IP:~/.ssh/id_foveamap"
printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\nregion = us-east-1\n' \
  "$(grep -A2 '^\[default\]' ~/.aws/credentials | grep key_id | cut -d= -f2 | tr -d ' ')" \
  "$(grep -A2 '^\[default\]' ~/.aws/credentials | grep secret | cut -d= -f2 | tr -d ' ')" \
  | $SSH "cat > ~/.aws/credentials && chmod 600 ~/.aws/credentials ~/.ssh/id_foveamap"

echo "Starting workload (detached): $CMD"
$SSH "nohup bash -c 'cd ~/foveamap && git fetch origin -q && git checkout main -q && git pull --ff-only -q && source .venv/bin/activate && aws s3 sync s3://sih-perception-dataset-882040517559/weights/ weights/ --only-show-errors && ($CMD) 2>&1; aws s3 sync runs/ s3://sih-perception-dataset-882040517559/runs/gpu-sessions/ --only-show-errors; sudo shutdown -h now' > ~/session.log 2>&1 & echo LAUNCHED"
echo "Session running on $IID ($IP). It will terminate itself when done."
echo "Watch live:  ssh -i $KEY ubuntu@$IP 'tail -f ~/session.log'"