# Stage 5 GPU setup — requirements checklist

**Purpose:** everything the GPU stage needs, split into what is already
prepared (nothing to do) and what must come from the owner. Fill in the
owner-side items and stages 5-7 can run end to end in one session.

Standing rule unchanged: no instance is created without explicit owner
permission; `terraform apply` is always a human decision.

## Already prepared - no action needed

| Item | State |
|---|---|
| SalsaNext weights | `weights/salsanext_state_dict.pt`, T-28-verified (weights_only load + smoke inference); local on the Azure VM and in S3 `weights/` |
| Dataset | Sequences 00 + 08 with labels/calib/poses in S3 (canonical) - the GPU node pulls them in-region, free and fast |
| Bootstrap | `infra/aws/bootstrap_gpu.sh` - idempotent: GPU sanity, account check, CUDA venv, repo clone, S3 data pull, test-suite verify |
| Terraform | `infra/aws/gpu_instance.tf` (g4dn.xlarge, Deep Learning AMI, spot by default, 150 GB gp3, gated behind `enable_gpu_instance=false`) |
| Pipeline config | `configs/pipeline.yaml`: `device: auto` picks CUDA; `fp16: false` is the first fallback if over budget |
| Measurement commands | `latency-b` (stage 5), `run --mode live` (stage 6), `benchmark --seq 08` (stage 7) - all exercised on CPU already |
| Dashboard wiring | `demoUI/wire_metrics.py <run-dir>` ingests the GPU benchmark results the moment they exist |

## Needed from the owner - the checklist to answer

### Decisions
1. **Account confirmation.** Standing decision is the OLD account
   (454096720696) for GPU; confirm it still holds. If it changes to the new
   account, say so - the terraform needs a fresh init/state there.
2. **Instance type: DECIDED (2026-08-27) - g4dn.xlarge** (4 vCPU, T4
   16 GB, ~$0.53/hr on-demand). The spec cites the T4 as the metrics
   hardware, so faster GPUs (g5/A10G) would weaken the latency story, not
   strengthen it. Fits the granted quota with headroom.
3. **Quota status: GRANTED - 8 G/VT vCPUs, us-east-1** (request for 96
   partially approved; 8 is enough for one g4dn.xlarge, even two). Note
   this is the ON-DEMAND quota; spot uses the separate "All G and VT Spot
   Instance Requests" line - if that is still 0, set
   `enable_spot_instance = false` and run on-demand (cost delta under $1
   at these session lengths).
4. **Budget/hours approval.** Estimated timeline for stages 5-7 in one
   session: boot 3-5 min, bootstrap 15-20 min, latency-b ~10 min, live
   seq-00 run 10-20 min, official benchmark 25-40 min, sync + teardown
   ~5 min - about 1.5-2 hours; budget 3 hours to cover the fp16/width
   fallback ladder and a rerun (~$1.60 on-demand at the 3-hour ceiling).

### Access material (bring tomorrow)
4. **Old-account admin credentials or console access** - whoever runs
   `terraform apply`. The Azure VM holds no old-account admin (by design),
   so apply runs from your machine, or you provide admin keys for the
   session and rotate them after.
5. **G-family vCPU quota confirmed >= 4** in us-east-1 for the chosen
   account: Service Quotas -> EC2 -> "Running On-Demand G and VT instances"
   (and "All G and VT Spot Instance Requests" if spot). New/idle accounts
   often sit at 0 and the increase takes about a day - CHECK THIS FIRST,
   it is the long pole.
6. **Your public IP** for `allowed_ssh_cidr` (find it with
   `curl ifconfig.me`; goes into `terraform.tfvars` as `<ip>/32`).
7. **SSH key**: `infra/keys/foveamap_key` present on whichever machine will
   ssh to the GPU node (the .pub side is read by terraform automatically).
8. **Two files to copy onto the GPU node after it boots** (from the Azure
   VM or your machine):
   - `~/.aws/credentials` (the NEW account's default profile - the
     old-account instance role cannot read the new bucket; the bootstrap
     aborts with instructions if this is missing)
   - `~/.ssh/id_foveamap` (GitHub deploy key, for the repo clone)

## Execution sequence once the above is provided

```bash
# owner, from a machine with old-account admin:
cd infra/aws
cp terraform.tfvars.example terraform.tfvars    # set allowed_ssh_cidr
terraform init
terraform plan  -var enable_gpu_instance=true   # review: GPU resources only
terraform apply -var enable_gpu_instance=true   # OWNER DECISION - billing starts

# copy access material to the node, then on the node:
bash infra/aws/bootstrap_gpu.sh                 # verifies GPU + account + tests

# measurements: run all 3 stages end-to-end with automated S3 sync and auto-shutdown:
bash infra/aws/run_gpu_pipeline.sh

# or individually:
python -m src.main latency-b                    # stage 5: 40 ms budget, record median/p90
python -m src.main run --seq 00 --mode live     # stage 6: live pipeline
python -m src.main benchmark --seq 08 --mode live # stage 7: the official Pareto
aws s3 sync runs/ s3://sih-perception-dataset-882040517559/runs/gpu/  # BEFORE terminating
sudo shutdown -h now                             # auto-terminates instance to stop billing

```

If latency-b exceeds 40 ms, the fallback ladder is applied one step at a
time with re-measurement: `fp16: true`, then projection width 1024, then
pruning.

Open methodology question to answer before the official stage-7 run
(from the preliminary benchmark): whether to add a like-for-like baseline
(uniform 5 cm cells with the same 84 B stats payload) alongside the spec's
skeletal baselines.
