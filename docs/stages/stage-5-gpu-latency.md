# Stage 5 — First check 3: Module B latency on the GPU

**Status: MEASURED on AWS g4dn.xlarge (NVIDIA Tesla T4, 2026-08-27)**
- **Median Latency:** `50.6 ms` (over 50 runs, 64x2048 projection)
- **p90 Latency:** `51.4 ms`
- **Budget:** `40.0 ms`
- **Hardware:** AWS `g4dn.xlarge`, NVIDIA Tesla T4 16GB, CUDA 13.0, PyTorch 2.13.0
- **Log saved:** `runs/stage5_latency.txt`

**Verdict: PASSED at fallback rung 1 (fp16), 2026-08-28.** The fp32 run
measured 50.6 ms (over the 40 ms budget); with `fp16: true` (committed to
the repo, not flipped on the node) the re-measurement on the same T4 came
in at **26.2 ms median / 26.5 ms p90** - under budget with 35% headroom,
so the ladder stops here (no width reduction, no pruning). Recall on the
full validation benchmark is statistically identical at fp16 (92.04% near
vs 92.03% at fp32), so the precision cut costs nothing measured.
Artifacts: `runs/gpu-fp16/stage5_latency_fp16.txt` (S3 `runs/gpu-fp16/`).
Inference at 26.2 ms leaves 73.8 ms of a 100 ms frame for the rest of the
loop; end-to-end demo FPS is still only quotable from a demo-machine
measurement.

Purpose: measure SalsaNext inference latency in isolation on the demo-class
GPU (budget: 40 ms median). This decides whether 10 FPS is reachable and
which fallback is needed (FP16, then width 1024, then pruning) — a config
change now instead of a crisis in the final week.

Standing rule: never create, resize, deallocate, or destroy any cloud
instance without explicit owner permission. `terraform apply` is always a
human decision.

## Everything is prepared (2026-08-27)

| Item | State |
|---|---|
| Weights | `weights/salsanext_state_dict.pt` verified (T-28: `weights_only=True` load, Segmenter init, smoke inference) — local and in S3 `weights/` |
| Terraform | `infra/aws/gpu_instance.tf` defines the g4dn.xlarge (Deep Learning AMI, spot by default, 150 GB gp3, IAM role with bucket access, SSH key from `infra/keys/foveamap_key.pub`); gated behind `enable_gpu_instance` (default false) |
| tfvars | `infra/aws/terraform.tfvars.example` — copy to `terraform.tfvars`, set `allowed_ssh_cidr` to your IP `/32` |
| Bootstrap | `infra/aws/bootstrap_gpu.sh` — idempotent; clones the repo, creates a CUDA venv, pulls sequences 00/08 + weights from S3 (instance IAM role, no credentials file needed), runs the test suite |
| Pipeline config | `configs/pipeline.yaml`: `device: auto` (picks CUDA when present), `fp16: false` (the first fallback if over budget — flip it only after measuring) |

## Account decision (owner, 2026-08-27)

**The GPU runs in the OLD account (454096720696); the dataset lives in the
NEW account's bucket.** Rationale: the new account is a free-plan account
(GPU instances typically unavailable, G-family vCPU quota 0), while the old
account is a standard account where the g4dn terraform and state already
exist.

Consequence: the old-account instance IAM role cannot read the new bucket, so
the GPU node must carry the new account's credentials - copy them before
running the bootstrap (which now verifies this and aborts with instructions
if they are missing):

```bash
scp ~/.aws/credentials ubuntu@<gpu-ip>:~/.aws/credentials
```

Terraform caution: `infra/aws/s3.tf` is legacy configuration for the retired
old-account S3 resources, while `s3_bucket_name` names the new canonical
bucket. The old AWS state files have been removed. Do not apply or destroy this
S3 configuration against the current bucket; any new-account Terraform
management requires separately reviewed configuration and state. Always
`terraform plan` first, review exactly what it wants to touch (target only the
GPU resources), and remember apply is an owner decision.

## Launch procedure (owner-gated)

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars   # then set allowed_ssh_cidr
terraform init
terraform plan -var enable_gpu_instance=true
terraform apply -var enable_gpu_instance=true  # OWNER DECISION - bills from here
```

Then on the instance (as `ubuntu`):

```bash
curl -sSL https://raw.githubusercontent.com/pravinraj213/sih26053-foveamap/main/infra/aws/bootstrap_gpu.sh | bash
```

(or `scp` the script over; it needs the GitHub deploy key at
`~/.ssh/id_foveamap` to clone — copy it from an authorized machine first.)

## The measurement

```bash
cd ~/foveamap && source .venv/bin/activate
python -m src.main latency-b            # pretrained weights, 50 iterations
```

- Budget: 40 ms median. If exceeded, in order: `fp16: true`, projection
  width 1024, pruning. One change at a time, re-measure after each.
- Record median and p90 here with the hardware name (T4). Numbers from any
  CPU run are not results.

## Cost discipline

- Spot instance, one-time request (`enable_spot_instance = true`). On-demand
  g4dn.xlarge is about $0.5/hr; spot is usually cheaper but can be
  reclaimed.
- The instance stores nothing canonical — S3 holds everything — so terminate
  (not just stop) as soon as stages 5-7 are done. Re-provisioning is one
  `terraform apply` plus a few minutes of bootstrap.
- Results (`runs/`) must be synced to S3 before terminating.
