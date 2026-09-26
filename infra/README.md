# AWS Infrastructure for Project FoveaMap (SIH26053)

Adaptive Variable-Resolution 2.5D Lidar Mapping System — Infrastructure as Code (IaC) powered by Terraform.

> **WARNING - account migration (2026-08-27):** the canonical dataset bucket
> is `sih-perception-dataset-882040517559` in the NEW AWS account
> (882040517559, `[default]` profile). The S3 resources in `infra/aws/s3.tf`
> are legacy definitions for the retired OLD account (454096720696), not the
> Terraform configuration for the current bucket. The old AWS state files have
> been removed. Do NOT run `terraform apply` or `terraform destroy` for this S3
> configuration against the current bucket. Managing the current bucket with
> Terraform requires a separately reviewed configuration and state.

---

## AWS Account and Terraform State Layout

- **OLD AWS account `454096720696`**: GPU infrastructure and the Terraform
	remote-state bucket `sih-perception-tfstate-454096720696` in `us-east-1`.
- **NEW AWS account `882040517559`**: free-tier account holding the canonical
	dataset bucket `sih-perception-dataset-882040517559` for normal data tasks.

The dataset bucket in the NEW account is externally managed and is not the
Terraform state bucket. The remote state bucket in the OLD account was created
with the AWS CLI and is used only for Terraform state and lock files:

- AWS state: `foveamap/aws/terraform.tfstate`

The state backend is a private, versioned, AES256-encrypted S3 bucket with
public access blocked. The Azure VM and its Terraform configuration were retired
on 2026-09-24; only the AWS GPU configuration remains.

---

## Shared SSH Key Pair
A unified, dedicated RSA key pair is generated in [`infra/keys/`](file:///home/pravinraj/foveamap/infra/keys):
- **Private Key**: `infra/keys/foveamap_key` (protected `chmod 600`, ignored by `.gitignore`)
- **Public Key**: `infra/keys/foveamap_key.pub` (AWS GPU node and GitHub deploy key)

---

## 1. AWS S3 & Team Access (`infra/aws/`)

### Current Dataset (Externally Managed)
- **Bucket**: `sih-perception-dataset-882040517559` (`us-east-1`)
- **Bucket Versioning**: Enabled
- **Public Access Block**: All public access blocked
- **Default Encryption**: SSE AES256
- **Management**: Created and maintained outside the legacy Terraform config

The current bucket is the canonical dataset store. The S3 definitions in
`infra/aws/s3.tf` are retained only as historical legacy configuration and are
not authoritative for this bucket.

### Team Credentials
- **Access Key ID**: (retrieve via `terraform -chdir=infra/aws output -raw iam_access_key_id`)
- **Secret Access Key**: (retrieve via `terraform -chdir=infra/aws output -raw iam_secret_access_key`)

