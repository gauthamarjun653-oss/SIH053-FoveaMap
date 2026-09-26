# ==============================================================================
# AWS GPU Spot Instance for Batch ML Inference
# Project FoveaMap (SIH26053)
# ==============================================================================

# ------------------------------------------------------------------------------
# Data Source: Official AWS Deep Learning AMI (Ubuntu 22.04)
# ------------------------------------------------------------------------------
data "aws_ami" "deep_learning" {
  count       = var.enable_gpu_instance ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name = "name"
    values = [
      "Deep Learning OSS Server AMI (Ubuntu 22.04) *",
      "Deep Learning OSS Server AMI GPU (Ubuntu 22.04) *",
      "Deep Learning Base OSS Server AMI (Ubuntu 22.04) *",
      "Deep Learning AMI GPU PyTorch * (Ubuntu 22.04)*",
      "Deep Learning Base AMI (Ubuntu 22.04) *"
    ]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ------------------------------------------------------------------------------
# Networking & Security Group
# ------------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "gpu_instance_sg" {
  count       = var.enable_gpu_instance ? 1 : 0
  name        = "foveamap-gpu-inference-sg"
  description = "Security group for FoveaMap GPU Spot instance allowing restricted SSH and all outbound"
  vpc_id      = data.aws_vpc.default.id

  # Inbound SSH restricted to specified CIDR
  ingress {
    description = "SSH from authorized CIDR"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Inbound TensorBoard / Web UI
  ingress {
    description = "TensorBoard / API Dashboard from authorized CIDR"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # All outbound traffic allowed
  egress {
    description      = "Allow all outbound traffic"
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "foveamap-gpu-sg"
  }
}

# ------------------------------------------------------------------------------
# IAM Role & Instance Profile for S3 Data Access
# ------------------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume_role_policy" {
  statement {
    sid     = "EC2AssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu_instance_role" {
  count              = var.enable_gpu_instance ? 1 : 0
  name               = "foveamap-gpu-instance-role"
  description        = "EC2 role granting GPU instance direct read/write access to S3 perception dataset"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role_policy.json
}

# Scoped IAM Policy Document for S3 Perception Dataset
data "aws_iam_policy_document" "gpu_s3_access_policy_doc" {
  count = var.enable_gpu_instance ? 1 : 0

  statement {
    sid    = "AllowBucketListing"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = [
      "arn:aws:s3:::${var.s3_bucket_name}"
    ]
  }

  statement {
    sid    = "AllowObjectOperations"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ]
    resources = [
      "arn:aws:s3:::${var.s3_bucket_name}/*"
    ]
  }
}

resource "aws_iam_policy" "gpu_s3_policy" {
  count       = var.enable_gpu_instance ? 1 : 0
  name        = "foveamap-gpu-s3-access-policy"
  description = "Scoped S3 access policy for FoveaMap GPU instance"
  policy      = data.aws_iam_policy_document.gpu_s3_access_policy_doc[0].json
}

# Attach dataset policy directly to the GPU instance role
resource "aws_iam_role_policy_attachment" "gpu_s3_policy_attachment" {
  count      = var.enable_gpu_instance ? 1 : 0
  role       = aws_iam_role.gpu_instance_role[0].name
  policy_arn = aws_iam_policy.gpu_s3_policy[0].arn
}

resource "aws_iam_instance_profile" "gpu_instance_profile" {
  count = var.enable_gpu_instance ? 1 : 0
  name  = "foveamap-gpu-instance-profile"
  role  = aws_iam_role.gpu_instance_role[0].name
}

# ------------------------------------------------------------------------------
# SSH Key Pair (Uses shared infra/keys/foveamap_key.pub)
# ------------------------------------------------------------------------------
locals {
  resolved_aws_pub_key = var.aws_ssh_public_key != "" ? var.aws_ssh_public_key : (
    fileexists("${path.module}/${var.aws_ssh_public_key_path}") ? file("${path.module}/${var.aws_ssh_public_key_path}") : ""
  )
}

resource "aws_key_pair" "generated_key" {
  count      = (var.enable_gpu_instance && var.aws_ssh_key_name == "" && local.resolved_aws_pub_key != "") ? 1 : 0
  key_name   = "foveamap-shared-key"
  public_key = local.resolved_aws_pub_key
}

locals {
  selected_key_name = var.aws_ssh_key_name != "" ? var.aws_ssh_key_name : (
    length(aws_key_pair.generated_key) > 0 ? aws_key_pair.generated_key[0].key_name : null
  )
  selected_ami = var.custom_ami_id != "" ? var.custom_ami_id : (
    length(data.aws_ami.deep_learning) > 0 ? data.aws_ami.deep_learning[0].id : null
  )
}

# ------------------------------------------------------------------------------
# GPU Instance (Spot Instance with 150 GB GP3 EBS storage)
# ------------------------------------------------------------------------------
resource "aws_instance" "gpu_inference_node" {
  count                = var.enable_gpu_instance ? 1 : 0
  ami                  = local.selected_ami
  instance_type        = var.instance_type
  key_name             = local.selected_key_name
  iam_instance_profile = aws_iam_instance_profile.gpu_instance_profile[0].name

  instance_initiated_shutdown_behavior = var.instance_initiated_shutdown_behavior

  vpc_security_group_ids = [aws_security_group.gpu_instance_sg[0].id]

  # Root volume: 150 GB GP3 EBS for storing extracted datasets & models
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gb
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "foveamap-gpu-root-volume"
    }
  }

  # Dynamic Spot instance configuration
  dynamic "instance_market_options" {
    for_each = var.enable_spot_instance ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price          = var.spot_max_price
        spot_instance_type = "one-time"
      }
    }
  }

  user_data = <<-EOF
              #!/bin/bash
              set -e
              exec > >(tee -a /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
              echo "=== Initializing FoveaMap GPU Node ==="
              nvidia-smi || echo "NVIDIA driver initializing..."
              if ! command -v aws &> /dev/null; then
                apt-get update -y && apt-get install -y unzip curl
                curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
                unzip -q awscliv2.zip
                ./aws/install
              fi

              # Safety auto-shutdown timeout (can be cancelled anytime via 'sudo shutdown -c')
              if [ "${var.auto_idle_shutdown_minutes}" -gt 0 ]; then
                echo "Scheduling safety auto-poweroff after ${var.auto_idle_shutdown_minutes} minutes..."
                shutdown -h +${var.auto_idle_shutdown_minutes} "Safety timeout: shutting down after ${var.auto_idle_shutdown_minutes} minutes" &
              fi

              echo "FoveaMap GPU Node ready."
              EOF

  tags = {
    Name        = "foveamap-gpu-inference-worker"
    Workload    = "Batch-ML-Inference"
    SpotEnabled = tostring(var.enable_spot_instance)
  }
}
