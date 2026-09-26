# ==============================================================================
# S3 Dataset Storage & IAM Configuration
# Project FoveaMap (SIH26053)
# ==============================================================================
# LEGACY CONFIGURATION: this file describes the retired old-account S3/IAM
# resources. The current canonical bucket is managed outside this Terraform
# configuration in AWS account 882040517559. Do not apply this S3 configuration
# to the current bucket without creating and reviewing a separate state.

# S3 Bucket for Lidar Perception Dataset
resource "aws_s3_bucket" "perception_dataset" {
  count         = var.manage_legacy_s3 ? 1 : 0
  bucket        = var.s3_bucket_name
  force_destroy = false

  tags = {
    Name        = var.s3_bucket_name
    Description = "Lidar Point Cloud and KITTI Perception Dataset for FoveaMap"
  }
}

# Bucket Versioning
resource "aws_s3_bucket_versioning" "perception_dataset_versioning" {
  count  = var.manage_legacy_s3 ? 1 : 0
  bucket = aws_s3_bucket.perception_dataset[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-Side Encryption by default (AES256)
resource "aws_s3_bucket_server_side_encryption_configuration" "perception_dataset_encryption" {
  count  = var.manage_legacy_s3 ? 1 : 0
  bucket = aws_s3_bucket.perception_dataset[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block all Public Access
resource "aws_s3_bucket_public_access_block" "perception_dataset_pab" {
  count  = var.manage_legacy_s3 ? 1 : 0
  bucket = aws_s3_bucket.perception_dataset[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ------------------------------------------------------------------------------
# Team IAM User & Programmatic Access
# ------------------------------------------------------------------------------

# IAM User for team data sync & local perception pipeline
resource "aws_iam_user" "perception_team_user" {
  count = var.manage_legacy_s3 ? 1 : 0
  name  = var.iam_user_name
  path  = "/system/"

  tags = {
    Purpose = "Team dataset access for FoveaMap Lidar ingestion and pipeline"
  }
}

# Programmatic Access Key for IAM User
resource "aws_iam_access_key" "perception_team_key" {
  count = var.manage_legacy_s3 ? 1 : 0
  user  = aws_iam_user.perception_team_user[0].name
}

# IAM Policy Document scoped strictly to perception dataset bucket and its objects
data "aws_iam_policy_document" "s3_access_policy_doc" {
  count = var.manage_legacy_s3 ? 1 : 0

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

# IAM Policy for S3 access
resource "aws_iam_policy" "perception_dataset_policy" {
  count       = var.manage_legacy_s3 ? 1 : 0
  name        = "${var.iam_user_name}-s3-access-policy"
  description = "Scoped S3 access policy for FoveaMap perception dataset (${var.s3_bucket_name})"
  policy      = data.aws_iam_policy_document.s3_access_policy_doc[0].json
}

# Attach IAM Policy to IAM User
resource "aws_iam_user_policy_attachment" "perception_team_attachment" {
  count      = var.manage_legacy_s3 ? 1 : 0
  user       = aws_iam_user.perception_team_user[0].name
  policy_arn = aws_iam_policy.perception_dataset_policy[0].arn
}

