# ==============================================================================
# AWS Outputs
# Project FoveaMap (SIH26053)
# ==============================================================================

# ------------------------------------------------------------------------------
# S3 Dataset Outputs
# ------------------------------------------------------------------------------
output "s3_bucket_name" {
  description = "Name of the perception dataset S3 bucket"
  value       = var.s3_bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the perception dataset S3 bucket"
  value       = "arn:aws:s3:::${var.s3_bucket_name}"
}

# ------------------------------------------------------------------------------
# IAM Team Access Credentials (Legacy S3 mode)
# ------------------------------------------------------------------------------
output "iam_user_name" {
  description = "Name of the IAM user created for dataset access"
  value       = length(aws_iam_user.perception_team_user) > 0 ? aws_iam_user.perception_team_user[0].name : "Legacy IAM user not managed (canonical bucket in account 882040517559)"
}

output "iam_access_key_id" {
  description = "AWS Access Key ID for perception team data ingestion"
  value       = length(aws_iam_access_key.perception_team_key) > 0 ? aws_iam_access_key.perception_team_key[0].id : null
}

output "iam_secret_access_key" {
  description = "AWS Secret Access Key for perception team data ingestion"
  value       = length(aws_iam_access_key.perception_team_key) > 0 ? aws_iam_access_key.perception_team_key[0].secret : null
  sensitive   = true
}

# ------------------------------------------------------------------------------
# GPU Instance Outputs (Conditional)
# ------------------------------------------------------------------------------
output "gpu_instance_id" {
  description = "EC2 Instance ID of the GPU worker"
  value       = length(aws_instance.gpu_inference_node) > 0 ? aws_instance.gpu_inference_node[0].id : "GPU Instance not provisioned"
}

output "gpu_instance_public_ip" {
  description = "Public IPv4 address of the GPU instance"
  value       = length(aws_instance.gpu_inference_node) > 0 ? aws_instance.gpu_inference_node[0].public_ip : null
}

output "gpu_ssh_command" {
  description = "SSH connection string to access the GPU instance using shared key"
  value = length(aws_instance.gpu_inference_node) > 0 ? (
    "ssh -i ../keys/foveamap_key ubuntu@${aws_instance.gpu_inference_node[0].public_ip}"
  ) : "GPU instance not provisioned"
}

output "s3_sync_command" {
  description = "AWS CLI S3 sync command for dataset download"
  value       = "aws s3 sync s3://${var.s3_bucket_name}/ ./data/"
}

output "auto_shutdown_info" {
  description = "Information on auto-shutdown and instance termination"
  value       = "Instance configured with instance_initiated_shutdown_behavior='${var.instance_initiated_shutdown_behavior}' and safety timeout ${var.auto_idle_shutdown_minutes}m. Running 'sudo shutdown -h now' terminates/stops the instance and halts billing."
}

