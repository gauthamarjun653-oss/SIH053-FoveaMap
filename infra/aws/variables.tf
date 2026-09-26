variable "aws_region" {
  description = "AWS region to deploy resources into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment identifier (e.g. dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for dataset storage"
  type        = string
  default     = "sih-perception-dataset-882040517559"
}

variable "iam_user_name" {
  description = "IAM user name for the perception dataset access"
  type        = string
  default     = "sih-perception-team"
}

variable "enable_gpu_instance" {
  description = "Whether to provision the GPU inference instance. Set to false for S3-only deployment"
  type        = bool
  default     = false
}

variable "instance_type" {
  description = "EC2 GPU instance type for ML batch inference (e.g. g4dn.xlarge, g5.xlarge)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "enable_spot_instance" {
  description = "Whether to request a Spot instance for ML workloads to reduce costs"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum hourly price (USD) for Spot instance. Set to null to use on-demand price as cap."
  type        = string
  default     = null
}

variable "root_volume_size_gb" {
  description = "Root EBS GP3 volume size in GB"
  type        = number
  default     = 150
}

variable "allowed_ssh_cidr" {
  description = "CIDR block permitted for inbound SSH access (set to your public IP/32)"
  type        = string
  default     = "0.0.0.0/0"
}

variable "aws_ssh_key_name" {
  description = "Name of an existing AWS Key Pair to attach to the EC2 instance"
  type        = string
  default     = ""
}

variable "aws_ssh_public_key" {
  description = "SSH public key content to create a new AWS Key Pair. If empty, aws_ssh_public_key_path will be used"
  type        = string
  default     = ""
}

variable "aws_ssh_public_key_path" {
  description = "File path to the SSH public key (default uses infra/keys/foveamap_key.pub)"
  type        = string
  default     = "../keys/foveamap_key.pub"
}

variable "custom_ami_id" {
  description = "Optional custom AMI ID. If empty, official AWS Deep Learning AMI (Ubuntu 22.04) will be dynamically queried"
  type        = string
  default     = ""
}

variable "manage_legacy_s3" {
  description = "Whether to manage legacy S3 bucket and IAM user in the old AWS account. Defaults to false since canonical bucket is in account 882040517559"
  type        = bool
  default     = false
}

variable "instance_initiated_shutdown_behavior" {
  description = "Shutdown behavior when OS shuts down (terminate or stop). Spot instances must terminate; on-demand can stop or terminate."
  type        = string
  default     = "terminate"
}

variable "auto_idle_shutdown_minutes" {
  description = "Safety timeout in minutes after boot before automatic instance poweroff if idle (0 to disable)"
  type        = number
  default     = 180
}

