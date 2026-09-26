terraform {
  backend "s3" {
    bucket       = "sih-perception-tfstate-454096720696"
    key          = "foveamap/aws/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
