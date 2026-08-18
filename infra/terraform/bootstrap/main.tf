# Bootstrap: run ONCE, locally, with admin credentials (aws sso login).
#
# Local state on purpose — this module creates the CI deploy role and the
# demo-data bucket that everything else depends on. Terraform state for the
# demo stack itself lives in the data-qa tfstate bucket (same account), so
# no new state bucket is created here.
#
#   cd infra/terraform/bootstrap
#   terraform init && terraform apply
#
# The GitHub OIDC *provider* already exists in this account (created by
# data-qa-agent's bootstrap); this module only adds a role trusting this
# repo. Deleting data-qa would take the provider with it — recreate it there
# or move the resource here if that ever happens.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "ap-southeast-2"
}

variable "project" {
  type    = string
  default = "yt-agent"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the deploy role."
  type        = string
  default     = "nmp-dsci/transcript-rag-agent"
}

data "aws_caller_identity" "current" {}

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

# ── Demo data bucket ─────────────────────────────────────────────────────
# The Chroma index and knowledge-graph snapshot are dev state, not git
# content, but CI needs them to build the demo image. `scripts/
# upload_demo_data.sh` syncs them here; the deploy workflow pulls them into
# its build context.

resource "aws_s3_bucket" "demo_data" {
  bucket = "${var.project}-demo-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "demo_data" {
  bucket                  = aws_s3_bucket.demo_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "demo_data" {
  bucket = aws_s3_bucket.demo_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ── CI deploy role (GitHub OIDC, no stored keys) ─────────────────────────

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only this repo — any branch/workflow ref within it, so
    # workflow_dispatch from a branch under review still works.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${var.project}-github-deploy"
  description        = "Assumed by GitHub Actions (OIDC) to deploy the ${var.project} demo."
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

# The role manages the demo stack end to end through Terraform (ECR, App
# Runner, alarms) plus the two S3 touchpoints (its own state key, the demo
# data). Scoped to this project's name prefix where the service supports it.
data "aws_iam_policy_document" "deploy" {
  statement {
    sid = "TerraformStateKey"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${var.tfstate_bucket}",
      "arn:aws:s3:::${var.tfstate_bucket}/${var.project}/*",
    ]
  }

  statement {
    sid = "DemoData"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]
    resources = [
      aws_s3_bucket.demo_data.arn,
      "${aws_s3_bucket.demo_data.arn}/*",
    ]
  }

  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrRepo"
    actions = [
      "ecr:*",
    ]
    resources = [
      "arn:aws:ecr:${var.region}:${data.aws_caller_identity.current.account_id}:repository/${var.project}-*",
    ]
  }

  statement {
    sid       = "AppRunner"
    actions   = ["apprunner:*"]
    resources = ["*"]
  }

  statement {
    sid = "IamForServiceRoles"
    actions = [
      "iam:GetRole",
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:TagRole",
      "iam:PassRole",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:GetRolePolicy",
      "iam:CreateServiceLinkedRole",
    ]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project}-*",
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-service-role/*",
    ]
  }

  statement {
    sid = "Observability"
    actions = [
      "cloudwatch:PutMetricAlarm",
      "cloudwatch:DeleteAlarms",
      "cloudwatch:DescribeAlarms",
      "cloudwatch:ListTagsForResource",
      "cloudwatch:TagResource",
      "logs:CreateLogGroup",
      "logs:DescribeLogGroups",
      "logs:PutRetentionPolicy",
      "logs:ListTagsForResource",
      "logs:TagResource",
    ]
    resources = ["*"]
  }
}

variable "tfstate_bucket" {
  description = "Existing S3 bucket holding Terraform state (created by data-qa's bootstrap)."
  type        = string
  default     = "data-qa-tfstate-089783391188"
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

output "deploy_role_arn" {
  description = "Set as AWS_DEPLOY_ROLE_ARN (or hardcode) in deploy-aws.yml."
  value       = aws_iam_role.github_deploy.arn
}

output "demo_data_bucket" {
  value = aws_s3_bucket.demo_data.id
}
