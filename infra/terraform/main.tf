# ============================================================
# ChurnGuard — infrastructure de production (AWS, eu-north-1)
#
#   EC2 (m7i-flex.large)   : API FastAPI + MLflow + Airflow (webserver, scheduler) + PostgreSQL via docker compose (docker/prod/)
#   S3                     : données processées (référence, fenêtre incoming, hold-out), artefacts MLflow
#   IAM                    : rôle d'instance (S3 + SSM), user `github-deploy` (déploiement continu via SSM, aucun port SSH ouvert aux runners)
#   PostgreSQL             : conteneur sur l'EC2 (volume EBS chiffré) — RDS bloqué par le quota du plan gratuit (1 instance, déjà utilisée)
#
#   cd infra/terraform && terraform init && terraform apply
#   terraform destroy      ← après la soutenance (≈ 2,5 $/jour sinon)
# ============================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.70" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
    tls    = { source = "hashicorp/tls", version = "~> 4.0" }
    http   = { source = "hashicorp/http", version = "~> 3.4" }
    local  = { source = "hashicorp/local", version = "~> 2.5" }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "churnguard", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "me" {}

# ───────────────────────── Réseau : VPC par défaut ─────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# IP publique de l'opérateur (utilisée seulement si operator_cidr = "")
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

locals {
  my_cidr = var.operator_cidr != "" ? var.operator_cidr : "${chomp(data.http.my_ip.response_body)}/32"
  name    = "churnguard"
  ui_ports = {
    api       = 8000
    mlflow    = 5000
    airflow   = 8080
    dashboard = 8501
  }
  # Fichiers poussés dans S3 puis synchronisés sur l'instance au boot (data/processed/, non versionnés dans git)
  data_files = {
    "features_engineered.csv"      = "${var.data_dir}/features_engineered.csv"
    "features_incoming.csv"        = "${var.data_dir}/features_incoming.csv"
    "features_engineered_test.csv" = "${var.data_dir}/features_engineered_test.csv"
  }
}

resource "aws_security_group" "ec2" {
  name        = "${local.name}-ec2"
  description = "ChurnGuard API + MLflow + Airflow host"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH operateur"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.my_cidr]
  }

  dynamic "ingress" {
    for_each = local.ui_ports
    content {
      description = "${ingress.key} operateur"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = [local.my_cidr]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ───────────────────────── S3 : données processées + artefacts MLflow ─────────────────────────

resource "random_id" "bucket" {
  byte_length = 3
}

resource "aws_s3_bucket" "data" {
  bucket        = "${local.name}-${data.aws_caller_identity.me.account_id}-${random_id.bucket.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

# Données processées (sorties DVC `dvc.yaml`, 39 Mo) : l'instance les télécharge au boot via le rôle d'instance
resource "aws_s3_object" "data" {
  for_each = local.data_files
  bucket   = aws_s3_bucket.data.id
  key      = "data/processed/${each.key}"
  source   = each.value
  etag     = filemd5(each.value)
}

# Contacts SeoLap du dashboard (données utilisateurs, hors git) — montés dans le conteneur dashboard
resource "aws_s3_object" "users_csv" {
  bucket = aws_s3_bucket.data.id
  key    = "demo/users.csv"
  source = var.users_csv_path
  etag   = filemd5(var.users_csv_path)
}

# ───────────────────────── Secrets (générés, jamais dans le dépôt) ─────────────────────────

resource "random_password" "postgres" {
  length  = 24
  special = false
}

resource "random_password" "airflow_admin" {
  length  = 16
  special = false
}

resource "random_password" "airflow_secret_key" {
  length  = 48
  special = false
}

# ───────────────────────── IAM : rôle d'instance (S3 + SSM) ─────────────────────────

resource "aws_iam_role" "ec2" {
  name = "${local.name}-ec2"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "ec2_s3" {
  name = "s3-data-bucket"
  role = aws_iam_role.ec2.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket", "s3:GetBucketLocation"], Resource = aws_s3_bucket.data.arn },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], Resource = "${aws_s3_bucket.data.arn}/*" }
    ]
  })
}

# SSM : permet au job `deploy` de GitHub Actions d'exécuter `git pull && compose up` sans ouvrir SSH
resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${local.name}-ec2"
  role = aws_iam_role.ec2.name
}

# ───────────────────────── IAM : user de déploiement pour GitHub Actions (SSM uniquement) ─────────────────────────

resource "aws_iam_user" "github_deploy" {
  name = "${local.name}-github-deploy"
}

resource "aws_iam_user_policy" "github_deploy" {
  name = "ssm-deploy-churnguard"
  user = aws_iam_user.github_deploy.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ssm:SendCommand"]
        Resource = [
          aws_instance.app.arn,
          "arn:aws:ssm:${var.region}::document/AWS-RunShellScript"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations", "ssm:DescribeInstanceInformation"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_access_key" "github_deploy" {
  user = aws_iam_user.github_deploy.name
}

# ───────────────────────── EC2 : stack docker compose ─────────────────────────

resource "tls_private_key" "ssh" {
  algorithm = "ED25519"
}

resource "aws_key_pair" "ssh" {
  key_name   = "${local.name}-key"
  public_key = tls_private_key.ssh.public_key_openssh
}

resource "local_sensitive_file" "ssh_key" {
  content         = tls_private_key.ssh.private_key_openssh
  filename        = "${path.module}/keys/${local.name}.pem"
  file_permission = "0600"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.ec2_instance_type
  key_name               = aws_key_pair.ssh.key_name
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    encrypted   = true
  }

  # IMDSv2 accessible depuis les conteneurs Docker (2 sauts réseau) : MLflow écrit dans S3 via le rôle d'instance
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    repo_url           = var.repo_url
    repo_ref           = var.repo_ref
    aws_region         = var.region
    s3_bucket          = aws_s3_bucket.data.bucket
    postgres_password  = random_password.postgres.result
    airflow_admin_user = "airflow"
    airflow_admin_pass = random_password.airflow_admin.result
    airflow_secret_key = random_password.airflow_secret_key.result
    alert_webhook_url  = var.alert_webhook_url
  })
  # user_data ne sert qu'au premier boot : les mises à jour passent par scripts/deploy.sh (job CI `deploy`).
  # Pour re-provisionner l'instance de zéro : terraform apply -replace=aws_instance.app
  lifecycle {
    ignore_changes = [user_data]
  }

  tags = { Name = "${local.name}-app" }

  depends_on = [aws_s3_object.data, aws_s3_object.users_csv]
}
