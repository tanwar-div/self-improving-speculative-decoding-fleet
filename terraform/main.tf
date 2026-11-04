# Aurora PostgreSQL-Compatible cluster for the draft-model registry.
#
# Serverless v2 with a 0.5 ACU floor, because the registry workload is a few
# small rows per request and a handful of aggregate queries per canary
# evaluation -- it does not need provisioned capacity, and scaling to the floor
# between benchmark runs keeps the cost of standing this up in the low dollars.
#
# The same DDL in sql/001_schema.sql applies here and to the local Postgres
# container: the schema deliberately uses core SQL only, so development does
# not drift away from what Aurora will accept.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "aws" {
  region = var.region
}

# Default VPC rather than a bespoke network: Aurora only needs two subnets in
# different AZs, and a demo cluster does not justify a NAT gateway.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "random_password" "db" {
  length           = 32
  special          = true
  override_special = "!#$%*()-_=+[]{}<>:?"
}

resource "aws_db_subnet_group" "fleet" {
  name       = "${var.name_prefix}-subnets"
  subnet_ids = data.aws_subnets.default.ids
  tags       = var.tags
}

# Ingress is restricted to the operator's own address. The cluster is publicly
# addressable only so that a laptop-based fleet can reach it; nothing else is
# permitted in.
resource "aws_security_group" "fleet" {
  name        = "${var.name_prefix}-db"
  description = "Draft-model registry access from the operator address only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "PostgreSQL from the operator"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_rds_cluster" "fleet" {
  cluster_identifier = "${var.name_prefix}-registry"
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned" # required for Serverless v2
  engine_version     = var.engine_version

  database_name   = var.database_name
  master_username = var.master_username
  master_password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.fleet.name
  vpc_security_group_ids = [aws_security_group.fleet.id]

  # The RDS Data API lets the registry be queried over HTTPS with SigV4 instead
  # of a VPC connection -- useful for Lambda-based controllers, and for reaching
  # the registry from outside the VPC without opening a port.
  enable_http_endpoint = var.enable_data_api

  serverlessv2_scaling_configuration {
    min_capacity = var.min_acu
    max_capacity = var.max_acu
  }

  backup_retention_period = var.backup_retention_days
  storage_encrypted       = true

  # Demo posture: this cluster is meant to be created, measured, and destroyed.
  # Flip both for anything long-lived.
  deletion_protection = var.deletion_protection
  skip_final_snapshot = var.skip_final_snapshot
  final_snapshot_identifier = (
    var.skip_final_snapshot ? null : "${var.name_prefix}-final-${formatdate("YYYYMMDDhhmmss", timestamp())}"
  )

  apply_immediately = true
  tags              = var.tags

  lifecycle {
    ignore_changes = [final_snapshot_identifier]
  }
}

resource "aws_rds_cluster_instance" "writer" {
  identifier          = "${var.name_prefix}-writer"
  cluster_identifier  = aws_rds_cluster.fleet.id
  instance_class      = "db.serverless"
  engine              = aws_rds_cluster.fleet.engine
  engine_version      = aws_rds_cluster.fleet.engine_version
  publicly_accessible = var.publicly_accessible
  apply_immediately   = true
  tags                = var.tags
}

# Optional reader. The canary controller's SLO aggregates and the drift
# detector's window scans are read-only and can be pointed here to keep
# analytical queries off the writer that serving telemetry is inserting into.
resource "aws_rds_cluster_instance" "reader" {
  count               = var.create_reader ? 1 : 0
  identifier          = "${var.name_prefix}-reader"
  cluster_identifier  = aws_rds_cluster.fleet.id
  instance_class      = "db.serverless"
  engine              = aws_rds_cluster.fleet.engine
  engine_version      = aws_rds_cluster.fleet.engine_version
  publicly_accessible = var.publicly_accessible
  apply_immediately   = true
  tags                = var.tags
}

resource "aws_secretsmanager_secret" "db" {
  name                    = "${var.name_prefix}/registry-credentials"
  recovery_window_in_days = 0
  tags                    = var.tags
}

resource "aws_secretsmanager_secret_version" "db" {
  secret_id = aws_secretsmanager_secret.db.id
  secret_string = jsonencode({
    username = var.master_username
    password = random_password.db.result
    host     = aws_rds_cluster.fleet.endpoint
    port     = 5432
    dbname   = var.database_name
    engine   = "postgres"
  })
}
