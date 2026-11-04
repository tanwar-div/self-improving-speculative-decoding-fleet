variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
  default     = "fleet"
}

variable "engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "16.4"
}

variable "database_name" {
  type    = string
  default = "fleet"
}

variable "master_username" {
  type    = string
  default = "fleet"
}

variable "allowed_cidrs" {
  description = "CIDRs allowed to reach the registry on 5432. Set this to your own address; do not widen it."
  type        = list(string)

  validation {
    condition     = !contains(var.allowed_cidrs, "0.0.0.0/0")
    error_message = "Refusing to expose the registry to 0.0.0.0/0; pass your own address, e.g. [\"203.0.113.7/32\"]."
  }
}

variable "min_acu" {
  description = "Serverless v2 minimum capacity. 0.5 ACU is the floor."
  type        = number
  default     = 0.5
}

variable "max_acu" {
  type    = number
  default = 2
}

variable "enable_data_api" {
  description = "Expose the RDS Data API (HTTPS + SigV4) alongside the Postgres endpoint"
  type        = bool
  default     = true
}

variable "publicly_accessible" {
  description = "Needed to reach the cluster from a laptop-hosted fleet"
  type        = bool
  default     = true
}

variable "create_reader" {
  description = "Add a reader instance for SLO/drift analytical queries"
  type        = bool
  default     = false
}

variable "backup_retention_days" {
  type    = number
  default = 1
}

variable "deletion_protection" {
  type    = bool
  default = false
}

variable "skip_final_snapshot" {
  type    = bool
  default = true
}

variable "tags" {
  type = map(string)
  default = {
    Project = "self-improving-fleet"
    Purpose = "draft-model-registry"
  }
}
