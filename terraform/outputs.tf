output "writer_endpoint" {
  description = "Aurora writer endpoint"
  value       = aws_rds_cluster.fleet.endpoint
}

output "reader_endpoint" {
  description = "Aurora reader endpoint (serves the writer when no reader instance exists)"
  value       = aws_rds_cluster.fleet.reader_endpoint
}

output "cluster_arn" {
  description = "Cluster ARN, needed when talking to the RDS Data API"
  value       = aws_rds_cluster.fleet.arn
}

output "secret_arn" {
  description = "Secrets Manager ARN holding the registry credentials"
  value       = aws_secretsmanager_secret.db.arn
}

output "database_url_command" {
  description = "Prints the FLEET_DATABASE_URL to export, reading the password from Secrets Manager rather than from Terraform output"
  value = join(" ", [
    "aws secretsmanager get-secret-value",
    "--secret-id ${aws_secretsmanager_secret.db.name}",
    "--region ${var.region}",
    "--query SecretString --output text |",
    "python3 -c \"import json,sys;d=json.load(sys.stdin);print(f'postgresql://{d[\\\"username\\\"]}:{d[\\\"password\\\"]}@{d[\\\"host\\\"]}:5432/{d[\\\"dbname\\\"]}')\""
  ])
}
