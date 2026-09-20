output "api_url" {
  value = "http://${aws_instance.app.public_ip}:8000"
}

output "api_docs_url" {
  value = "http://${aws_instance.app.public_ip}:8000/docs"
}

output "dashboard_url" {
  value = "http://${aws_instance.app.public_ip}:8501"
}

output "mlflow_url" {
  value = "http://${aws_instance.app.public_ip}:5000"
}

output "airflow_url" {
  value = "http://${aws_instance.app.public_ip}:8080"
}

output "airflow_login" {
  value     = "airflow / ${random_password.airflow_admin.result}"
  sensitive = true
}

output "ssh" {
  value = "ssh -i infra/terraform/keys/churnguard.pem ubuntu@${aws_instance.app.public_ip}"
}

output "bootstrap_log" {
  value = "ssh -i infra/terraform/keys/churnguard.pem ubuntu@${aws_instance.app.public_ip} 'sudo tail -f /var/log/churnguard-bootstrap.log'"
}

output "s3_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "instance_id" {
  value = aws_instance.app.id
}

# Secrets GitHub Actions (Settings → Secrets → Actions) pour le job `deploy` : terraform output -json github_secrets
output "github_secrets" {
  value = {
    AWS_ACCESS_KEY_ID     = aws_iam_access_key.github_deploy.id
    AWS_SECRET_ACCESS_KEY = aws_iam_access_key.github_deploy.secret
    AWS_REGION            = var.region
    EC2_INSTANCE_ID       = aws_instance.app.id
  }
  sensitive = true
}
