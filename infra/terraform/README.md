# Infrastructure AWS (Terraform)

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars   # alert_webhook_url (optionnel)
cd infra/terraform
terraform init
terraform apply                    # ≈ 2 min, puis ≈ 15–20 min de bootstrap sur l'EC2 (build des 3 images + entraînement baseline)
terraform output                   # api_url, mlflow_url, airflow_url, ssh, bootstrap_log, s3_bucket
terraform output -raw airflow_login
terraform output -json github_secrets   # → secrets GitHub Actions du job `deploy`
terraform destroy                  # après la soutenance (≈ 2,5 $/jour)
```

Voir `docs/deployment-aws.md`.
