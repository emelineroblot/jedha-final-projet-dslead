variable "region" {
  description = "Région AWS (Stockholm : résidence des données UE)"
  type        = string
  default     = "eu-north-1"
}

variable "repo_url" {
  description = "Dépôt public cloné par l'instance au démarrage"
  type        = string
  default     = "https://github.com/emelineroblot/jedha-final-projet-dslead.git"
}

variable "repo_ref" {
  description = "Branche ou tag déployé"
  type        = string
  default     = "main"
}

variable "ec2_instance_type" {
  description = "API + MLflow + Airflow (2 processus) + PostgreSQL + entraînement XGBoost : 8 Go de RAM"
  type        = string
  default     = "m7i-flex.large"
}

variable "operator_cidr" {
  description = "CIDR autorisé (SSH, API, UIs). 0.0.0.0/0 = ouvert (démo jury : SSH par clé, Airflow par mot de passe) ; vide = IP publique courante"
  type        = string
  default     = "0.0.0.0/0"
}

variable "data_dir" {
  description = "Dossier local des CSV processés (sorties `dvc repro`), poussés dans S3 — relatif au dossier terraform"
  type        = string
  default     = "../../data/processed"
}

variable "users_csv_path" {
  description = "Export des utilisateurs SeoLap pour le dashboard Streamlit (hors git), poussé dans S3 — relatif au dossier terraform"
  type        = string
  default     = "../../hf-demo/users.csv"
}

variable "alert_webhook_url" {
  description = "Webhook Discord/Slack pour les alertes (dérive, promotion, échec du DAG). Vide = alertes loguées seulement"
  type        = string
  sensitive   = true
  default     = ""
}
