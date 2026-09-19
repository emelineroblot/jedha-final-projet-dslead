#!/usr/bin/env bash
# cloud-init : installe Docker, clone le dépôt, télécharge les données depuis S3, démarre la stack
# (docker/prod/), entraîne le modèle baseline (v1 → Production) et recharge l'API.
# Journal : /var/log/churnguard-bootstrap.log
# Fichier rendu par templatefile() : les variables Terraform sont interpolées, le reste est du bash.
set -euxo pipefail
exec > >(tee -a /var/log/churnguard-bootstrap.log) 2>&1
export HOME=/root   # cloud-init ne le définit pas (git config --global échouerait)

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl git gnupg unzip

# ─── AWS CLI v2 (téléchargement des données depuis S3 via le rôle d'instance) ───
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install

# ─── Docker Engine + compose plugin ───
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
usermod -aG docker ubuntu

# ─── Dépôt ───
APP=/opt/churnguard
mkdir -p $APP
git clone --branch "${repo_ref}" --depth 1 "${repo_url}" $APP
git config --global --add safe.directory $APP

# ─── Secrets (uniquement sur l'instance, jamais dans le dépôt) ───
cat > $APP/docker/prod/.env <<EOF
POSTGRES_USER=churnguard
POSTGRES_PASSWORD=${postgres_password}
AIRFLOW_USER=${airflow_admin_user}
AIRFLOW_PASSWORD=${airflow_admin_pass}
AIRFLOW_SECRET_KEY=${airflow_secret_key}
API_WORKERS=2
CHURNGUARD_HOME=$APP
ALERT_WEBHOOK_URL=${alert_webhook_url}
# Artefacts MLflow dans S3 (rôle d'instance, aucune clé). Endpoint régional obligatoire :
# sinon MLflow signe des URLs présignées sur s3.amazonaws.com -> SignatureDoesNotMatch côté client
MLFLOW_ARTIFACTS_DESTINATION=s3://${s3_bucket}/mlflow-artifacts
MLFLOW_S3_ENDPOINT_URL=https://s3.${aws_region}.amazonaws.com
AWS_DEFAULT_REGION=${aws_region}
EOF
chmod 600 $APP/docker/prod/.env

# ─── Données processées depuis S3 (référence, fenêtre incoming, hold-out) ───
mkdir -p $APP/data/processed $APP/reports
aws s3 sync "s3://${s3_bucket}/data/processed" $APP/data/processed

# ─── Permissions pour l'UID airflow (50000) des conteneurs : src/ et data/ sont montés ───
chown -R 50000:0 $APP/src $APP/data $APP/reports
chmod -R g+rwX $APP/src $APP/data $APP/reports

# ─── Stack (images construites sur l'instance depuis le clone — pas de registry privé à authentifier) ───
cd $APP
COMPOSE="docker compose --env-file docker/prod/.env -f docker/prod/docker-compose.yml"
$COMPOSE build
$COMPOSE up -d

# MLflow prêt -> entraînement baseline (référence seule) : 3 modèles comparés, le meilleur enregistré v1.
# Le gate de promotion automatique (F1 hold-out >= 0.70) ne s'applique pas à la mise en service initiale :
# la baseline (F1 ~ 0.69, dérive train/production) est mise en Production explicitement — c'est le point
# de comparaison que le DAG auto_retraining devra battre.
for i in $(seq 1 60); do curl -fs http://localhost:5000/health >/dev/null 2>&1 && break; sleep 5; done
$COMPOSE run --rm --no-deps airflow-scheduler python -m src.training.train
$COMPOSE run --rm --no-deps airflow-scheduler python -m src.training.registry set-production --version 1

# L'API a démarré sans modèle -> rechargement de la version Production
for i in $(seq 1 60); do curl -fs http://localhost:8000/health >/dev/null 2>&1 && break; sleep 5; done
curl -fs -X POST http://localhost:8000/model/reload || true

# Airflow prêt -> batch_scoring actif (un run immédiat alimente la table predictions) ; auto_retraining reste EN PAUSE
# (dépauser un DAG hebdo lance le dernier intervalle manqué — déclenchement manuel pendant la démo)
for i in $(seq 1 60); do curl -fs http://localhost:8080/health >/dev/null 2>&1 && break; sleep 5; done
sleep 30   # laisser le scheduler parser les DAGs
$COMPOSE exec -T airflow-scheduler airflow dags unpause batch_scoring || true

echo "bootstrap terminé : $(date -u)"
