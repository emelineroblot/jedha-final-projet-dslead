#!/usr/bin/env bash
# Déploiement sur l'instance de production (appelé par le job CI `deploy` via SSM, ou à la main en SSH) :
# met à jour le clone, reconstruit les images (cache Docker : rapide si seuls src/ ou les DAGs changent),
# redémarre les services modifiés et recharge le modèle Production dans l'API.
# Usage : bash scripts/deploy.sh [branche|tag]   (défaut : main)
set -euo pipefail
APP=${CHURNGUARD_HOME:-/opt/churnguard}
REF=${1:-main}
cd "$APP"

git fetch --depth 1 origin "$REF"
git reset --hard FETCH_HEAD
echo "déploiement de $(git rev-parse --short HEAD) ($REF)"

# src/ et data/ sont montés dans les conteneurs Airflow (UID 50000)
chown -R 50000:0 src data reports 2>/dev/null || true
chmod -R g+rwX src data reports 2>/dev/null || true

COMPOSE="docker compose --env-file docker/prod/.env -f docker/prod/docker-compose.yml"
$COMPOSE build
$COMPOSE up -d --remove-orphans

for i in $(seq 1 30); do curl -fs http://localhost:8000/health >/dev/null 2>&1 && break; sleep 5; done
curl -fs -X POST http://localhost:8000/model/reload && echo
echo "déploiement terminé : $(date -u)"
