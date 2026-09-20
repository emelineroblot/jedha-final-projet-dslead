#!/usr/bin/env bash
# Déploiement sur l'instance de production (appelé par le job CI `deploy` via SSM, ou à la main en SSH) :
# met à jour le clone, reconstruit les images (cache Docker : rapide si seuls src/ ou les DAGs changent),
# redémarre les services modifiés et recharge le modèle Production dans l'API.
# Usage : bash scripts/deploy.sh [branche|tag]   (défaut : main)
#
# Tout le corps est dans une fonction appelée en dernière ligne : bash lit un script au fil de l'eau, et
# `git reset --hard` réécrit ce fichier pendant son exécution — sans la fonction, la suite du script
# serait lue dans le NOUVEAU fichier à l'ancien offset (commandes tronquées, déploiement cassé — P46).
set -euo pipefail

deploy() {
  local app=${CHURNGUARD_HOME:-/opt/churnguard}
  local ref=${1:-main}
  cd "$app"

  git fetch --depth 1 origin "$ref"
  git reset --hard FETCH_HEAD
  echo "déploiement de $(git rev-parse --short HEAD) ($ref)"

  # src/, data/, reports/ et airflow-logs/ sont montés dans les conteneurs Airflow (UID 50000)
  mkdir -p airflow-logs reports
  chown -R 50000:0 src data reports airflow-logs 2>/dev/null || true
  chmod -R g+rwX src data reports airflow-logs 2>/dev/null || true

  local compose="docker compose --env-file docker/prod/.env -f docker/prod/docker-compose.yml"
  $compose build
  $compose up -d --remove-orphans

  for _ in $(seq 1 30); do curl -fs http://localhost:8000/health >/dev/null 2>&1 && break; sleep 5; done
  curl -fs -X POST http://localhost:8000/model/reload && echo
  echo "déploiement terminé : $(date -u)"
}

deploy "$@"
