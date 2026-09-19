# Déploiement en production (AWS)

> Le pipeline complet tourne dans le cloud : infrastructure déclarée en **Terraform**, stack **docker compose**
> (API FastAPI, MLflow, Airflow, PostgreSQL) sur une **EC2**, données et artefacts MLflow dans **S3**,
> déploiement continu depuis GitHub Actions via **SSM**. Ce document décrit ce qui est déployé, comment le
> reproduire, ce qui diffère de la stack de développement, et ce que montre la vidéo.

- Infrastructure : [`infra/terraform/`](../infra/terraform/) (`main.tf`, `variables.tf`, `outputs.tf`, `user_data.sh`)
- Stack : [`docker/prod/docker-compose.yml`](../docker/prod/docker-compose.yml) · mise à jour : [`scripts/deploy.sh`](../scripts/deploy.sh)
- CI/CD : [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (job `deploy`)
- Région : **eu-north-1 (Stockholm)** — résidence des données UE
- Schéma : [`architecture/05-production-aws.svg`](architecture/05-production-aws.svg)

---

## 1. Architecture déployée

```mermaid
flowchart TB
    OP["Poste opérateur — IP unique autorisée<br/>terraform apply · API :8000 · Airflow :8080 · MLflow :5000"]
    GH[("GitHub<br/>jedha-final-projet-dslead")]
    CI["GitHub Actions<br/>test → validate-model → build → deploy"]

    subgraph AWS["AWS eu-north-1 — VPC par défaut — 23 ressources Terraform"]
        subgraph EC2["EC2 m7i-flex.large · EBS 30 Go chiffré · rôle d'instance (S3 + SSM) · docker compose"]
            API["API FastAPI :8000<br/>/predict · /predict/batch · /model/info · /model/reload · /metrics"]
            AF["Airflow 2.9<br/>webserver · scheduler<br/>DAG batch_scoring · DAG auto_retraining"]
            MLF["MLflow 3 server<br/>registry churnguard-model @ Production"]
            PG[("PostgreSQL 15<br/>churnguard (predictions) · mlflow · airflow")]
        end
        S3[("S3 — chiffré, versionné<br/>data/processed/*.csv · mlflow-artifacts/")]
        SSM["SSM Run Command"]
    end

    DISC["Discord<br/>alertes dérive · promotion · échec"]

    OP -- "terraform apply" --> AWS
    GH -- "git clone au boot" --> EC2
    GH -- "push main" --> CI
    CI -- "send-command : scripts/deploy.sh" --> SSM --> EC2
    S3 -- "s3 sync (données)" --> EC2
    MLF -- "artefacts (rôle d'instance)" --> S3
    AF -- "train / evaluate / promote" --> MLF
    MLF -- "load_model Production" --> API
    AF -- "/predict/batch · /model/reload" --> API
    API -- "predictions" --> PG
    AF -- "features récentes (dérive)" --> PG
    MLF -- "backend store" --> PG
    AF -- "webhook" --> DISC
```

| Brique | Développement (`docker/dev/`) | Production (`docker/prod/` + Terraform) |
|---|---|---|
| Images | `build:` local, hot-reload (`--reload`, `src/` monté dans l'API) | **construites sur l'instance** depuis le clone (`compose build`), même tag que les images GHCR de la CI ; API 2 workers uvicorn, `restart: always` |
| PostgreSQL | conteneur, 3 bases | **même conteneur**, volume Docker sur l'EBS chiffré, mot de passe généré par Terraform. **RDS était la cible** : bloqué par le quota du plan gratuit du compte (1 instance, déjà utilisée par un autre projet) |
| Artefacts MLflow | dossier `mlruns/` monté | **S3** `mlflow-artifacts/` (`--artifacts-destination`, rôle d'instance, aucune clé) |
| Données | `data/processed/` local (DVC) | **S3** `data/processed/`, poussées par Terraform (`aws_s3_object`), synchronisées sur l'instance au boot |
| Modèle initial | `python -m src.training.train` sur le poste | conteneur Airflow, une fois au boot : 3 modèles comparés sur la référence, le meilleur enregistré **v1** puis mis en Production explicitement (`registry set-production`) — F1 hold-out **0,690** (LogReg ; XGBoost 0,657) : sous le gate automatique de 0,70, la dérive train/production est réelle — puis `POST /model/reload` |
| DAGs | déclenchement manuel | les deux DAGs activés au boot : `batch_scoring` (un run immédiat alimente `predictions`), `auto_retraining` (premier réentraînement automatique → **v2**). Un DAG en pause n'exécute pas ses runs, même déclenchés à la main |
| Rapports Evidently | `reports/` local | `reports/` du clone en bind mount (`chown 50000`) : `drift_report.{json,html}` + `history/` |
| API | `--reload`, 1 process | `restart: always`, **1 worker** uvicorn (état du modèle en mémoire : `/model/reload` ne toucherait qu'un worker sur N) — montée en charge par réplication du conteneur |
| Secrets | `.env` écrit à la main | **générés par Terraform** (`random_password`), écrits dans `/opt/churnguard/docker/prod/.env` (`chmod 600`) |
| Accès | `localhost` | security group restreint à **l'IP publique de l'opérateur** (22, 8000, 5000, 8080) |
| Mise à jour du code | rebuild manuel | **job CI `deploy`** : SSM → `scripts/deploy.sh` (git reset sur `main`, build, `up -d`, reload API) |

## 2. Ce que fait `user_data.sh` au premier démarrage

1. Installe Docker, compose, AWS CLI v2.
2. Clone le dépôt (`repo_ref`, défaut `main`) dans `/opt/churnguard`.
3. Écrit `docker/prod/.env` avec les secrets Terraform (Postgres, admin Airflow, secret key, webhook Discord, bucket S3).
4. `aws s3 sync` des données processées (référence 440 k lignes, fenêtre incoming, hold-out).
5. `docker compose build && up -d` (prod) — le conteneur Postgres crée les 3 bases au premier démarrage.
6. Quand MLflow répond : `compose run --no-deps airflow-scheduler python -m src.training.train` → LogReg / RF / XGBoost comparés sur la référence (≈ 4 min sur 2 vCPU), le meilleur enregistré **v1**, puis `registry set-production --version 1` (mise en service initiale, hors gate F1 ≥ 0,70).
7. `POST /model/reload` : l'API (démarrée sans modèle, `/predict` en 503) charge v1.
8. Quand Airflow répond : activation de `batch_scoring` (run immédiat : 500 comptes scorés → table `predictions`) et d'`auto_retraining` (run immédiat du dernier intervalle hebdo = premier réentraînement automatique → v2 promue).

Journal : `/var/log/churnguard-bootstrap.log` (commande dans `terraform output bootstrap_log`).

## 3. Reproduire

Prérequis : compte AWS avec droits EC2, S3, IAM ; CLI AWS configurée ; Terraform ≥ 1.5 ; `data/processed/*.csv` présents (`dvc pull` ou `dvc repro`).

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars   # alert_webhook_url (optionnel)
cd infra/terraform
terraform init
terraform apply                         # ≈ 1 min, puis ≈ 12 min de bootstrap sur l'EC2 (build 5 min, entraînement 4 min)
terraform output                        # api_url, airflow_url, mlflow_url, ssh, bootstrap_log, s3_bucket
terraform output -raw airflow_login
terraform output -json github_secrets   # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, EC2_INSTANCE_ID → secrets GitHub
```

**Déploiement continu** : une fois les 4 secrets ajoutés dans GitHub (Settings → Secrets → Actions), chaque push sur `main`
qui passe `test` et `validate-model` exécute `scripts/deploy.sh` sur l'instance via SSM Run Command (`AWS-RunShellScript`).
L'user IAM `churnguard-github-deploy` ne peut que `ssm:SendCommand` sur cette instance — aucun port SSH n'est ouvert aux runners.
Sans secrets, le job est ignoré avec un avertissement. À la main : `terraform output -raw ssh` puis `sudo bash /opt/churnguard/scripts/deploy.sh main`.

**Arrêt** : `terraform destroy` — supprime les 23 ressources, bucket inclus (`force_destroy`).

Pause sans détruire (données conservées sur l'EBS, ~0,1 $/jour) : `aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)` ;
l'IP publique change au redémarrage (`terraform refresh` puis `output`). Si l'IP de l'opérateur change : `terraform apply` met à jour le security group.

## 4. Coût

| Ressource | Tarif eu-north-1 | Par jour |
|---|---|---|
| EC2 `m7i-flex.large` (2 vCPU, 8 Go) | ~0,10 $/h | ~2,4 $ |
| EBS 30 Go gp3 | ~0,09 $/Go/mois | ~0,1 $ |
| S3 (< 100 Mo, versionné) | — | < 0,01 $ |
| SSM Run Command | gratuit | — |
| **Total** | | **≈ 2,5 $/jour** — à détruire après la soutenance |

## 5. Sécurité du déploiement

- Aucune clé AWS dans le code ni sur l'instance : S3 est accédé via le **rôle d'instance** (IMDSv2, `hop_limit = 2` pour que les conteneurs MLflow/Airflow y accèdent).
- Mots de passe Postgres et Airflow, secret key Airflow **générés** par Terraform, stockés dans l'état local (gitignoré) et dans `.env` sur l'instance.
- Clé SSH générée par Terraform → `infra/terraform/keys/` (gitignoré). Clés du user `github-deploy` uniquement dans l'état Terraform et les secrets GitHub.
- Security group : tout est fermé sauf l'IP de l'opérateur ; les runners GitHub passent par SSM (canal sortant de l'agent), pas par SSH.
- Chiffrement at-rest : volume EC2 (Postgres inclus), S3 (AES-256) ; versioning S3 ; bucket privé.
- Limites assumées : UIs et API en HTTP (pas de TLS, pas de domaine) ; base non managée (pas de sauvegarde automatique — snapshot EBS ou RDS en cible) — acceptable pour une démo restreinte à une IP, pas pour une vraie production (reverse proxy TLS, RDS privé, Secrets Manager, auth sur l'API).

## 6. Pièges rencontrés

- **MLflow 3 + S3** : le serveur renvoie des URLs présignées au client pour le téléchargement des artefacts → `MLFLOW_S3_ENDPOINT_URL=https://s3.<région>.amazonaws.com` obligatoire (sinon `SignatureDoesNotMatch` côté API/Airflow alors que l'upload passe).
- **MLflow 3 + Host** : `--allowed-hosts *` sinon `403 Invalid Host header` depuis `http://mlflow:5000` (P37).
- **Airflow UID 50000** : `src/` et `data/` montés depuis le clone doivent être `chown 50000:0` (bootstrap et `deploy.sh`).
- **`compose run --no-deps`** pour l'entraînement baseline : le scheduler dépend de l'API « healthy », l'API démarre sans modèle → l'entraînement ne doit pas attendre la chaîne de dépendances.
- **Gate de promotion** : le meilleur modèle baseline (F1 hold-out 0,690) est sous le seuil automatique de 0,70 → `train` l'enregistre sans le promouvoir ; la mise en service initiale est explicite (`registry set-production`).
- **DAG en pause = runs jamais exécutés** (même `airflow dags trigger` reste `queued`). Dépauser un DAG hebdo lance immédiatement son dernier intervalle manqué (`catchup=False` n'empêche pas ce run unique) → les deux DAGs sont activés au bootstrap ; `max_active_runs=1` sérialise run planifié et run manuel.
- **Volume nommé `reports`** : créé root car le dossier n'existe pas dans l'image Airflow → `PermissionError` dans `check_drift`. Bind mount du clone (`chown 50000`).
- **`runs:/<id>/model` = 8 min** de chargement avec des artefacts S3 (résolution des logged models MLflow 3) contre < 1 s via `models:/churnguard-model/N` → le DAG évalue le candidat par sa version de registry.
- **Workers uvicorn** : l'état du modèle est en mémoire du process → `--workers 1` par conteneur, sinon `/model/reload` ne recharge qu'un worker et l'API sert deux versions.
- **`user_data` = premier boot uniquement** (`lifecycle.ignore_changes`) : les mises à jour passent par `deploy.sh` ; pour repartir de zéro, `terraform apply -replace=aws_instance.app`.

## 7. Validation (2026-09-19)

Run manuel d'`auto_retraining` sur l'EC2 après `registry rollback --version 1` : `check_drift` 6 s (dérive détectée, source `predictions` en base) →
`retrain_model` 32 s (3 modèles, 100 k lignes de référence + fenêtre récente) → `evaluate_model` 3 s (candidat 0,978 vs Production 0,690) →
`promote_model` 3 s (v3 Production, `/model/reload`, smoke test, alerte Discord). **Boucle complète : 47 s.** Déploiement continu testé par SSM
avec les clés du user `github-deploy` : `Success` (rebuild depuis le cache, redémarrage, reload).

## 8. Ce que la vidéo montre (script, ≈ 3 min)

Préparation (hors vidéo) : `registry rollback --version 1` + `POST /model/reload` pour repartir de la baseline.

1. Console AWS ou `terraform output` : l'instance `churnguard-app`, le bucket S3 (`data/processed/`, `mlflow-artifacts/`).
2. API (`http://<ip>:8000/docs`) : `GET /model/info` → **v1**, F1 hold-out 0,690 ; `POST /predict` sur un compte à risque.
3. MLflow (`http://<ip>:5000`) : l'expérience `churnguard` (runs baseline + réentraînements), le registry `churnguard-model`, les artefacts dans S3.
4. Airflow (`http://<ip>:8080`) : `batch_scoring` vert (500 comptes), puis **▶ trigger `auto_retraining`** → `check_drift` (9/15 features en dérive) → `retrain_model` → `evaluate_model` → `promote_model` (< 1 min).
5. Discord : « Réentraînement déclenché » puis « Nouveau modèle en Production — v1 → vN (F1 0,690 → 0,978) ».
6. Retour API : `GET /model/info` → **vN** sans redémarrage ; MLflow : v1 archivée ; rollback en une commande si besoin.
7. GitHub Actions : le run du dernier push sur `main` avec le job `deploy` vert (SSM).
