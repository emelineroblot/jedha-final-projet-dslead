# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

**ChurnGuard** — Pipeline MLOps end-to-end de prédiction de churn pour SeoLap (projet final certification Jedha Data Science Lead).

Entraînement sur le dataset **muhammadshahidazeem** (Kaggle, 440k lignes, subscription service générique), réentraînable sur les données réelles SeoLap. Les 6 composantes MLOps sont toutes obligatoires : preprocessing → modèle → API → CI/CD → monitoring → réentraînement automatisé.

**Note** : le dataset rivalytics (500 comptes fictifs) a été abandonné en Phase 3 — corrélations < 0.09, F1 plafonné à 0.52. Voir `contexte/problematiques-rencontrees.md` P1.

## Structure cible du repo

```
final-project-dslead/
├── data/                        # Données brutes muhammadshahidazeem + rivalytics (archivé)
│   └── processed/               # features_engineered.csv (train) + features_engineered_test.csv
├── notebooks/                   # EDA + expérimentations
├── src/
│   ├── preprocessing/           # Pipeline flat dataset : loader → cleaner → features
│   ├── training/                # Entraînement Scikit-learn/XGBoost + évaluation
│   ├── api/                     # FastAPI app (POST /predict, POST /predict/batch, GET /model/info)
│   ├── monitoring/              # Rapports Evidently (data drift + model performance)
│   └── retraining/              # DAGs Airflow
├── tests/
├── docker/
│   ├── dev/docker-compose.yml
│   └── prod/docker-compose.yml
├── .github/workflows/           # CI/CD GitHub Actions
└── mlruns/                      # MLflow (gitignored — non versionné par DVC)
```

## Roadmap (11 phases)

| Phase | Intitulé | Semaine | Statut |
|---|---|---|---|
| 0 | Setup repo & environnement | S1 | ✓ |
| 1 | EDA & Preprocessing | S1 | ✓ |
| 2 | Feature Engineering | S1 | ✓ |
| 3 | Entraînement & sélection du modèle | S1–S2 | ✓ |
| 4 | MLflow + DVC — Versioning | S2 | ✓ |
| 5 | API FastAPI | S2 | ✓ |
| 6 | Containerisation Docker | S2 | ✓ |
| 7 | Pipeline CI/CD GitHub Actions | S2–S3 | ✓ |
| 8 | Orchestration Airflow | S3 | ✓ |
| 9 | Monitoring Evidently | S3–S4 | ✓ |
| 10 | Documentation & diagramme | S4 | ✓ |
| 11 | Production AWS (Terraform) — vidéo du pipeline en prod | S5 | ✓ (2026-09-19) |
| 12 | Présentation jury | S5 | ⏳ (semaine du 21/09/2026) |

**Workflow git** : `main` + `feature/*` uniquement (branche `develop` supprimée le 2026-09-15 — phases 0–10 mergées dans `main`). Une feature = une branche `feature/nom-court`, mergée dans `main` une fois validée.

## Phase 10 — Audit & durcissement (2026-09-15) ✓

Audit complet vs énoncé Jedha dans `docs/audit.md` (non versionné). Corrections livrées sur `feature/audit-fixes`.

**Découvertes majeures (à connaître pour la soutenance)** :
- **Le fichier test Kaggle ne suit pas la distribution du train** : le modèle "F1 0.999" (validation sur split du train) fait **F1 0.66 / AUC 0.73** sur le fichier test (prédit presque tout en churn). Décision : le test = production qui a dérivé → scindé en `features_incoming.csv` (nouvelles données labellisées, 32k) et `features_engineered_test.csv` (hold-out, 32k). Evidently détecte 9/15 features en dérive **naturellement** (simulate_drift devient optionnel).
- **Bug Evidently (P23)** : `drift_share` dans `DatasetDriftMetric` = le SEUIL paramétré (0.5), pas la part observée → l'ancien `check_drift` détectait toujours une dérive. Utiliser `share_of_drifted_columns`.
- **Bug rapport figé (P24)** : `check_drift` ne régénérait le rapport que s'il était absent → verdict figé. Désormais toujours régénéré (+ copie horodatée `reports/history/`).
- **Réentraînement sur les mêmes données (P25)** : `train()` relisait toujours le même CSV → candidat = prod, jamais promu. Désormais `train(extra_data_path=…, reference_rows=100k)` : validation/seuil calibrés sur la fenêtre récente. Résultat hold-out : **0.657 → 0.978** (XGBoost), RF 0.973, LogReg 0.753.
- **DAGs non exécutables dans Docker (P26)** : image Airflow stock sans deps ML ni `src/`. → `Dockerfile.airflow` (`apache/airflow:2.9.1-python3.11` + `requirements-airflow.txt`), mounts `src/ data/ reports/ mlruns/`, `CHURNGUARD_ROOT=/opt/airflow`, `MLFLOW_TRACKING_URI` + `CHURNGUARD_API_URL` dans l'env Airflow. **Stack Docker validée en live le 2026-09-16** (Avast en pause) : DAGs importés sans erreur, `check_drift` dans le conteneur = 60 % de features en dérive, DAG `auto_retraining` vert de bout en bout (candidat F1 0.9777 vs prod 0.6566 → v5 promue → `/model/reload` → smoke test OK → alerte), rollback v5→v1→v5 en 6 s, `batch_scoring` OK (500 comptes, 399 à risque élevé), 503 prédictions en base et **contrôle de dérive alimenté par la table `predictions`**, latence p95 59 ms / 1 000 comptes en 110 ms.

**Nouveautés** : `src/paths.py` (chemins via `CHURNGUARD_ROOT`) · `FEATURE_COLUMNS` canonique (`features.py`, ordre = modèle) · API : `Features` Pydantic typée (422), batch vectorisé, `POST /model/reload`, `/metrics` Prometheus, latence loguée, stockage des prédictions (`store.py`, `DATABASE_URL`) · `notify.py` (webhook `ALERT_WEBHOOK_URL`) · `registry.py` CLI (`list/promote/rollback`) · `tuner.py` (RandomizedSearchCV → `best_params.json`) · `validate.py` (gate CI F1 ≥ 0.75) · `evaluate.py` (métriques + figures + importance) · lineage MLflow (`git_sha`, `data_dvc_md5` via dvc.lock) · `dvc.yaml` (preprocess → train) · `model_artifacts.dvc` · DAG retraining : trigger dérive OU ≥ 5000 nouvelles lignes, rollback réel + smoke test + reload API · CI : triggers `main`, `validate-model`, build 3 images, `deploy-model.yml` (HF Space) · docs : `docs/architecture/*.svg` (5 schémas), `api.md`, `dataset-report.md`, `model-card.md` · 36 tests.

**Pitfalls** :
- **P27** : `validate.py` doit utiliser le `decision_threshold` du modèle (0.89 après retrain), pas 0.5 — sinon F1 0.72 au lieu de 0.97 sur la fixture.
- **P28** : `apache/airflow:2.9.1` (tag par défaut) est en Python 3.8 → mlflow 3 / xgboost 3 / numpy 2 non installables. Utiliser `2.9.1-python3.11`.
- **P29** : Python 3.13 local vs 3.11 Docker — `numpy>=2.4` exige ≥ 3.11, OK. `test_dags.py` skippé sur Windows (Airflow 3 importe `fcntl`).
- **P30** : le hook rtk casse les heredocs bash contenant du code Python multi-blocs → utiliser l'outil Write / un script Python pour les patchs.
- **P31** : `docs/` versionné sauf `docs/soutenance*` et `docs/audit.md` (gitignore). `.env.example` autorisé malgré `.env.*`.
- **P33 (machine Emeline)** : **Avast intercepte le TLS** → `pip` dans les conteneurs échoue (`CERTIFICATE_VERIFY_FAILED` sur pypi.org). Les Dockerfiles acceptent `--build-arg PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"` (vide par défaut = CI normale). En local : `docker compose -f docker/dev/docker-compose.yml -f docker/dev/docker-compose.override.yml build` (override gitignoré, modèle dans `.override.yml.example`). Compose ne charge PAS l'override automatiquement quand `-f` est utilisé.
- **P35** : `mlflow-skinny 3.12.0` exige `starlette<1` → `fastapi==0.116.1` + `prometheus-fastapi-instrumentator==7.1.0` (pas 0.136 / 8.1). Le venv local était incohérent (`pip check`) mais fonctionnait par chance ; l'image Docker, elle, refusait de se résoudre.
- **P36** : avec Avast, Docker Hub lui-même est inaccessible par moments (`x509: certificate` / `DeadlineExceeded` sur `registry-1.docker.io`) et le débit conteneur tombe à ~370 kB/s. Validation Docker faite le 2026-09-16 avec Avast en pause (débit 8 Mo/s au lieu de 370 kB/s). Les Dockerfiles ont un cache pip partagé (`id=churnguard-pip`) + `wheels/` (bind mount, vide en CI) + `PIP_DEFAULT_TIMEOUT/RETRIES`.
- **P34** : d'autres projets (`fraud-detection-*`, `food_impact_db`) occupent les ports 8080/5000/5432 → `docker stop` avant de lancer la stack ChurnGuard.
- **P37** : **MLflow 3.x rejette tout Host ≠ localhost** (`403 Invalid Host header - possible DNS rebinding attack`) → depuis les conteneurs (`http://mlflow:5000`) l'API et Airflow ne peuvent pas joindre le registry. Obligatoire : `--allowed-hosts * --cors-allowed-origins *` dans la commande `mlflow server` (compose dev + prod).
- **P38** : dépauser un DAG `schedule` hebdo lance immédiatement le dernier intervalle manqué (même avec `catchup=False`) — mais un DAG en pause n'exécute **aucun** run, même déclenché à la main → voir P38 corrigé (Phase 11).
- **P32** : `mlflow server` doit tourner avec `--serve-artifacts --artifacts-destination /mlruns` pour que les clients (host, API, Airflow) n'aient pas besoin du chemin `/mlruns` local.

**Démo (validée)** : DAG `auto_retraining` **dépausé** (sinon le run reste `queued`, P38 corrigé) puis `airflow dags trigger auto_retraining` ; `max_active_runs=1`. `promote_model` réutilise la version déjà enregistrée par `train(auto_promote=False)` (plus de doublon v2→v4).

**Reste à faire (côté Emeline)** : (1) vidéo de la démo sur la stack AWS (script dans `docs/deployment-aws.md` §7) ; (2) pousser les Spaces HF (`hf-space/`, `hf-airflow/`, `hf-mlflow/` mis à jour localement) ; (3) secrets GitHub `DAGSHUB_USER/TOKEN`, `HF_TOKEN` (les 4 secrets AWS du job `deploy` sont posés depuis le 2026-09-19) ; (4) intégrer les schémas dans les slides ; (5) `terraform destroy` après la soutenance.

## Phase 11 — Production AWS (2026-09-19) ✓

Livrable Jedha « vidéo de la solution fonctionnant en production » → le pipeline complet tourne sur AWS. Même pattern que
`automatic-fraud-detection` (validé le même jour). Doc publique : `docs/deployment-aws.md` ; schéma `docs/architecture/05-production-aws.svg`.

- **`infra/terraform/`** (24 ressources, eu-north-1) : EC2 `churnguard-app` m7i-flex.large Ubuntu 24.04 (EBS 30 Go chiffré), S3 `churnguard-<acct>-<rand>` (`data/processed/*.csv` poussés par Terraform + `mlflow-artifacts/`), rôle d'instance S3 + `AmazonSSMManagedInstanceCore`, user IAM `churnguard-github-deploy` (`ssm:SendCommand` sur cette seule instance), SG **ouvert** 22/8000/8501/5000/8080 (`operator_cidr = 0.0.0.0/0` depuis le 2026-09-20, décision Emeline : le jury doit pouvoir ouvrir les URLs ; `""` = IP détectée), secrets `random_password`, clé SSH → `infra/terraform/keys/` (gitignoré avec tfstate/tfvars).
- **Terraform natif** (1.9.8 installé, Avast désactivé) : `cd infra/terraform && terraform apply` ≈ 1 min, bootstrap EC2 ≈ 15–20 min. `terraform.tfvars` = webhook Discord (réutilisé de fraud-detection).
- **Bootstrap `user_data.sh`** (≈ 12 min) : Docker → clone (`repo_ref`) → `docker/prod/.env` généré → `aws s3 sync` → `compose build && up -d` → `compose run --no-deps airflow-scheduler python -m src.training.train` (v1 = LogReg, F1 hold-out 0,690 < gate 0,70 → enregistrée sans promotion) → `registry set-production --version 1` (mise en service initiale) → `POST /model/reload` → `dags unpause batch_scoring` + `auto_retraining` (run immédiat du dernier intervalle = premier réentraînement auto → v2). Log `/var/log/churnguard-bootstrap.log`. Code dans `/opt/churnguard`.
- **`docker/prod/docker-compose.yml`** : `build:` sur les 3 services (images construites sur l'instance, taguées comme GHCR → `compose pull` reste possible), artefacts MLflow `MLFLOW_ARTIFACTS_DESTINATION` (S3 en prod, `/mlruns` sinon), `MLFLOW_S3_ENDPOINT_URL` + `AWS_DEFAULT_REGION` propagés à mlflow/api/airflow. `Dockerfile.mlflow` + `boto3`. **Dashboard Streamlit en prod (2026-09-20)** : service `dashboard` :8501 (`Dockerfile.dashboard`, `demo/`), `CHURNGUARD_API_URL=http://api:8000` (défaut = Space HF pour `hf-demo/`), `demo/users.csv` (contacts SeoLap, gitignoré) poussé dans S3 par Terraform (`aws_s3_object.users_csv`, source `hf-demo/users.csv`) et monté `:ro` ; 4ᵉ image dans la matrice CI. Les features de `demo/data.py` ont été calibrées sur l'ancien XGBoost → avec le modèle réentraîné 42/50 contacts sortent « élevé » (cosmétique).
- **Validation live (2026-09-19)** : `auto_retraining` manuel après `rollback --version 1` → **47 s** (check_drift 6 s, retrain 32 s, evaluate 3 s, promote+reload+smoke 3 s), v3 Production, alerte Discord ; chemin SSM testé avec les clés `github-deploy` (`Success`).
- **CD** : job `deploy` du CI (`needs: [test, validate-model]`, push `main`) → `aws ssm send-command` → `scripts/deploy.sh main` sur l'instance (git reset, build, `up -d`, reload API). Secrets GitHub : `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `EC2_INSTANCE_ID` (`terraform output -json github_secrets`). Sans secrets → warning, job ignoré. **P45** : le job `build` GHCR échouait en 10 s (`buildx failed … build-cache-backends`) — `build-push-action@v5` + cache `type=gha` sur l'ancien service de cache Actions (arrêté en 2025) → `setup-buildx-action@v3` + `build-push-action@v6`. CI entièrement verte le 2026-09-20 (test, validate-model, 3 builds, deploy).
- **Coût ≈ 2,5 $/jour** (3ᵉ EC2 du compte avec `stripe-pipeline-*` et `fraud-detection-*`, à ne jamais toucher) → **`terraform destroy` après la soutenance**. Pause : `aws ec2 stop-instances` (IP publique change → `terraform refresh`).

**Pitfalls** :
- **P39** : sous cloud-init `$HOME` n'est pas défini → `git config --global` échoue et `set -e` tue le bootstrap. `export HOME=/root` en tête de `user_data.sh`.
- **P40** : `user_data` est en `lifecycle.ignore_changes` (bootstrap = premier boot uniquement, les mises à jour passent par `deploy.sh`). Pour re-provisionner : `terraform apply -replace=aws_instance.app` (nouvelle IP publique).
- **P41** : l'API démarre sans modèle (`/health` 200, `/predict` 503) et le scheduler dépend de l'API « healthy » → l'entraînement baseline se fait avec `compose run --rm --no-deps airflow-scheduler …`, puis `POST /model/reload`.
- **P42** : `src/`, `data/` et `reports/` montés dans Airflow (UID 50000) → `chown -R 50000:0` au bootstrap et dans `deploy.sh`. **Pas de volume nommé pour `reports`** : créé root (dossier absent de l'image) → `PermissionError` dans `check_drift`.
- **P43** : `--workers N` uvicorn + état modèle en mémoire → `/model/reload` ne recharge qu'un worker (l'API servait v1 et 503 en alternance). **1 worker par conteneur** en prod ; scaler par réplication.
- **P44** : avec artefacts S3, `load_model("runs:/<id>/model")` prend **494 s** (résolution logged models MLflow 3) contre 0,9 s via `models:/churnguard-model/N` → `evaluate_model` charge le candidat par sa version (`registry.version_for_run`).
- **P38 (corrigé)** : un DAG **en pause n'exécute jamais ses runs**, même `airflow dags trigger` (reste `queued`). Le dépauser lance une fois son dernier intervalle manqué (`catchup=False` n'empêche pas ce run unique). Prod : les deux DAGs activés au bootstrap, `max_active_runs=1` sérialise planifié + manuel.

---

## Phase 7 — CI/CD GitHub Actions ✓

**Fichier** : `.github/workflows/ci.yml`

**Jobs** (mis à jour Phase 10) :
- `test` : `ruff check src/ tests/` + `pytest tests/` (hors `test_dags.py`) — install légère (`requirements-api.txt` + evidently + pytest/ruff), pas d'airflow
- `validate-model` : `dvc pull model_artifacts.dvc` + `validate.py` F1 ≥ 0.75 sur `tests/fixtures/sample_test.csv` — ignoré avec warning sans secrets DagsHub
- `build` : matrice 3 images (`churnguard-api`, `churnguard-mlflow`, `churnguard-airflow`) → GHCR `ghcr.io/emelineroblot/<image>` — tags `main` + `main-<sha>`
- `deploy` : désactivé (`if: false`) — démo déployée sur HF Spaces via `deploy-model.yml`

**Triggers** : push / PR sur `main`, `workflow_dispatch`

**Pitfall P14** (résolu Phase 10) : `test_preprocessing.py` réécrit pour le dataset muhammadshahidazeem, réintégré au CI.

**Pitfall P15** : `pip install -e ".[dev]"` installe apache-airflow → build CI > 5 min. Toujours utiliser `requirements-api.txt` pour les jobs de test.

---

## Phase 9 — Monitoring Evidently ✓

**Fichiers** :
- `src/monitoring/drift_report.py` — `generate_drift_report(reference, current, target_column=None, prediction_column=None)` : DataDriftPreset obligatoire, ClassificationPreset conditionnel si prediction_column fourni
- `src/monitoring/alert.py` — `run_drift_check()` → `DriftResult` ; `check_drift()` (bool, régénère toujours le rapport). Données courantes : chemin explicite → table `predictions` → `features_drifted.csv` → `features_incoming.csv`. Alerte si `share_of_drifted_columns > 0.2` ou F1 drop `> 0.05`

**Déclenchement de la démo** :
```bash
# 1. Injecter une dérive artificielle
python src/retraining/scripts/simulate_drift.py --noise 0.3

# 2. Vérifier que la dérive est détectée
python -c "from src.monitoring.alert import check_drift; print(check_drift())"
```

**Pitfall P17** : `as_dict()` d'Evidently encode le nom de la metric sous la clé `"metric"` (ex: `"DatasetDriftMetric"`). Pour parser, tester `"DatasetDrift" in metric_name` (substring) plutôt que l'égalité stricte — le nom exact peut varier selon la version d'Evidently. **Et lire `share_of_drifted_columns`, pas `drift_share` (= seuil) — voir P23.**

---

## Phase 8 — Orchestration Airflow ✓

**DAGs** :
- `src/retraining/dags/batch_scoring_dag.py` — schedule `0 2 * * *` : charge le CSV test (500 premières lignes en démo), appelle `/predict/batch` par chunks de 100, simule l'envoi Mautic
- `src/retraining/dags/retraining_dag.py` — schedule `0 3 * * 1` (lundi 3h) : `check_drift` (dérive OU nouvelles lignes) → `retrain_model` (`auto_promote=False`, `extra_data_path`) → `evaluate_model` (F1 hold-out, seuil propre à chaque modèle) → `decide` → `promote_model` (reload API + smoke test + rollback si échec) / `keep_current`

**Variables d'environnement Airflow** :
- `CHURNGUARD_API_URL` : URL de l'API (défaut `http://api:8000` dans le conteneur Docker)
- `MLFLOW_TRACKING_URI` : `http://mlflow:5000` — défini dans `x-airflow-common` du compose dev (Phase 10 ; absent avant)
- `DATABASE_URL`, `ALERT_WEBHOOK_URL`, `CHURNGUARD_ROOT=/opt/airflow`, `RETRAIN_REFERENCE_ROWS`, `NEW_DATA_MIN_ROWS`

**Pitfall P18** : `train(auto_promote=False)` enregistre le modèle dans le Registry sans le transitionner en Production — indispensable pour que `evaluate_model` puisse comparer candidat vs Production courante avant de décider.

**Pitfall P19** : XCom Airflow est limité à ~48 KB. Ne jamais passer de DataFrames par XCom — toujours passer des chemins de fichiers ou des identifiants (run_id, version).

**Pitfall P20** : MLflow client 3.x appelle `/api/2.0/mlflow/logged-models` (endpoint inexistant en 2.x). `Dockerfile.mlflow` doit utiliser la même version majeure que le client local. Vérifier avec `python -c "import mlflow; print(mlflow.__version__)"` et aligner le Dockerfile en conséquence. Actuellement : client = serveur = **3.12.0** (`python:3.11-slim` + `pip install mlflow==3.12.0`).

**Pitfall P21** : Entraîner avec `MLFLOW_TRACKING_URI=http://localhost:5000` via PowerShell (`$env:MLFLOW_TRACKING_URI = "..."`) — ne pas utiliser le Bash tool pour les variables d'env sur Windows (syntaxe ignorée silencieusement). Sans cette variable, MLflow écrit en local (`mlruns/`) et le serveur Docker reste vide.

**Pitfall P22** : `ColumnMapping` d'Evidently a été déplacé dans `evidently.legacy.pipeline.column_mapping` en version 0.7.x. Les imports `from evidently import ColumnMapping` et `from evidently.metric_preset import DataDriftPreset` sont cassés — utiliser `from evidently.legacy.pipeline.column_mapping import ColumnMapping` et `from evidently.legacy.metric_preset import DataDriftPreset`.

---

## Phase 6 — Containerisation Docker ✓

**Fichiers** :
- `Dockerfile` — Python 3.11-slim, non-root, HEALTHCHECK, installe depuis `requirements-api.txt` (pinné)
- `Dockerfile.mlflow` — `python:3.11-slim` + `mlflow==3.12.0` + `psycopg2-binary`
- `Dockerfile.airflow` — `apache/airflow:2.9.1-python3.11` + `requirements-airflow.txt` (mlflow, xgboost, sklearn, evidently)
- `.dockerignore` — exclut `data/`, `mlruns/`, `.venv313/`, `notebooks/` du contexte de build
- `requirements-api.txt` — dépendances API pinnées (pandas, sklearn, xgboost, mlflow 3.12.0, fastapi, sqlalchemy, psycopg2, prometheus) — pas airflow, pas dvc
- `docker/dev/init-db.sql` — crée les bases `mlflow` et `airflow` au premier démarrage PostgreSQL

**Pitfalls** : voir P10–P13 dans `contexte/problematiques-rencontrees.md`.

**Ports dev** (attention : SeoLap tourne déjà sur 8000) :
- API → `localhost:8001` (conteneur écoute 8000, mappage 8001:8000)
- MLflow → `localhost:5000`
- Airflow → `localhost:8080`

**Commandes** :
```bash
# Démarrer la stack dev
docker compose -f docker/dev/docker-compose.yml up -d

# Arrêter et supprimer les volumes (reset complet)
docker compose -f docker/dev/docker-compose.yml down -v

# Rebuild uniquement l'API après modif src/
docker compose -f docker/dev/docker-compose.yml up -d --build api
```

---

## Phase 5 — API FastAPI ✓

**Endpoints** : `GET /health`, `POST /predict`, `POST /predict/batch` (vectorisé, ≤ 5000), `GET /model/info`, `POST /model/reload`, `GET /metrics`

**Pitfall** : MLflow logue XGBoost avec le flavor `xgboost` (pas `sklearn`). `mlflow.sklearn.load_model` échoue sur le modèle en Production. Fix : `registry.load_model()` essaie sklearn puis xgboost en fallback.

**Tests** : 13 tests dans `tests/test_api.py` — mock via `patch("src.api.main.load_model_state", return_value=ModelState(...))` + `TestClient`. Lancer avec `.venv313/Scripts/python.exe -m pytest tests/ -v` (36 tests)

**Venv** : `.venv313/` (Python 3.13, gitignored) — le `.venv/` original n'avait pas pip.

Détail complet dans `contexte/roadmap.md`.

## Commandes de développement

```bash
# Lancer la stack complète en dev
docker compose -f docker/dev/docker-compose.yml up

# API uniquement
uvicorn src.api.main:app --reload

# Lancer le preprocessing
python -m src.preprocessing.pipeline

# Entraîner le modèle
python -m src.training.train

# Lancer les tests (venv313 requis)
.venv313/Scripts/python.exe -m pytest tests/ -v

# Lancer un test précis
.venv313/Scripts/python.exe -m pytest tests/test_preprocessing.py::test_engineer_features_columns_and_order -v

# Linter
.venv313/Scripts/python.exe -m ruff check src/

# DVC
dvc pull        # récupérer les données
dvc push        # versionner les artefacts

# MLflow — registry
python -m src.training.registry list
python -m src.training.registry rollback --version N

# Contrôle de dérive (rapport Evidently + verdict)
python -m src.monitoring.alert

# Simuler une dérive amplifiée (optionnel — la dérive naturelle incoming vs train suffit)
python -m src.retraining.scripts.simulate_drift --noise 0.3

# Tuning XGBoost → best_params.json
python -m src.training.tuner --sample 100000 --n-iter 20

# Valider le modèle exporté (gate CI)
python -m src.training.validate --model model_artifacts/model.joblib --data tests/fixtures/sample_test.csv

# Bench latence API
python scripts/bench_latency.py --url http://localhost:8001
```

## Stack technique

| Brique | Outil |
|---|---|
| ML | Scikit-learn + XGBoost |
| API | FastAPI + Uvicorn |
| Versioning modèles | MLflow |
| Versioning données | DVC |
| Orchestration | Apache Airflow |
| Monitoring dérive | Evidently |
| CI/CD | GitHub Actions |
| Containerisation | Docker Compose (dev/prod séparés) |
| Infra prod | AWS (EC2 + S3 + SSM) via Terraform |

## Données rivalytics (archivé — abandonné Phase 3)

5 tables, 500 comptes fictifs. Corrélations < 0.09, F1 max 0.52. Voir `contexte/problematiques-rencontrees.md`.

## Données muhammadshahidazeem (dataset actif)

- **Fichiers** : `data/customer_churn_dataset-training-master.csv` (440 832 lignes) + `data/customer_churn_dataset-testing-master.csv` (64 374 lignes)
- **Target** : `Churn` (float 0.0/1.0). **Taux : 56.7% train / 47.4% test** — incohérence train/test à surveiller.
- **Features** : `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`
- **Dataset plat** (1 seule table, pas de jointure)

### Learnings EDA (Phase 1 — nouveau dataset)

- `Support Calls` corrélation 0.574 avec Churn — feature la plus discriminante de loin.
- `Total Spend` corrélation 0.429 (inversé : plus on dépense, plus on churne — probable artefact synthétique).
- `Payment Delay` corrélation 0.312 — fort signal de risque.
- `Contract Length = Monthly` → **100% churn** — fuite de données probable. À encoder avec précaution ou exclure.
- `Subscription Type` quasi non-discriminant (55.9–58.2% sur toutes les modalités).
- `Tenure` et `Usage Frequency` corrélations faibles (< 0.06) malgré p-values significatives (effet volume).
- Train/test split fourni **mais distributions différentes** (voir Phase 10) : test scindé en incoming (réentraînement) / hold-out (évaluation).

### Feature Engineering Phase 2 — implémenté ✓

**15 features + cible** (`data/processed/features_engineered.csv`) — ordre canonique dans `FEATURE_COLUMNS` (`features.py`) = ordre d'entraînement du modèle (one-hot trié alphabétiquement : Basic, Premium, Standard) :

| Type | Colonnes |
|---|---|
| Numériques brutes | Age, Gender (0/1), Tenure, Usage Frequency, Support Calls, Payment Delay, Contract Length (ordinal), Total Spend, Last Interaction |
| One-hot | Subscription Type_Basic, Subscription Type_Standard, Subscription Type_Premium |
| Dérivées | `support_intensity` = Support Calls / (Tenure+1), `spend_per_month` = Total Spend / (Tenure+1), `payment_risk_score` = Payment Delay × Support Calls |
| Target | `churn_flag` (int 0/1) |

**Encodages** :
- `Gender` : male=1, female=0
- `Contract Length` : Monthly=0, Quarterly=1, Annual=2 — Monthly=100% churn dans les données brutes, mais importance XGBoost = 12.5% (non dominant, conservé)
- `Subscription Type` : one-hot sans drop_first (non-ordinal)

**Modules** :
- `loader.py` : `load_train()` / `load_test()` — charge les 2 CSV bruts
- `cleaner.py` : `clean()` — drop CustomerID, dropna, encode Gender
- `features.py` : `engineer_features()` — features dérivées + encodages + rename target
- `pipeline.py` : `run()` — orchestre et exporte train + test processés
- `merger.py` : supprimé Phase 10 (dataset plat)

## Architecture des modules clés

- `src/preprocessing/pipeline.py` — point d'entrée : charge les 2 CSV muhammadshahidazeem, nettoie, produit `data/processed/features_engineered.csv` (train, 440k) et `features_engineered_test.csv` (test, 64k)
- `src/preprocessing/loader.py` — `load_train()` / `load_test()`
- `src/preprocessing/cleaner.py` — `clean()` : drop CustomerID, dropna, encode Gender
- `src/preprocessing/features.py` — `engineer_features()` : 3 features dérivées + encodages + rename target
- `src/training/train.py` — entraîne 3 modèles (LogReg, RandomForest, XGBoost), log dans MLflow, promeut le meilleur en `Production`
- `src/training/registry.py` — fonctions `promote_model()` et `rollback_to_version(n)` via MLflow Registry
- `src/api/main.py` — FastAPI, charge le modèle depuis MLflow Registry au démarrage
- `src/api/schemas.py` — Pydantic : `PredictRequest`, `PredictResponse`, `BatchPredictRequest`, `ModelInfoResponse`
- `src/monitoring/drift_report.py` — génère rapport Evidently JSON + HTML dans `reports/`
- `src/monitoring/alert.py` — lit le rapport JSON, déclenche alerte si drift_score > 0,2 ou F1 drop > 0,05
- `src/retraining/dags/batch_scoring_dag.py` — DAG quotidien 2h00 : score tous les comptes actifs
- `src/retraining/dags/retraining_dag.py` — DAG déclenché sur dérive : retrain → évalue → promeut ou rollback
- `src/retraining/scripts/simulate_drift.py` — injecte du bruit dans les données de test (démo jury)

## Services Docker (stack dev)

| Service | Port hôte | Rôle |
|---|---|---|
| `api` | **8001** | FastAPI — endpoints /predict, /predict/batch, /model/info, /health |
| `mlflow` | 5000 | MLflow Tracking Server + Model Registry |
| `airflow-webserver` | 8080 | Airflow UI |
| `airflow-scheduler` | — | Exécution des DAGs |
| `postgres` | 5432 | Serveur PostgreSQL — 3 bases : `churnguard`, `mlflow`, `airflow` |

**Note** : port 8001 (et non 8000) — SeoLap occupe déjà le port 8000 sur cette machine. À ajuster si déployé sur un serveur dédié.

## Conventions

- Python 3.11, snake_case, type hints 3.9+
- Pas de mock de base de données dans les tests
- Docker Compose séparés dev/prod — jamais `docker/docker-compose.yml`
- `contexte/`, `.claude/`, `docs/soutenance*`, `docs/audit.md`, `docs/journal-session.md`, `docs/livrables-soutenances.md`, `docs/attendus-theoriques.md` exclus par le gitignore (documents de travail soutenance, même structure que le projet spotify-data-governance) ; le reste de `docs/` est versionné (livrables jury)
- Modèle MLflow Registry nommé `churnguard-model`, stage `Production` = modèle actif servi par l'API

## Phase 4 — DVC Versioning ✓

**Remote DagsHub** : `https://dagshub.com/emelineroblot/churnguard.dvc`

**Fichiers trackés** (64 MB) :
- `data/customer_churn_dataset-training-master.csv` (22 MB) — raw train
- `data/customer_churn_dataset-testing-master.csv` (3 MB) — raw test
- `data/processed/features_engineered.csv` (35 MB) — features train (référence) — via `dvc.yaml` stage `preprocess` (dvc.lock)
- `data/processed/features_incoming.csv` (2 MB) — nouvelles données de production labellisées (½ test Kaggle)
- `data/processed/features_engineered_test.csv` (2 MB) — hold-out (½ test Kaggle)
- `model_artifacts/` — modèle exporté + `model_info.json` (`model_artifacts.dvc`)

**Auth DagsHub** : stockée dans `.dvc/config.local` (gitignored). À reconfigurer sur nouvelle machine :
```bash
python -m dvc remote modify dagshub --local auth basic
python -m dvc remote modify dagshub --local user emelineroblot
python -m dvc remote modify dagshub --local password <token>
```

**Note env** : DVC 3.67.1 dans `.venv313/` (`python -m dvc`). Le `.venv` original n'avait pas pip — remplacé par `.venv313/` (Python 3.13, toutes dépendances installées). Voir P7 + P9 dans `contexte/problematiques-rencontrees.md`.

---

## Résultats Phase 3 — Entraînement ✓

| Modèle | F1 | AUC | Threshold optimal |
|---|---|---|---|
| LogisticRegression | 0.885 | 0.945 | 0.45 |
| RandomForest | 0.992 | 0.999 | 0.23 |
| **XGBoost** | **0.999** | **1.000** | **0.15** |

**⚠️ Ces scores sont mesurés sur un split du fichier train.** Sur le hold-out (fichier test Kaggle, distribution différente) le même XGBoost fait **F1 0.657 / AUC 0.731**. Après réentraînement avec la fenêtre récente (Phase 10) : **F1 0.978 / AUC 0.995**. Voir `docs/model-card.md`.

**Modèle en Production (stack Docker, 2026-09-16)** : v5 (réentraîné par le DAG, F1 hold-out 0.9777, seuil 0.89) ; v1 = baseline archivée (F1 0.6566). `model_artifacts/` (DVC) contient le modèle réentraîné (F1 hold-out 0.978, seuil 0.89).

Top features XGBoost (importance) : Total Spend (21%) > Support Calls (18%) > Contract Length (12%) > payment_risk_score (11%) > Payment Delay (10%).

## Objectifs de performance modèle

- F1-score ≥ 0,75 sur le dataset muhammadshahidazeem ✓ (atteint : 0.999)
- Latence API < 200ms ✓ (mesuré 2026-09-16 : p95 59 ms sur /predict, 1 000 comptes en 110 ms)
- Rollback MLflow en < 5 minutes ✓ (mesuré : 6 s, registry + `/model/reload`)

## Périmètre exclu

Kubernetes, intégration production SeoLap, appel Mautic réel (simulé en démo), interface front-end.
