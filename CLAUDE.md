# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

**ChurnGuard** — Pipeline MLOps end-to-end de prédiction de churn pour SeoLap (projet final certification Jedha Data Science Lead).

Entraînement initial sur le dataset rivalytics (Kaggle, 500 comptes SaaS fictifs), réentraînable sur les données réelles SeoLap. Les 6 composantes MLOps sont toutes obligatoires : preprocessing → modèle → API → CI/CD → monitoring → réentraînement automatisé.

## Structure cible du repo

```
final-project-dslead/
├── data/                        # Données brutes rivalytics (DVC tracké)
├── notebooks/                   # EDA + expérimentations
├── src/
│   ├── preprocessing/           # Feature engineering (agrégations sur 5 tables)
│   ├── training/                # Entraînement Scikit-learn/XGBoost + évaluation
│   ├── api/                     # FastAPI app (POST /predict, POST /predict/batch, GET /model/info)
│   ├── monitoring/              # Rapports Evidently (data drift + model performance)
│   └── retraining/              # DAGs Airflow
├── tests/
├── docker/
│   ├── dev/docker-compose.yml
│   └── prod/docker-compose.yml
├── .github/workflows/           # CI/CD GitHub Actions
└── mlruns/                      # MLflow (gitignored, DVC tracké)
```

## Roadmap (11 phases)

| Phase | Intitulé | Semaine |
|---|---|---|
| 0 | Setup repo & environnement | S1 |
| 1 | EDA & Preprocessing | S1 |
| 2 | Feature Engineering | S1 |
| 3 | Entraînement & sélection du modèle | S1–S2 |
| 4 | MLflow + DVC — Versioning | S2 |
| 5 | API FastAPI | S2 |
| 6 | Containerisation Docker | S2 |
| 7 | Pipeline CI/CD GitHub Actions | S2–S3 |
| 8 | Orchestration Airflow | S3 |
| 9 | Monitoring Evidently | S3–S4 |
| 10 | Documentation & diagramme | S4 |
| 11 | Présentation jury | S5 |

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

# Lancer les tests
pytest tests/

# Lancer un test précis
pytest tests/test_preprocessing.py::test_feature_engineering -v

# Linter
ruff check src/

# DVC
dvc pull        # récupérer les données
dvc push        # versionner les artefacts

# MLflow — rollback vers une version précédente
python -m src.training.registry rollback --version N

# Simuler une dérive pour déclencher le réentraînement
python src/retraining/scripts/simulate_drift.py
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
| Hébergement | Hetzner VPS |

## Données rivalytics (5 tables)

- **rivalytics_accounts** — target : `churn_flag` (booléen). Clé : `account_id`.
- **rivalytics_subscriptions** — MRR=0 pour trials, `end_date` null = actif, upgrades/downgrades = signal fort.
- **rivalytics_feature_usage** — noms génériques `feature_1..N`, `error_count` (frustration), `is_beta_feature` (engagement).
- **rivalytics_churn_events** — `reason_code`, `feedback_text`, exclure `is_reactivation=True` pour éviter double comptage.
- **rivalytics_support_tickets** — `satisfaction_score` nullable à imputer, `escalation_flag` signal fort.

## Architecture des modules clés

- `src/preprocessing/pipeline.py` — point d'entrée : charge les 5 CSV, nettoie, joint, produit `data/processed/features_engineered.csv`
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

| Service | Port | Rôle |
|---|---|---|
| `api` | 8000 | FastAPI — endpoints /predict, /predict/batch, /model/info, /health |
| `mlflow` | 5000 | MLflow Tracking Server + Model Registry |
| `airflow-webserver` | 8080 | Airflow UI |
| `airflow-scheduler` | — | Exécution des DAGs |
| `postgres` | 5432 | Backend Airflow + MLflow |

## Conventions

- Python 3.11, snake_case, type hints 3.9+
- Pas de mock de base de données dans les tests
- Docker Compose séparés dev/prod — jamais `docker/docker-compose.yml`
- `contexte/` et `.claude/` exclus du gitignore (documents de cadrage non versionnés)
- Modèle MLflow Registry nommé `churnguard-model`, stage `Production` = modèle actif servi par l'API

## Objectifs de performance modèle

- F1-score ≥ 0,75 sur le dataset rivalytics
- Latence API < 200ms
- Rollback MLflow en < 5 minutes

## Périmètre exclu

Kubernetes, intégration production SeoLap, appel Mautic réel (simulé en démo), interface front-end.
