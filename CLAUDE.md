# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

**ChurnGuard** — Pipeline MLOps end-to-end de prédiction de churn pour SeoLap (projet final certification Jedha Data Science Lead).

Entraînement sur le dataset **muhammadshahidazeem** (Kaggle, 440k lignes, subscription service générique), réentraînable sur les données réelles SeoLap. Les 6 composantes MLOps sont toutes obligatoires : preprocessing → modèle → API → CI/CD → monitoring → réentraînement automatisé.

**Note** : le dataset rivalytics (500 comptes fictifs) a été abandonné en Phase 3 — corrélations < 0.09, F1 plafonné à 0.52. Voir `contexte/problematiques-rencontrees.md` P1.

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

- **rivalytics_accounts** — 500 lignes. Target : `churn_flag` (22% True / 78% False). Clé : `account_id`. CSV dans `data/` (pas `data/raw/`).
- **rivalytics_subscriptions** — 5 000 lignes (10/compte). `end_date` null = actif (4 514 actifs). `mrr_amount=0` = trial (778). `is_trial` déjà présent dans le CSV, ne pas recalculer. Clé de jointure avec `feature_usage` : `subscription_id`.
- **rivalytics_feature_usage** — 25 000 lignes. **Pas de `account_id` direct** : joint via `subscription_id → subscriptions → account_id`. 40 features génériques, `error_count`, `is_beta_feature`.
- **rivalytics_churn_events** — 600 lignes (539 hors réactivation). **Ne pas utiliser `churn_event_count` comme feature ML** : 339 comptes ont des events mais seulement 110 ont `churn_flag=True` → fuite de données garantie.
- **rivalytics_support_tickets** — 2 000 lignes. `satisfaction_score` : 825 nulls (41%) → imputation médiane. 8 comptes sans ticket → nulls dans le merge, imputer à 0.

### Données rivalytics (archivé — abandonné Phase 3)

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
- Train/test split déjà fourni — utiliser les fichiers tels quels, pas de re-split.

### Preprocessing Phase 2 (nouveau dataset)

1. Drop `CustomerID`
2. `Gender` → LabelEncoder (0/1)
3. `Subscription Type` → one-hot (Basic/Standard/Premium non-ordinal confirmé)
4. `Contract Length` → ordinal (Monthly=0, Quarterly=1, Annual=2) **ou** exclure si fuite confirmée
5. StandardScaler sur numériques pour LogReg
6. 1 null par colonne (ligne malformée) → `dropna()`

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

- F1-score ≥ 0,75 sur le dataset muhammadshahidazeem (benchmarks publiés : XGBoost F1 > 0.80)
- Latence API < 200ms
- Rollback MLflow en < 5 minutes

## Périmètre exclu

Kubernetes, intégration production SeoLap, appel Mautic réel (simulé en démo), interface front-end.
