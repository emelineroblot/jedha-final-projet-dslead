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
| 10 | Documentation & diagramme | S4 | ⏳ |
| 11 | Présentation jury | S5 | ⏳ |

**Branche courante** : `develop` — phases 0–9 sur develop (0–3 mergées depuis `feature/training`, phases 4–9 committées directement).

## Phase 7 — CI/CD GitHub Actions ✓

**Fichier** : `.github/workflows/ci.yml`

**Jobs** :
- `test` : lint ruff + `pytest tests/test_api.py` — install légère (`requirements-api.txt` + pytest/ruff), pas d'airflow ni dvc
- `build` : build image Docker + push GHCR (`ghcr.io/<org>/churnguard-api`) — tags `branch` + `branch-<sha>`
- `deploy` : désactivé (`if: false`) — à activer Phase 10 après config secrets Hetzner

**Triggers** : push sur `develop` ou `main`, PR vers `main`

**Pitfall P14** : `test_preprocessing.py` cible l'ancien code rivalytics (5 tables, `load_all()`) — exclu du pipeline CI. À réécrire pour le dataset muhammadshahidazeem avant réactivation.

**Pitfall P15** : `pip install -e ".[dev]"` installe apache-airflow → build CI > 5 min. Toujours utiliser `requirements-api.txt` pour les jobs de test.

---

## Phase 9 — Monitoring Evidently ✓

**Fichiers** :
- `src/monitoring/drift_report.py` — `generate_drift_report(reference, current, target_column=None, prediction_column=None)` : DataDriftPreset obligatoire, ClassificationPreset conditionnel si prediction_column fourni
- `src/monitoring/alert.py` — `check_drift()` : génère le rapport si absent (données train vs test ou drifted), parse le JSON Evidently, retourne `True` si `drift_share > 0.2` ou F1 drop `> 0.05`

**Déclenchement de la démo** :
```bash
# 1. Injecter une dérive artificielle
python src/retraining/scripts/simulate_drift.py --noise 0.3

# 2. Vérifier que la dérive est détectée
python -c "from src.monitoring.alert import check_drift; print(check_drift())"
```

**Pitfall P17** : `as_dict()` d'Evidently encode le nom de la metric sous la clé `"metric"` (ex: `"DatasetDriftMetric"`). Pour parser, tester `"DatasetDrift" in metric_name` (substring) plutôt que l'égalité stricte — le nom exact peut varier selon la version d'Evidently.

---

## Phase 8 — Orchestration Airflow ✓

**DAGs** :
- `src/retraining/dags/batch_scoring_dag.py` — schedule `0 2 * * *` : charge le CSV test (500 premières lignes en démo), appelle `/predict/batch` par chunks de 100, simule l'envoi Mautic
- `src/retraining/dags/retraining_dag.py` — schedule `0 3 * * 1` (lundi 3h) : `check_drift` (BranchPythonOperator) → `retrain_model` (`auto_promote=False`) → `evaluate_model` (F1 candidat vs Production via XCom) → `promote_or_rollback`

**Variables d'environnement Airflow** :
- `CHURNGUARD_API_URL` : URL de l'API (défaut `http://api:8000` dans le conteneur Docker)
- `MLFLOW_TRACKING_URI` : doit pointer vers le serveur MLflow (configuré dans docker-compose.yml)

**Pitfall P18** : `train(auto_promote=False)` enregistre le modèle dans le Registry sans le transitionner en Production — indispensable pour que `evaluate_model` puisse comparer candidat vs Production courante avant de décider.

**Pitfall P19** : XCom Airflow est limité à ~48 KB. Ne jamais passer de DataFrames par XCom — toujours passer des chemins de fichiers ou des identifiants (run_id, version).

**Pitfall P20** : MLflow client 3.x appelle `/api/2.0/mlflow/logged-models` (endpoint inexistant en 2.x). `Dockerfile.mlflow` doit utiliser la même version majeure que le client local. Vérifier avec `python -c "import mlflow; print(mlflow.__version__)"` et aligner le Dockerfile en conséquence. Actuellement : client = serveur = **3.12.0** (`python:3.11-slim` + `pip install mlflow==3.12.0`).

**Pitfall P21** : Entraîner avec `MLFLOW_TRACKING_URI=http://localhost:5000` via PowerShell (`$env:MLFLOW_TRACKING_URI = "..."`) — ne pas utiliser le Bash tool pour les variables d'env sur Windows (syntaxe ignorée silencieusement). Sans cette variable, MLflow écrit en local (`mlruns/`) et le serveur Docker reste vide.

**Pitfall P22** : `ColumnMapping` d'Evidently a été déplacé dans `evidently.legacy.pipeline.column_mapping` en version 0.7.x. Les imports `from evidently import ColumnMapping` et `from evidently.metric_preset import DataDriftPreset` sont cassés — utiliser `from evidently.legacy.pipeline.column_mapping import ColumnMapping` et `from evidently.legacy.metric_preset import DataDriftPreset`.

---

## Phase 6 — Containerisation Docker ✓

**Fichiers** :
- `Dockerfile` — single-stage Python 3.11-slim, installe depuis `requirements-api.txt` uniquement
- `Dockerfile.mlflow` — `ghcr.io/mlflow/mlflow:v2.13.0` + `psycopg2-binary` (absent de l'image officielle)
- `.dockerignore` — exclut `data/`, `mlruns/`, `.venv313/`, `notebooks/` du contexte de build
- `requirements-api.txt` — 8 dépendances API seulement (pas airflow, pas dvc)
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

**Endpoints** : `GET /health`, `POST /predict`, `POST /predict/batch`, `GET /model/info`

**Pitfall** : MLflow logue XGBoost avec le flavor `xgboost` (pas `sklearn`). `mlflow.sklearn.load_model` échoue sur le modèle en Production. Fix : `_load_from_registry()` essaie sklearn puis xgboost en fallback (`src/api/main.py`).

**Tests** : 4 tests dans `tests/test_api.py` — mock via `patch("src.api.main._load_from_registry", ...)` + `TestClient`. Lancer avec `.venv313/Scripts/python.exe -m pytest tests/test_api.py -v`

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
.venv313/Scripts/python.exe -m pytest tests/test_preprocessing.py::test_feature_engineering -v

# Linter
.venv313/Scripts/python.exe -m ruff check src/

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
- Train/test split déjà fourni — utiliser les fichiers tels quels, pas de re-split.

### Feature Engineering Phase 2 — implémenté ✓

**16 features produites** (`data/processed/features_engineered.csv`) :

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
- `merger.py` : **déprécié** (dataset plat, plus de jointures)

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
- `contexte/` et `.claude/` exclus du gitignore (documents de cadrage non versionnés)
- Modèle MLflow Registry nommé `churnguard-model`, stage `Production` = modèle actif servi par l'API

## Phase 4 — DVC Versioning ✓

**Remote DagsHub** : `https://dagshub.com/emelineroblot/churnguard.dvc`

**Fichiers trackés** (64 MB) :
- `data/customer_churn_dataset-training-master.csv` (22 MB) — raw train
- `data/customer_churn_dataset-testing-master.csv` (3 MB) — raw test
- `data/processed/features_engineered.csv` (35 MB) — features train
- `data/processed/features_engineered_test.csv` (4 MB) — features test

**Auth DagsHub** : stockée dans `.dvc/config.local` (gitignored). À reconfigurer sur nouvelle machine :
```bash
python -m dvc remote modify dagshub --local auth basic
python -m dvc remote modify dagshub --local user emelineroblot
python -m dvc remote modify dagshub --local password <token>
```

**Note env** : DVC installé dans Python système (3.13). Le `.venv` original n'avait pas pip — remplacé par `.venv313/` (Python 3.13, toutes dépendances installées). Voir P7 + P9 dans `contexte/problematiques-rencontrees.md`.

---

## Résultats Phase 3 — Entraînement ✓

| Modèle | F1 | AUC | Threshold optimal |
|---|---|---|---|
| LogisticRegression | 0.885 | 0.945 | 0.45 |
| RandomForest | 0.992 | 0.999 | 0.23 |
| **XGBoost** | **0.999** | **1.000** | **0.15** |

**Modèle en Production** : XGBoost, version 3 du registry MLflow (`churnguard-model`).

**Note performances** : F1 quasi-parfait attendu sur dataset synthétique (les features reconstruisent presque parfaitement la cible). Contract Length = 12.5% de l'importance — pas la cause principale, voir P5 dans problematiques-rencontrees.md.

Top features XGBoost (importance) : Total Spend (21%) > Support Calls (18%) > Contract Length (12%) > payment_risk_score (11%) > Payment Delay (10%).

## Objectifs de performance modèle

- F1-score ≥ 0,75 sur le dataset muhammadshahidazeem ✓ (atteint : 0.999)
- Latence API < 200ms
- Rollback MLflow en < 5 minutes

## Périmètre exclu

Kubernetes, intégration production SeoLap, appel Mautic réel (simulé en démo), interface front-end.
