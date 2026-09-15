# ChurnGuard API — guide d'intégration

Base URL (stack dev) : `http://localhost:8001` · Space HF : `https://emeliner-churnguard-api.hf.space`
Documentation interactive : `GET /docs` (Swagger UI) · `GET /redoc` · schéma OpenAPI : `GET /openapi.json`

Version API : `1.1.0` · Aucune authentification (périmètre projet — à placer derrière un reverse proxy / clé API en production).

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | Liveness + version du modèle chargé |
| `POST` | `/predict` | Score de churn d'un compte |
| `POST` | `/predict/batch` | Score de 1 à 5 000 comptes en un appel (vectorisé) |
| `GET` | `/model/info` | Métadonnées du modèle servi (version, métriques, seuil, features) |
| `POST` | `/model/reload` | Recharge la version `Production` du registry sans redémarrer |
| `GET` | `/metrics` | Métriques Prometheus (latence, volume, version) |

## Features d'entrée (15)

Toutes obligatoires. Les noms avec espaces sont acceptés tels quels **ou** en `snake_case` (`Usage Frequency` ≡ `usage_frequency`). Toute feature inconnue, manquante ou hors bornes → **422** avec le détail Pydantic.

| Feature | Type | Bornes | Description |
|---|---|---|---|
| `Age` | float | 0–120 | Âge du client |
| `Gender` | int | 0/1 | 1 = homme, 0 = femme |
| `Tenure` | float | ≥ 0 | Ancienneté (mois) |
| `Usage Frequency` | float | ≥ 0 | Sessions par mois |
| `Support Calls` | float | ≥ 0 | Appels au support sur la période |
| `Payment Delay` | float | ≥ 0 | Jours de retard de paiement |
| `Contract Length` | int | 0–2 | 0 = Monthly, 1 = Quarterly, 2 = Annual |
| `Total Spend` | float | ≥ 0 | Dépense totale |
| `Last Interaction` | float | ≥ 0 | Jours depuis la dernière interaction |
| `support_intensity` | float | ≥ 0 | `Support Calls / (Tenure + 1)` |
| `spend_per_month` | float | ≥ 0 | `Total Spend / (Tenure + 1)` |
| `payment_risk_score` | float | ≥ 0 | `Payment Delay × Support Calls` |
| `Subscription Type_Basic` | int | 0/1 | One-hot offre Basic |
| `Subscription Type_Premium` | int | 0/1 | One-hot offre Premium |
| `Subscription Type_Standard` | int | 0/1 | One-hot offre Standard |

> Les 3 features dérivées sont calculées côté client (ou par `src.preprocessing.features.engineer_features`) — l'API attend les features finales.

## `POST /predict`

```bash
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{
  "account_id": "ACC-001",
  "features": {
    "Age": 35, "Gender": 1, "Tenure": 24, "Usage Frequency": 15, "Support Calls": 3,
    "Payment Delay": 10, "Contract Length": 1, "Total Spend": 800.0, "Last Interaction": 7,
    "support_intensity": 0.12, "spend_per_month": 32.0, "payment_risk_score": 30.0,
    "Subscription Type_Basic": 0, "Subscription Type_Premium": 0, "Subscription Type_Standard": 1
  }
}'
```

```json
{
  "account_id": "ACC-001",
  "churn_score": 0.1823,
  "churn_risk": "low",
  "churn_predicted": false,
  "model_version": "5"
}
```

| Champ | Description |
|---|---|
| `churn_score` | Probabilité de churn ∈ [0, 1] |
| `churn_risk` | Bande métier pour le CRM : `low` < 0.4 ≤ `medium` < 0.7 ≤ `high` |
| `churn_predicted` | `churn_score ≥ decision_threshold` — seuil du modèle optimisé sur validation (voir `/model/info`) |
| `model_version` | Version du registry MLflow qui a produit le score (traçabilité) |

Header de réponse : `X-Process-Time-Ms` (latence serveur).

## `POST /predict/batch`

```json
{ "accounts": [ {"account_id": "ACC-001", "features": {…}}, {"account_id": "ACC-002", "features": {…}} ] }
```

```json
{
  "results": [
    {"account_id": "ACC-001", "churn_score": 0.1823, "churn_risk": "low", "churn_predicted": false, "model_version": "5"},
    {"account_id": "ACC-002", "churn_score": 0.8910, "churn_risk": "high", "churn_predicted": true, "model_version": "5"}
  ],
  "count": 2,
  "latency_ms": 3.4
}
```

Limite : 5 000 comptes par appel. Le batch est scoré en **une seule passe** `predict_proba` (≈ 1 000 comptes en quelques dizaines de ms).

## `GET /model/info`

```json
{
  "model_name": "churnguard-model",
  "version": "5",
  "stage": "Production",
  "metrics": {"f1_score": 0.93, "precision": 0.92, "recall": 0.94, "auc_roc": 0.97, "auc_pr": 0.97, "val_f1_score": 0.96},
  "decision_threshold": 0.45,
  "trained_at": "2026-09-15T10:12:00+00:00",
  "loaded_at": "2026-09-15T10:15:42+00:00",
  "feature_count": 15,
  "features": ["Age", "Gender", "…"],
  "source": "mlflow"
}
```

`source` = `mlflow` (registry) ou `file` (artefact standalone `MODEL_PATH`, mode HF Space / CI).

## `POST /model/reload`

Recharge la version `Production` courante. Utilisé par le DAG `auto_retraining` après une promotion ou un rollback ; utilisable à la main après `python -m src.training.registry rollback --version N`.

```json
{"reloaded": true, "previous_version": "5", "version": "4", "detail": "model updated"}
```

En cas d'échec (MLflow injoignable), l'API **conserve le modèle courant** et répond `"reloaded": false` avec le motif.

## Codes d'erreur

| Code | Cas |
|---|---|
| `422` | Feature manquante / inconnue / hors bornes, `account_id` vide, batch vide ou > 5 000 |
| `503` | Modèle non chargé (MLflow injoignable au démarrage et aucun `MODEL_PATH`) — `/health` renvoie `model_loaded: false` |

## Observabilité

- Chaque requête est loguée en JSON : `{ts, method, path, status, latency_ms, model_version}`.
- `GET /metrics` expose `churnguard_prediction_latency_ms` (histogramme par endpoint), `churnguard_predictions_total{risk}` et `churnguard_model_version`, plus les métriques HTTP standard.
- Si `DATABASE_URL` est défini, chaque prédiction est stockée dans la table `predictions` (features, score, version, latence) — c'est la source « production » du contrôle de dérive Evidently.

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `MLFLOW_TRACKING_URI` | — | Serveur MLflow (registry) |
| `MLFLOW_MODEL_NAME` / `MLFLOW_MODEL_STAGE` | `churnguard-model` / `Production` | Modèle servi |
| `MODEL_PATH` | — | Si défini, charge `model.joblib` (+ `model_info.json` voisin) au lieu du registry |
| `DATABASE_URL` | — | PostgreSQL pour le stockage des prédictions (optionnel) |
| `LOG_LEVEL` | `INFO` | Niveau de log |

## Client Python minimal

```python
import httpx

api = httpx.Client(base_url="http://localhost:8001", timeout=30)
resp = api.post("/predict/batch", json={"accounts": accounts})  # accounts: list[{account_id, features}]
resp.raise_for_status()
for r in resp.json()["results"]:
    if r["churn_risk"] == "high":
        ...  # → segment CRM
```
