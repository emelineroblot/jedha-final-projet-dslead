import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churnguard-model")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

_model = None


def _load_from_registry(model_uri: str):
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        return mlflow.xgboost.load_model(model_uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    _model = _load_from_registry(model_uri)
    yield
    _model = None


app = FastAPI(
    title="ChurnGuard API",
    description="Prédiction de churn SeoLap — pipeline MLOps Jedha",
    version="1.0.0",
    lifespan=lifespan,
)


def _score_to_risk(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    X = pd.DataFrame([request.features])
    score = float(_model.predict_proba(X)[0, 1])
    return PredictResponse(
        account_id=request.account_id,
        churn_score=round(score, 4),
        churn_risk=_score_to_risk(score),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    results = []
    for account in request.accounts:
        X = pd.DataFrame([account.features])
        score = float(_model.predict_proba(X)[0, 1])
        results.append(
            PredictResponse(
                account_id=account.account_id,
                churn_score=round(score, 4),
                churn_risk=_score_to_risk(score),
            )
        )
    return BatchPredictResponse(results=results)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise HTTPException(status_code=404, detail=f"Aucune version en stage '{MODEL_STAGE}'")
    v = versions[0]
    run = client.get_run(v.run_id)
    metrics = {k: float(val) for k, val in run.data.metrics.items()}
    trained_at = datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc).isoformat()
    feature_count = int(_model.n_features_in_) if _model is not None and hasattr(_model, "n_features_in_") else 0
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        version=v.version,
        stage=MODEL_STAGE,
        metrics=metrics,
        trained_at=trained_at,
        feature_count=feature_count,
    )
