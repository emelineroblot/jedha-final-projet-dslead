"""
ChurnGuard API — sert le modèle en Production (MLflow Registry) ou un artefact local (MODEL_PATH).

Observabilité : latence par requête (log JSON + header X-Process-Time-Ms), métriques Prometheus
sur /metrics, stockage des prédictions en base si DATABASE_URL est défini (boucle de monitoring).
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)
from src.monitoring import store
from src.preprocessing.features import FEATURE_COLUMNS

logger = logging.getLogger("churnguard.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churnguard-model")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MODEL_PATH = os.getenv("MODEL_PATH")  # si défini, charge depuis fichier (mode standalone HF / CI)
API_VERSION = "1.1.0"

RISK_MEDIUM, RISK_HIGH = 0.4, 0.7

PREDICTION_LATENCY = Histogram(
    "churnguard_prediction_latency_ms", "Latence du scoring (ms)", ["endpoint"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
)
PREDICTIONS_TOTAL = Counter("churnguard_predictions_total", "Comptes scorés", ["risk"])
MODEL_VERSION_GAUGE = Gauge("churnguard_model_version", "Version du modèle servie")


@dataclass
class ModelState:
    model: object
    version: str
    decision_threshold: float
    trained_at: str
    metrics: dict = field(default_factory=dict)
    source: str = "mlflow"
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


_state: Optional[ModelState] = None


def _load_from_file(path: str) -> ModelState:
    import joblib

    model = joblib.load(path)
    info_path = Path(path).parent / "model_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return ModelState(
        model=model,
        version=str(info.get("version", "file")),
        decision_threshold=float(info.get("decision_threshold", 0.5)),
        trained_at=info.get("trained_at", "unknown"),
        metrics=info.get("metrics", {}),
        source="file",
        feature_names=list(getattr(model, "feature_names_in_", FEATURE_COLUMNS)),
    )


def _load_from_registry() -> ModelState:
    from src.training.registry import get_production_info, load_model

    info = get_production_info()
    if info is None:
        raise RuntimeError(f"Aucune version de {MODEL_NAME} en stage {MODEL_STAGE}")
    model = load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    return ModelState(
        model=model,
        version=info["version"],
        decision_threshold=info["decision_threshold"],
        trained_at=info["trained_at"],
        metrics=info["metrics"],
        source="mlflow",
        feature_names=list(getattr(model, "feature_names_in_", FEATURE_COLUMNS)),
    )


def load_model_state() -> ModelState:
    """Point d'entrée unique de chargement (mocké dans les tests)."""
    return _load_from_file(MODEL_PATH) if MODEL_PATH else _load_from_registry()


def _set_state(state: Optional[ModelState]) -> None:
    global _state
    _state = state
    if state is not None:
        try:
            MODEL_VERSION_GAUGE.set(float(state.version))
        except ValueError:
            MODEL_VERSION_GAUGE.set(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store.init_db()
    except Exception as exc:
        logger.warning("Base de prédictions indisponible (%s) — stockage désactivé", exc)
    try:
        _set_state(load_model_state())
        logger.info("Modèle chargé : v%s (%s, seuil %.2f)", _state.version, _state.source, _state.decision_threshold)
    except Exception as exc:
        logger.warning("Modèle indisponible au démarrage (%s) — /predict retournera 503", exc)
    yield
    _set_state(None)


app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Prédiction de churn SeoLap — pipeline MLOps Jedha. "
        "Le modèle servi est la version `Production` du registry MLflow `churnguard-model`."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    if request.url.path not in ("/metrics", "/health"):
        logger.info(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "model_version": _state.version if _state else None,
        }))
    return response


def _score_to_risk(score: float) -> str:
    if score >= RISK_HIGH:
        return "high"
    if score >= RISK_MEDIUM:
        return "medium"
    return "low"


def _require_model() -> ModelState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé — vérifier MLflow ou MODEL_PATH")
    return _state


def _score_frame(state: ModelState, rows: list[dict]) -> np.ndarray:
    """Une seule passe predict_proba sur tout le batch, colonnes dans l'ordre du modèle."""
    X = pd.DataFrame(rows)[state.feature_names]
    return state.model.predict_proba(X)[:, 1]


def _build_responses(state: ModelState, requests_: list[PredictRequest], scores: np.ndarray) -> list[PredictResponse]:
    results = []
    for req, score in zip(requests_, scores):
        score = float(score)
        risk = _score_to_risk(score)
        PREDICTIONS_TOTAL.labels(risk=risk).inc()
        results.append(
            PredictResponse(
                account_id=req.account_id,
                churn_score=round(score, 4),
                churn_risk=risk,
                churn_predicted=score >= state.decision_threshold,
                model_version=state.version,
            )
        )
    return results


def _persist(requests_: list[PredictRequest], results: list[PredictResponse], latency_ms: float) -> None:
    store.log_predictions([
        {
            "account_id": r.account_id,
            "features": req.features.as_row(),
            "churn_score": r.churn_score,
            "churn_risk": r.churn_risk,
            "model_version": r.model_version,
            "latency_ms": latency_ms,
        }
        for req, r in zip(requests_, results)
    ])


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(status="ok", model_loaded=_state is not None, model_version=_state.version if _state else None)


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(request: PredictRequest, background: BackgroundTasks):
    state = _require_model()
    start = time.perf_counter()
    scores = _score_frame(state, [request.features.as_row()])
    result = _build_responses(state, [request], scores)[0]
    latency = (time.perf_counter() - start) * 1000
    PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency)
    if store.is_enabled():
        background.add_task(_persist, [request], [result], latency)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["scoring"])
def predict_batch(request: BatchPredictRequest, background: BackgroundTasks):
    state = _require_model()
    start = time.perf_counter()
    scores = _score_frame(state, [a.features.as_row() for a in request.accounts])
    results = _build_responses(state, request.accounts, scores)
    latency = (time.perf_counter() - start) * 1000
    PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(latency)
    if store.is_enabled():
        background.add_task(_persist, request.accounts, results, latency)
    return BatchPredictResponse(results=results, count=len(results), latency_ms=round(latency, 2))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info():
    state = _require_model()
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        version=state.version,
        stage=MODEL_STAGE if state.source == "mlflow" else "file",
        metrics=state.metrics,
        decision_threshold=state.decision_threshold,
        trained_at=state.trained_at,
        loaded_at=state.loaded_at,
        feature_count=len(state.feature_names),
        features=state.feature_names,
        source=state.source,
    )


@app.post("/model/reload", response_model=ReloadResponse, tags=["model"])
def model_reload():
    """
    Recharge le modèle Production sans redémarrer l'API. Appelé par le DAG de réentraînement
    après une promotion ou un rollback — c'est ce qui rend le rollback effectif en < 1 min.
    """
    previous = _state.version if _state else None
    try:
        new_state = load_model_state()
    except Exception as exc:
        logger.error("Rechargement impossible : %s", exc)
        return ReloadResponse(reloaded=False, previous_version=previous, version=previous, detail=str(exc))
    _set_state(new_state)
    logger.info("Modèle rechargé : v%s → v%s", previous, new_state.version)
    return ReloadResponse(
        reloaded=True,
        previous_version=previous,
        version=new_state.version,
        detail="unchanged" if previous == new_state.version else "model updated",
    )
