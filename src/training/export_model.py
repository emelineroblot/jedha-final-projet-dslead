"""
Exporte le modèle Production depuis MLflow vers model_artifacts/
pour un déploiement standalone (HuggingFace Spaces, etc.) sans serveur MLflow.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churnguard-model")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
OUT_DIR = Path(__file__).parents[2] / "model_artifacts"


def export() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        model = mlflow.xgboost.load_model(model_uri)

    model_path = OUT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Modele exporte : {model_path}")

    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    v = versions[0]
    run = client.get_run(v.run_id)
    metrics = {k: float(val) for k, val in run.data.metrics.items()}
    trained_at = datetime.fromtimestamp(
        v.creation_timestamp / 1000, tz=timezone.utc
    ).isoformat()
    feature_count = (
        int(model.n_features_in_)
        if hasattr(model, "n_features_in_")
        else 15
    )

    info = {
        "model_name": MODEL_NAME,
        "version": v.version,
        "stage": MODEL_STAGE,
        "metrics": metrics,
        "trained_at": trained_at,
        "feature_count": feature_count,
    }
    info_path = OUT_DIR / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Metadonnees exportees : {info_path}")
    print(f"F1={metrics.get('f1_score', '?')} | features={feature_count}")


if __name__ == "__main__":
    export()
