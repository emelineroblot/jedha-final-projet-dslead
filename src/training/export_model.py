"""
Exporte le modèle Production depuis MLflow vers model_artifacts/
pour un déploiement standalone (HuggingFace Spaces, CI) sans serveur MLflow.

    python -m src.training.export_model
"""
import json

import joblib

from src.paths import MODEL_ARTIFACTS_DIR
from src.training.registry import MODEL_NAME, PRODUCTION, get_production_info, load_model


def export() -> dict:
    MODEL_ARTIFACTS_DIR.mkdir(exist_ok=True)

    info = get_production_info()
    if info is None:
        raise RuntimeError(f"Aucune version de {MODEL_NAME} en {PRODUCTION}")

    model = load_model(f"models:/{MODEL_NAME}/{PRODUCTION}")
    model_path = MODEL_ARTIFACTS_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Modele exporte : {model_path}")

    payload = {
        "model_name": MODEL_NAME,
        "version": info["version"],
        "stage": PRODUCTION,
        "metrics": info["metrics"],
        "decision_threshold": info["decision_threshold"],
        "trained_at": info["trained_at"],
        "feature_count": int(getattr(model, "n_features_in_", 15)),
        "tags": {k: v for k, v in info["tags"].items() if not k.startswith("mlflow.")},
    }
    info_path = MODEL_ARTIFACTS_DIR / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Metadonnees exportees : {info_path}")
    print(f"v{payload['version']} | F1={payload['metrics'].get('f1_score', '?')} | features={payload['feature_count']}")
    return payload


if __name__ == "__main__":
    export()
