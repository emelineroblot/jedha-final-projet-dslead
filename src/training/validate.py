"""
Gate de validation du modèle : F1 sur un jeu labellisé ≥ seuil, sinon code de sortie 1.

Utilisé par la CI (job `validate-model`) avant de construire l'image, et utilisable
à la main avant une promotion :

    python -m src.training.validate --model model_artifacts/model.joblib \
        --data tests/fixtures/sample_test.csv --min-f1 0.75
    python -m src.training.validate --model models:/churnguard-model/Production --data data/processed/features_engineered_test.csv
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.paths import TARGET
from src.preprocessing.features import FEATURE_COLUMNS
from src.training.evaluate import compute_metrics


def load_any(model_ref: str) -> tuple[object, float]:
    """
    Chemin joblib local ou URI MLflow (models:/, runs:/).
    Retourne (modèle, seuil de décision) — le seuil vient de model_info.json (fichier) ou des params du run.
    """
    if model_ref.startswith(("models:/", "runs:/")):
        from src.training.registry import get_production_info, load_model

        model = load_model(model_ref)
        threshold = 0.5
        if model_ref.startswith("models:/"):
            info = get_production_info()
            threshold = info["decision_threshold"] if info else 0.5
        else:
            from mlflow.tracking import MlflowClient

            run_id = model_ref.split("/")[1]
            threshold = float(MlflowClient().get_run(run_id).data.params.get("decision_threshold", 0.5))
        return model, threshold

    import joblib

    model = joblib.load(model_ref)
    info_path = Path(model_ref).parent / "model_info.json"
    threshold = json.loads(info_path.read_text()).get("decision_threshold", 0.5) if info_path.exists() else 0.5
    return model, float(threshold)


def validate(model_ref: str, data_path: Path, min_f1: float, threshold: Optional[float] = None) -> dict:
    df = pd.read_csv(data_path)
    X, y = df[FEATURE_COLUMNS], df[TARGET].astype(int)
    model, model_threshold = load_any(model_ref)
    metrics = compute_metrics(y.values, model.predict_proba(X)[:, 1], threshold if threshold is not None else model_threshold)
    metrics["n_rows"] = int(len(df))
    metrics["passed"] = metrics["f1_score"] >= min_f1
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation du modèle (gate CI)")
    parser.add_argument("--model", required=True, help="Chemin .joblib ou URI MLflow")
    parser.add_argument("--data", type=Path, required=True, help="CSV de features labellisé")
    parser.add_argument("--min-f1", type=float, default=0.75)
    parser.add_argument("--threshold", type=float, default=None, help="Défaut : seuil du modèle (model_info.json / run MLflow)")
    args = parser.parse_args()

    result = validate(args.model, args.data, args.min_f1, args.threshold)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        print(f"ÉCHEC : F1 {result['f1_score']:.4f} < {args.min_f1}", file=sys.stderr)
        sys.exit(1)
    print(f"OK : F1 {result['f1_score']:.4f} >= {args.min_f1}")
