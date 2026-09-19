"""
MLflow Model Registry : promotion, rollback, inspection.

CLI :
    python -m src.training.registry list
    python -m src.training.registry promote --run-id <id>
    python -m src.training.registry set-production --version N   (mise en service initiale)
    python -m src.training.registry rollback --version N
"""
import os
from datetime import datetime, timezone
from typing import Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churnguard-model")
PRODUCTION = "Production"


def load_model(model_uri: str):
    """Charge un modèle quel que soit son flavor (sklearn ou xgboost)."""
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        return mlflow.xgboost.load_model(model_uri)


def promote_model(run_id: str, stage: str = PRODUCTION) -> str:
    """
    Promeut le modèle d'un run (archive l'ancienne version en Production). Retourne la version.
    Si le run est déjà enregistré (train(auto_promote=False)), sa version existante est réutilisée —
    pas de doublon dans le registry.
    """
    client = MlflowClient()
    existing = [mv for mv in client.search_model_versions(f"name='{MODEL_NAME}'") if mv.run_id == run_id]
    if existing:
        version = max(existing, key=lambda mv: int(mv.version)).version
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage=stage, archive_existing_versions=True
    )
    print(f"Modèle version {version} promu en {stage}")
    return str(version)


def rollback_to_version(version: int) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME, version=str(version), stage=PRODUCTION, archive_existing_versions=True
    )
    print(f"Rollback effectué : version {version} en {PRODUCTION}")


def get_production_version() -> str:
    versions = MlflowClient().get_latest_versions(MODEL_NAME, stages=[PRODUCTION])
    if not versions:
        raise RuntimeError("Aucun modèle en Production dans le registry")
    return versions[0].version


def get_production_info() -> Optional[dict]:
    """Version, run_id, métriques, seuil de décision et date du modèle en Production (None si absent)."""
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[PRODUCTION])
    if not versions:
        return None
    v = versions[0]
    run = client.get_run(v.run_id)
    return {
        "version": str(v.version),
        "run_id": v.run_id,
        "metrics": {k: float(val) for k, val in run.data.metrics.items()},
        "decision_threshold": float(run.data.params.get("decision_threshold", 0.5)),
        "trained_at": datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc).isoformat(),
        "creation_timestamp": v.creation_timestamp,
        "tags": dict(run.data.tags),
    }


def list_versions() -> list[dict]:
    client = MlflowClient()
    rows = []
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        run = client.get_run(mv.run_id)
        rows.append({
            "version": int(mv.version),
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "model_type": run.data.params.get("model_type", "?"),
            "f1_score": run.data.metrics.get("f1_score"),
            "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
        })
    return sorted(rows, key=lambda r: r["version"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ChurnGuard — MLflow Model Registry")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Liste les versions et leur stage")
    p_promote = sub.add_parser("promote", help="Enregistre + promeut le modèle d'un run")
    p_promote.add_argument("--run-id", required=True)
    p_rollback = sub.add_parser("rollback", help="Remet une version en Production")
    p_rollback.add_argument("--version", type=int, required=True)
    p_set = sub.add_parser("set-production", help="Met une version en Production (mise en service initiale, hors gate F1)")
    p_set.add_argument("--version", type=int, required=True)
    args = parser.parse_args()

    if args.command == "list":
        for r in list_versions():
            f1 = f"{r['f1_score']:.4f}" if r["f1_score"] is not None else "-"
            print(f"v{r['version']:<3} {r['stage']:<11} {r['model_type']:<20} F1={f1}  {r['created']}  run={r['run_id'][:8]}")
    elif args.command == "promote":
        promote_model(args.run_id)
    elif args.command in ("rollback", "set-production"):
        rollback_to_version(args.version)


if __name__ == "__main__":
    main()
