import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "churnguard-model"


def promote_model(run_id: str, stage: str = "Production") -> None:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage=stage,
        archive_existing_versions=True,
    )
    print(f"Modèle version {mv.version} promu en {stage}")


def rollback_to_version(version: int) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Rollback effectué : version {version} en Production")


def get_production_version() -> str:
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("Aucun modèle en Production dans le registry")
    return versions[0].version
