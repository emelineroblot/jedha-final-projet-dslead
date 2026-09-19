"""
DAG `auto_retraining` — réentraînement conditionnel, hebdomadaire (lundi 03:00).

check_drift ─┬─ no_retrain
             └─ retrain_model → evaluate_model → decide ─┬─ keep_current
                                                         └─ promote_model (+ reload API, smoke test, rollback si échec)

Déclencheurs du réentraînement : dérive détectée par Evidently (drift_share > 20 % ou chute de F1)
OU volume de nouvelles données reçues par l'API depuis le dernier entraînement ≥ NEW_DATA_MIN_ROWS.
XCom ne transporte que des identifiants et des scalaires (jamais de DataFrame).
"""
import os
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3.x
    from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import BranchPythonOperator, PythonOperator

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churnguard-model")
API_URL = os.getenv("CHURNGUARD_API_URL", "http://api:8000")
NEW_DATA_MIN_ROWS = int(os.getenv("NEW_DATA_MIN_ROWS", "5000"))
MIN_IMPROVEMENT = float(os.getenv("MIN_F1_IMPROVEMENT", "0.0"))

default_args = {
    "owner": "churnguard",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _notify(title: str, message: str, level: str = "info") -> None:
    from src.monitoring.notify import send_alert

    send_alert(title, message, level=level)


def _on_failure(context) -> None:
    ti = context.get("task_instance")
    _notify(
        "Échec du DAG auto_retraining",
        f"Tâche `{ti.task_id}` en erreur (run {context.get('run_id')}) : {context.get('exception')}",
        level="critical",
    )


with DAG(
    dag_id="auto_retraining",
    description="Réentraînement automatique sur détection de dérive ou nouvelles données",
    schedule="0 3 * * 1",  # hebdomadaire le lundi à 3h00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,  # un seul réentraînement à la fois (2 vCPU en prod ; runs planifié + manuel sérialisés)
    default_args={**default_args, "on_failure_callback": _on_failure},
    tags=["churnguard", "mlops"],
) as dag:

    def check_drift(**context) -> str:
        from datetime import timezone

        from src.monitoring import store
        from src.monitoring.alert import run_drift_check
        from src.training.registry import get_production_info

        ti = context["ti"]
        prod = get_production_info()
        ti.xcom_push(key="prod_version_before", value=prod["version"] if prod else None)

        result = run_drift_check()
        ti.xcom_push(key="drift", value=result.as_dict())

        new_rows = 0
        if prod:
            since = datetime.fromtimestamp(prod["creation_timestamp"] / 1000, tz=timezone.utc)
            new_rows = store.count_since(since)
        ti.xcom_push(key="new_rows", value=new_rows)

        reasons = []
        if result.drifted:
            reasons.append(f"dérive : {result.reason}")
        if new_rows >= NEW_DATA_MIN_ROWS:
            reasons.append(f"{new_rows} nouvelles lignes depuis le dernier entraînement")
        print(f"Dérive : {result.drifted} ({result.reason}) | nouvelles données : {new_rows} | prod : v{prod['version'] if prod else '-'}")

        if reasons:
            _notify("Réentraînement déclenché", " ; ".join(reasons), level="warning")
            return "retrain_model"
        return "no_retrain"

    def retrain_model(**context):
        from src.paths import FEATURES_DRIFTED_PATH, FEATURES_INCOMING_PATH
        from src.training.train import train

        # La fenêtre courante labellisée (données de production, ou version bruitée pour la démo)
        # est ajoutée à la référence pour que le candidat apprenne la nouvelle distribution.
        extra = FEATURES_DRIFTED_PATH if FEATURES_DRIFTED_PATH.exists() else FEATURES_INCOMING_PATH
        best_run_id = train(auto_promote=False, extra_data_path=extra)
        if best_run_id is None:
            raise RuntimeError("Aucun modèle entraîné")
        context["ti"].xcom_push(key="best_run_id", value=best_run_id)

    def evaluate_model(**context):
        import pandas as pd
        from mlflow.tracking import MlflowClient

        from src.paths import FEATURES_TEST_PATH, TARGET
        from src.preprocessing.features import FEATURE_COLUMNS
        from src.training.evaluate import evaluate
        from src.training.registry import get_production_info, load_model

        ti = context["ti"]
        best_run_id = ti.xcom_pull(task_ids="retrain_model", key="best_run_id")

        df = pd.read_csv(FEATURES_TEST_PATH)
        X_test, y_test = df[FEATURE_COLUMNS], df[TARGET].astype(int)

        client = MlflowClient()
        cand_threshold = float(client.get_run(best_run_id).data.params.get("decision_threshold", 0.5))
        new_metrics = evaluate(load_model(f"runs:/{best_run_id}/model"), X_test, y_test, cand_threshold)

        prod = get_production_info()
        if prod:
            prod_metrics = evaluate(load_model(f"models:/{MODEL_NAME}/Production"), X_test, y_test, prod["decision_threshold"])
        else:
            prod_metrics = {"f1_score": 0.0}

        print(f"Candidat F1 : {new_metrics['f1_score']:.4f} | Production F1 : {prod_metrics['f1_score']:.4f}")
        ti.xcom_push(key="new_f1", value=new_metrics["f1_score"])
        ti.xcom_push(key="prod_f1", value=prod_metrics["f1_score"])

    def decide(**context) -> str:
        ti = context["ti"]
        new_f1 = ti.xcom_pull(task_ids="evaluate_model", key="new_f1")
        prod_f1 = ti.xcom_pull(task_ids="evaluate_model", key="prod_f1")
        return "promote_model" if new_f1 > prod_f1 + MIN_IMPROVEMENT else "keep_current"

    def promote_model(**context):

        from src.training.registry import promote_model as _promote, rollback_to_version

        ti = context["ti"]
        best_run_id = ti.xcom_pull(task_ids="retrain_model", key="best_run_id")
        prev_version = ti.xcom_pull(task_ids="check_drift", key="prod_version_before")
        new_f1 = ti.xcom_pull(task_ids="evaluate_model", key="new_f1")
        prod_f1 = ti.xcom_pull(task_ids="evaluate_model", key="prod_f1")

        new_version = _promote(run_id=best_run_id, stage="Production")
        ti.xcom_push(key="new_version", value=new_version)

        try:
            _reload_and_smoke_test(expected_version=new_version)
        except Exception as exc:
            # Rollback : l'ancienne version revient en Production et l'API la recharge
            if prev_version:
                rollback_to_version(int(prev_version))
                try:
                    _reload_and_smoke_test(expected_version=str(prev_version))
                except Exception as exc2:
                    print(f"Rechargement après rollback impossible : {exc2}")
            _notify(
                "Rollback effectué",
                f"v{new_version} promue puis rejetée ({exc}) — retour à v{prev_version}",
                level="critical",
            )
            raise

        _notify(
            "Nouveau modèle en Production",
            f"v{prev_version} → v{new_version} (F1 hold-out {prod_f1:.4f} → {new_f1:.4f})",
            level="success",
        )
        print(f"Nouveau modèle promu en Production : v{new_version} (F1 {prod_f1:.4f} → {new_f1:.4f})")

    def _reload_and_smoke_test(expected_version: str) -> None:
        """Recharge le modèle dans l'API puis vérifie une prédiction — lève une exception sinon."""
        import pandas as pd
        import requests

        from src.paths import FEATURES_TEST_PATH
        from src.preprocessing.features import FEATURE_COLUMNS

        resp = requests.post(f"{API_URL}/model/reload", timeout=120)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("reloaded") or str(body.get("version")) != str(expected_version):
            raise RuntimeError(f"API non rechargée sur v{expected_version} : {body}")

        row = pd.read_csv(FEATURES_TEST_PATH, nrows=1)[FEATURE_COLUMNS].iloc[0].to_dict()
        resp = requests.post(
            f"{API_URL}/predict",
            json={"account_id": "smoke-test", "features": {k: float(v) for k, v in row.items()}},
            timeout=30,
        )
        resp.raise_for_status()
        score = resp.json()["churn_score"]
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Score invalide : {score}")
        print(f"Smoke test OK — API sert v{expected_version}, score={score}")

    def keep_current(**context):
        ti = context["ti"]
        new_f1 = ti.xcom_pull(task_ids="evaluate_model", key="new_f1")
        prod_f1 = ti.xcom_pull(task_ids="evaluate_model", key="prod_f1")
        msg = f"Candidat F1 {new_f1:.4f} ≤ Production {prod_f1:.4f} — modèle en Production inchangé"
        print(msg)
        _notify("Réentraînement sans promotion", msg, level="info")

    def no_retrain(**_):
        print("Pas de dérive ni de nouvelles données — réentraînement non nécessaire")

    t_check = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    t_retrain = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
    t_evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t_decide = BranchPythonOperator(task_id="decide", python_callable=decide)
    t_promote = PythonOperator(task_id="promote_model", python_callable=promote_model)
    t_keep = PythonOperator(task_id="keep_current", python_callable=keep_current)
    t_skip = PythonOperator(task_id="no_retrain", python_callable=no_retrain)

    t_check >> [t_retrain, t_skip]
    t_retrain >> t_evaluate >> t_decide >> [t_promote, t_keep]
