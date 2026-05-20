from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

MODEL_NAME = "churnguard-model"
TARGET = "churn_flag"

default_args = {
    "owner": "churnguard",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="auto_retraining",
    description="Réentraînement automatique sur détection de dérive",
    schedule="0 3 * * 1",  # hebdomadaire le lundi à 3h00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    def check_drift(**context) -> str:
        from mlflow.tracking import MlflowClient

        from src.monitoring.alert import check_drift as _check_drift

        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        prod_version = versions[0].version if versions else None
        context["ti"].xcom_push(key="prod_version_before", value=prod_version)

        drift = _check_drift()
        return "retrain_model" if drift else "no_retrain"

    def retrain_model(**context):
        from src.training.train import train

        best_run_id = train(auto_promote=False)
        context["ti"].xcom_push(key="best_run_id", value=best_run_id)

    def evaluate_model(**context):
        import mlflow.sklearn
        import mlflow.xgboost
        from sklearn.metrics import f1_score

        from src.preprocessing.cleaner import clean
        from src.preprocessing.features import engineer_features
        from src.preprocessing.loader import load_test

        ti = context["ti"]
        best_run_id = ti.xcom_pull(task_ids="retrain_model", key="best_run_id")

        df_raw = load_test()
        df_clean = clean(df_raw)
        df = engineer_features(df_clean)
        X_test = df.drop(columns=[TARGET])
        y_test = df[TARGET].astype(int)

        def _load(uri):
            try:
                return mlflow.sklearn.load_model(uri)
            except Exception:
                return mlflow.xgboost.load_model(uri)

        new_model = _load(f"runs:/{best_run_id}/model")
        y_pred_new = (new_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        new_f1 = float(f1_score(y_test, y_pred_new))

        try:
            prod_model = _load(f"models:/{MODEL_NAME}/Production")
            y_pred_prod = (prod_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
            prod_f1 = float(f1_score(y_test, y_pred_prod))
        except Exception:
            prod_f1 = 0.0

        print(f"Nouveau modèle F1 : {new_f1:.4f} | Production F1 : {prod_f1:.4f}")
        ti.xcom_push(key="new_f1", value=new_f1)
        ti.xcom_push(key="prod_f1", value=prod_f1)

    def promote_or_rollback(**context):
        from src.training.registry import promote_model

        ti = context["ti"]
        new_f1 = ti.xcom_pull(task_ids="evaluate_model", key="new_f1")
        prod_f1 = ti.xcom_pull(task_ids="evaluate_model", key="prod_f1")
        best_run_id = ti.xcom_pull(task_ids="retrain_model", key="best_run_id")

        if new_f1 > prod_f1:
            promote_model(run_id=best_run_id, stage="Production")
            print(f"Nouveau modèle promu en Production (F1 {prod_f1:.4f} → {new_f1:.4f})")
        else:
            print(
                f"Nouveau modèle inférieur (F1 {new_f1:.4f} ≤ {prod_f1:.4f})"
                " — modèle en Production inchangé"
            )

    def no_retrain(**_):
        print("Pas de dérive détectée — réentraînement non nécessaire")

    branch = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    retrain = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    promote = PythonOperator(task_id="promote_or_rollback", python_callable=promote_or_rollback)
    skip = PythonOperator(task_id="no_retrain", python_callable=no_retrain)

    branch >> [retrain, skip]
    retrain >> evaluate >> promote
