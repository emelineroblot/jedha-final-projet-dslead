from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

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

    def check_drift(**_) -> str:
        from src.monitoring.alert import check_drift as _check
        drift = _check()
        return "retrain_model" if drift else "no_retrain"

    def retrain_model(**_):
        from src.training.train import train
        train()

    def evaluate_model(**_):
        # TODO: comparer F1 nouveau modèle vs modèle en Production
        pass

    def promote_or_rollback(**_):
        # TODO: promouvoir si F1_nouveau > F1_prod, sinon rollback
        pass

    def no_retrain(**_):
        print("Pas de dérive détectée — réentraînement non nécessaire")

    branch = BranchPythonOperator(task_id="check_drift", python_callable=check_drift)
    retrain = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    promote = PythonOperator(task_id="promote_or_rollback", python_callable=promote_or_rollback)
    skip = PythonOperator(task_id="no_retrain", python_callable=no_retrain)

    branch >> [retrain, skip]
    retrain >> evaluate >> promote
