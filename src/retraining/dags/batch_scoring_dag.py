from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "churnguard",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="batch_scoring",
    description="Score quotidien de tous les comptes actifs",
    schedule="0 2 * * *",  # chaque nuit à 2h00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=default_args,
) as dag:

    def load_active_accounts(**_):
        # TODO: charger les comptes actifs depuis la base SeoLap ou le CSV
        pass

    def score_accounts(**_):
        # TODO: appeler POST /predict/batch et stocker les résultats
        pass

    def push_to_mautic(**_):
        # TODO: simuler l'envoi des scores vers Mautic
        pass

    load = PythonOperator(task_id="load_active_accounts", python_callable=load_active_accounts)
    score = PythonOperator(task_id="score_accounts", python_callable=score_accounts)
    push = PythonOperator(task_id="push_to_mautic", python_callable=push_to_mautic)

    load >> score >> push
