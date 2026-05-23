import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

API_URL = os.getenv("CHURNGUARD_API_URL", "http://api:8000")
DATA_PATH = Path(__file__).parents[3] / "data" / "processed" / "features_engineered_test.csv"
ACCOUNTS_TMP = Path("/tmp/churnguard_active_accounts.csv")
SCORES_TMP = Path("/tmp/churnguard_scores.csv")
BATCH_SIZE = 100
SAMPLE_SIZE = 500  # limité pour la démo — retirer en production

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
        import pandas as pd

        df = pd.read_csv(DATA_PATH)
        df = df.drop(columns=["churn_flag"], errors="ignore")
        df = df.head(SAMPLE_SIZE)
        df.to_csv(ACCOUNTS_TMP, index=False)
        print(f"{len(df)} comptes actifs chargés depuis {DATA_PATH.name}")

    def score_accounts(**_):
        import pandas as pd
        import requests

        df = pd.read_csv(ACCOUNTS_TMP)
        feature_cols = df.columns.tolist()

        results = []
        for i in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[i : i + BATCH_SIZE]
            accounts = [
                {
                    "account_id": str(idx),
                    "features": {col: float(row[col]) for col in feature_cols},
                }
                for idx, row in chunk.iterrows()
            ]
            resp = requests.post(
                f"{API_URL}/predict/batch",
                json={"accounts": accounts},
                timeout=30,
            )
            resp.raise_for_status()
            results.extend(resp.json()["results"])

        df_scores = pd.DataFrame(results)
        df_scores.to_csv(SCORES_TMP, index=False)

        high_risk = (df_scores["churn_risk"] == "high").sum()
        print(f"Scoring terminé : {len(df_scores)} comptes | {high_risk} haut risque (churn_risk=high)")

    def push_to_mautic(**_):
        import pandas as pd

        df = pd.read_csv(SCORES_TMP)
        high_risk = df[df["churn_risk"] == "high"]
        medium_risk = df[df["churn_risk"] == "medium"]

        print(f"[Mautic simulé] Segment 'Risque élevé'  : {len(high_risk)} contacts")
        print(f"[Mautic simulé] Segment 'Risque moyen'  : {len(medium_risk)} contacts")

        top5 = high_risk.nlargest(5, "churn_score")
        print("Top 5 comptes à risque élevé :")
        for _, row in top5.iterrows():
            print(f"  → account_id={row['account_id']} | score={row['churn_score']:.4f}")

    load = PythonOperator(task_id="load_active_accounts", python_callable=load_active_accounts)
    score = PythonOperator(task_id="score_accounts", python_callable=score_accounts)
    push = PythonOperator(task_id="push_to_mautic", python_callable=push_to_mautic)

    load >> score >> push
