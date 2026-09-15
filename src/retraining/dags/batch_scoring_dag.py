"""
DAG `batch_scoring` — scoring quotidien (02:00) de tous les comptes actifs via /predict/batch,
puis push des segments de risque vers le CRM (Mautic — simulé en démo).

Les tâches communiquent par fichiers temporaires (XCom limité à ~48 KB — jamais de DataFrame).
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

try:  # Airflow 3.x
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator

API_URL = os.getenv("CHURNGUARD_API_URL", "http://api:8000")
TMP_DIR = Path(os.getenv("CHURNGUARD_TMP", "/tmp"))
ACCOUNTS_TMP = TMP_DIR / "churnguard_active_accounts.csv"
SCORES_TMP = TMP_DIR / "churnguard_scores.csv"
BATCH_SIZE = 500
SAMPLE_SIZE = int(os.getenv("BATCH_SAMPLE_SIZE", "500"))  # limité pour la démo — 0 = tous les comptes

default_args = {
    "owner": "churnguard",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _on_failure(context) -> None:
    from src.monitoring.notify import send_alert

    ti = context.get("task_instance")
    send_alert("Échec du DAG batch_scoring", f"Tâche `{ti.task_id}` : {context.get('exception')}", level="critical")


with DAG(
    dag_id="batch_scoring",
    description="Score quotidien de tous les comptes actifs",
    schedule="0 2 * * *",  # chaque nuit à 2h00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={**default_args, "on_failure_callback": _on_failure},
    tags=["churnguard", "scoring"],
) as dag:

    def load_active_accounts(**_):
        import pandas as pd

        from src.paths import FEATURES_INCOMING_PATH, TARGET
        from src.preprocessing.features import FEATURE_COLUMNS

        # En production SeoLap : requête sur la base applicative. Démo : nouvelles données labellisées.
        df = pd.read_csv(FEATURES_INCOMING_PATH)
        df = df.drop(columns=[TARGET], errors="ignore")[FEATURE_COLUMNS]
        if SAMPLE_SIZE:
            df = df.head(SAMPLE_SIZE)
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(ACCOUNTS_TMP, index=False)
        print(f"{len(df)} comptes actifs chargés depuis {FEATURES_INCOMING_PATH.name}")

    def score_accounts(**context):
        import pandas as pd
        import requests

        df = pd.read_csv(ACCOUNTS_TMP)
        feature_cols = df.columns.tolist()

        results, latencies = [], []
        for i in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[i : i + BATCH_SIZE]
            accounts = [
                {"account_id": str(idx), "features": {col: float(row[col]) for col in feature_cols}}
                for idx, row in chunk.iterrows()
            ]
            resp = requests.post(f"{API_URL}/predict/batch", json={"accounts": accounts}, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            results.extend(body["results"])
            latencies.append(body.get("latency_ms", 0.0))

        df_scores = pd.DataFrame(results)
        df_scores.to_csv(SCORES_TMP, index=False)

        high_risk = int((df_scores["churn_risk"] == "high").sum())
        context["ti"].xcom_push(key="n_scored", value=len(df_scores))
        context["ti"].xcom_push(key="n_high_risk", value=high_risk)
        print(
            f"Scoring terminé : {len(df_scores)} comptes | {high_risk} haut risque | "
            f"latence serveur moyenne par batch : {sum(latencies) / max(len(latencies), 1):.1f} ms"
        )

    def push_to_mautic(**context):
        import pandas as pd

        from src.monitoring.notify import send_alert

        df = pd.read_csv(SCORES_TMP)
        high_risk = df[df["churn_risk"] == "high"]
        medium_risk = df[df["churn_risk"] == "medium"]

        # Intégration réelle hors périmètre : POST /api/segments/{id}/contact/{id}/add sur Mautic
        print(f"[Mautic simulé] Segment 'Risque élevé'  : {len(high_risk)} contacts")
        print(f"[Mautic simulé] Segment 'Risque moyen'  : {len(medium_risk)} contacts")

        top5 = high_risk.nlargest(5, "churn_score")
        print("Top 5 comptes à risque élevé :")
        for _, row in top5.iterrows():
            print(f"  → account_id={row['account_id']} | score={row['churn_score']:.4f}")

        send_alert(
            "Batch scoring terminé",
            f"{len(df)} comptes scorés — {len(high_risk)} à risque élevé, {len(medium_risk)} à risque moyen "
            f"(modèle v{df['model_version'].iloc[0] if 'model_version' in df and len(df) else '?'})",
            level="info",
        )

    load = PythonOperator(task_id="load_active_accounts", python_callable=load_active_accounts)
    score = PythonOperator(task_id="score_accounts", python_callable=score_accounts)
    push = PythonOperator(task_id="push_to_mautic", python_callable=push_to_mautic)

    load >> score >> push
