"""
Stockage des prédictions en base PostgreSQL (base `churnguard`).

Activé uniquement si DATABASE_URL est défini — sinon toutes les fonctions sont des no-op,
ce qui permet de lancer l'API en standalone (HF Space, tests) sans base.

Ce stockage ferme la boucle de monitoring : Evidently compare la distribution
d'entraînement (référence) aux features réellement reçues par l'API (courant),
et le DAG de réentraînement peut compter les nouvelles données arrivées.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
TABLE = "predictions"

_engine = None


def is_enabled() -> bool:
    return bool(DATABASE_URL)


def _get_engine():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    return _engine


def init_db() -> None:
    if not is_enabled():
        return
    from sqlalchemy import text

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id BIGSERIAL PRIMARY KEY,
        predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        account_id TEXT NOT NULL,
        features JSONB NOT NULL,
        churn_score DOUBLE PRECISION NOT NULL,
        churn_risk TEXT NOT NULL,
        model_version TEXT,
        latency_ms DOUBLE PRECISION,
        actual_churn INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE}_predicted_at ON {TABLE} (predicted_at);
    """
    with _get_engine().begin() as conn:
        for stmt in ddl.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    logger.info("Table %s prête", TABLE)


def log_predictions(rows: list[dict]) -> None:
    """rows : [{account_id, features, churn_score, churn_risk, model_version, latency_ms}]"""
    if not is_enabled() or not rows:
        return
    from sqlalchemy import text

    stmt = text(
        f"INSERT INTO {TABLE} (account_id, features, churn_score, churn_risk, model_version, latency_ms) "
        "VALUES (:account_id, CAST(:features AS JSONB), :churn_score, :churn_risk, :model_version, :latency_ms)"
    )
    payload = [{**r, "features": json.dumps(r["features"])} for r in rows]
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt, payload)
    except Exception as exc:  # jamais bloquer une prédiction à cause du stockage
        logger.warning("Stockage des prédictions impossible : %s", exc)


def fetch_recent_features(days: int = 7, min_rows: int = 100) -> Optional[pd.DataFrame]:
    """Features reçues par l'API sur les N derniers jours, ou None si trop peu de lignes."""
    if not is_enabled():
        return None
    from sqlalchemy import text

    since = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        with _get_engine().connect() as conn:
            result = conn.execute(
                text(f"SELECT features, actual_churn FROM {TABLE} WHERE predicted_at >= :since"),
                {"since": since},
            )
            records = result.fetchall()
    except Exception as exc:
        logger.warning("Lecture des prédictions impossible : %s", exc)
        return None

    if len(records) < min_rows:
        return None
    df = pd.DataFrame([r[0] for r in records])
    df["actual_churn"] = [r[1] for r in records]
    return df


def count_since(ts: datetime) -> int:
    """Nombre de prédictions enregistrées depuis `ts` (= nouvelles données disponibles)."""
    if not is_enabled():
        return 0
    from sqlalchemy import text

    try:
        with _get_engine().connect() as conn:
            return int(
                conn.execute(
                    text(f"SELECT COUNT(*) FROM {TABLE} WHERE predicted_at >= :ts"), {"ts": ts}
                ).scalar()
            )
    except Exception as exc:
        logger.warning("Comptage des prédictions impossible : %s", exc)
        return 0
