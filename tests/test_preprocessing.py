import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_churn_events, clean_subscriptions
from src.preprocessing.loader import load_all


def test_load_all_tables():
    tables = load_all()
    assert set(tables.keys()) == {"accounts", "subscriptions", "feature_usage", "churn_events", "support_tickets"}
    for name, df in tables.items():
        assert len(df) > 0, f"Table {name} vide"


def test_clean_churn_events_excludes_reactivations():
    raw = load_all()["churn_events"]
    cleaned = clean_churn_events(raw)
    assert cleaned["is_reactivation"].sum() == 0


def test_clean_subscriptions_flags():
    raw = load_all()["subscriptions"]
    cleaned = clean_subscriptions(raw)
    assert "is_active" in cleaned.columns
    assert "is_trial" in cleaned.columns


def test_feature_engineering():
    from src.preprocessing.cleaner import (
        clean_accounts, clean_churn_events, clean_feature_usage,
        clean_subscriptions, clean_support_tickets,
    )
    from src.preprocessing.merger import merge_all
    from src.preprocessing.features import engineer_features

    raw = load_all()
    merged = merge_all(
        clean_accounts(raw["accounts"]),
        clean_subscriptions(raw["subscriptions"]),
        clean_feature_usage(raw["feature_usage"]),
        clean_churn_events(raw["churn_events"]),
        clean_support_tickets(raw["support_tickets"]),
    )
    features = engineer_features(merged)

    # Toutes les features dérivées sont présentes
    expected = [
        "tenure_days", "error_rate", "avg_session_duration",
        "beta_feature_ratio", "sessions_per_seat",
        "mrr_delta", "mrr_growth_rate",
        "escalation_rate", "tickets_per_seat",
    ]
    for col in expected:
        assert col in features.columns, f"Feature manquante : {col}"

    # Colonnes leakage/identifiants absentes
    forbidden = [
        "account_id", "account_name", "signup_date", "arr_amount", "country",
        "churn_event_count", "churn_reason_pricing",
    ]
    for col in forbidden:
        assert col not in features.columns, f"Colonne interdite présente : {col}"

    # Pas de NaN sur les colonnes numériques dérivées
    assert features[expected].isnull().sum().sum() == 0

    # 500 lignes conservées (une par compte)
    assert len(features) == 500
