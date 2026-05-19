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
    # Placeholder — à compléter en phase 2
    pass
