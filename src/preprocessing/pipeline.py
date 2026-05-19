from pathlib import Path

import pandas as pd

from src.preprocessing.cleaner import (
    clean_accounts,
    clean_churn_events,
    clean_feature_usage,
    clean_subscriptions,
    clean_support_tickets,
)
from src.preprocessing.features import engineer_features
from src.preprocessing.loader import load_all
from src.preprocessing.merger import merge_all

OUTPUT_PATH = Path(__file__).parents[2] / "data" / "processed" / "features_engineered.csv"


def run() -> pd.DataFrame:
    raw = load_all()

    accounts = clean_accounts(raw["accounts"])
    subscriptions = clean_subscriptions(raw["subscriptions"])
    feature_usage = clean_feature_usage(raw["feature_usage"])
    churn_events = clean_churn_events(raw["churn_events"])
    support_tickets = clean_support_tickets(raw["support_tickets"])

    merged = merge_all(accounts, subscriptions, feature_usage, churn_events, support_tickets)
    features = engineer_features(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Features exportées : {OUTPUT_PATH} ({len(features)} lignes, {features.shape[1]} colonnes)")

    return features


if __name__ == "__main__":
    run()
