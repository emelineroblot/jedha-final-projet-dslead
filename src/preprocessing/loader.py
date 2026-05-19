import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"


def load_accounts() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rivalytics_accounts.csv")


def load_subscriptions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rivalytics_subscriptions.csv")


def load_feature_usage() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rivalytics_feature_usage.csv")


def load_churn_events() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rivalytics_churn_events.csv")


def load_support_tickets() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rivalytics_support_tickets.csv")


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "accounts": load_accounts(),
        "subscriptions": load_subscriptions(),
        "feature_usage": load_feature_usage(),
        "churn_events": load_churn_events(),
        "support_tickets": load_support_tickets(),
    }
