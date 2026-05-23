import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "users.csv"

FEATURE_COLS = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Contract Length", "Total Spend", "Last Interaction",
    "Subscription Type_Basic", "Subscription Type_Standard", "Subscription Type_Premium",
    "support_intensity", "spend_per_month", "payment_risk_score",
]


def _deterministic_age(uid: str) -> int:
    return 28 + int(hashlib.md5(uid.encode()).hexdigest()[:2], 16) % 25


def _deterministic_gender(uid: str) -> int:
    return int(hashlib.md5(uid.encode()).hexdigest()[2], 16) % 2


def load_contacts() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Exclure les admins et les inactifs
    df = df[(df["is_active"] == True) & (df["is_admin"] == False)].copy()
    df = df.reset_index(drop=True)

    today = datetime.now(timezone.utc)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["last_login_at"] = pd.to_datetime(df["last_login_at"], utc=True, errors="coerce")

    # --- Features dérivées des données SeoLap ---
    df["Tenure"] = ((today - df["created_at"]).dt.days / 30).round(1).clip(lower=0.1)
    df["Usage Frequency"] = df["login_count"].fillna(0).astype(int)
    df["Last Interaction"] = df.apply(
        lambda r: int((today - r["last_login_at"]).days)
        if pd.notna(r["last_login_at"])
        else int((today - r["created_at"]).days),
        axis=1,
    )
    df["Contract Length"] = 0  # tous monthly
    df["Subscription Type_Basic"] = 1  # tous starter = Basic
    df["Subscription Type_Standard"] = 0
    df["Subscription Type_Premium"] = 0

    # --- Features non collectées — valeurs déterministes ou neutres ---
    df["Age"] = df["id"].apply(_deterministic_age)
    df["Gender"] = df["id"].apply(_deterministic_gender)
    df["Support Calls"] = 0
    df["Payment Delay"] = 0
    df["Total Spend"] = 0.0

    # --- Features dérivées ---
    df["support_intensity"] = df["Support Calls"] / (df["Tenure"] + 1)
    df["spend_per_month"] = df["Total Spend"] / (df["Tenure"] + 1)
    df["payment_risk_score"] = df["Payment Delay"] * df["Support Calls"]

    # --- Colonnes résultat (vides au départ) ---
    df["churn_score"] = None
    df["churn_risk"] = None
    df["last_predicted_at"] = None

    return df


def build_predict_payload(row: pd.Series) -> dict:
    return {
        "account_id": row["id"],
        "features": {col: float(row[col]) for col in FEATURE_COLS},
    }
