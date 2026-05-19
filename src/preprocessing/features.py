import pandas as pd

_CONTRACT_ORDER = {"Monthly": 0, "Quarterly": 1, "Annual": 2}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Features dérivées ---
    df["support_intensity"] = df["Support Calls"] / (df["Tenure"] + 1)
    df["spend_per_month"] = df["Total Spend"] / (df["Tenure"] + 1)
    # Payment Delay × Support Calls : score composite de risque
    df["payment_risk_score"] = df["Payment Delay"] * df["Support Calls"]

    # --- Encodage Contract Length : ordinal (Monthly a 100% churn — à surveiller) ---
    df["Contract Length"] = df["Contract Length"].map(_CONTRACT_ORDER)

    # --- One-hot Subscription Type (non-ordinal) ---
    df = pd.get_dummies(df, columns=["Subscription Type"], drop_first=False, dtype=int)

    # --- Renommage target pour compatibilité avec train.py ---
    df = df.rename(columns={"Churn": "churn_flag"})
    df["churn_flag"] = df["churn_flag"].astype(int)

    return df
