import pandas as pd

_CONTRACT_ORDER = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
_SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]

# Ordre canonique des 15 features — identique à l'ordre d'entraînement du modèle
# (get_dummies trie les modalités par ordre alphabétique : Basic, Premium, Standard).
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "support_intensity",
    "spend_per_month",
    "payment_risk_score",
    "Subscription Type_Basic",
    "Subscription Type_Premium",
    "Subscription Type_Standard",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Features dérivées ---
    df["support_intensity"] = df["Support Calls"] / (df["Tenure"] + 1)
    df["spend_per_month"] = df["Total Spend"] / (df["Tenure"] + 1)
    # Payment Delay × Support Calls : score composite de risque
    df["payment_risk_score"] = df["Payment Delay"] * df["Support Calls"]

    # --- Encodage Contract Length : ordinal (Monthly a 100% churn — à surveiller) ---
    df["Contract Length"] = df["Contract Length"].map(_CONTRACT_ORDER)

    # --- One-hot Subscription Type (non-ordinal), colonnes garanties même si une
    # modalité est absente du batch (petit échantillon, données SeoLap) ---
    df = pd.get_dummies(df, columns=["Subscription Type"], drop_first=False, dtype=int)
    for sub in _SUBSCRIPTION_TYPES:
        col = f"Subscription Type_{sub}"
        if col not in df.columns:
            df[col] = 0

    # --- Renommage target pour compatibilité avec train.py ---
    df = df.rename(columns={"Churn": "churn_flag"})
    df["churn_flag"] = df["churn_flag"].astype(int)

    return df[FEATURE_COLUMNS + ["churn_flag"]]
