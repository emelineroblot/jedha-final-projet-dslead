import pandas as pd


def clean_accounts(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def clean_subscriptions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_active"] = df["end_date"].isna()
    # is_trial déjà présent dans le CSV ; pas de re-calcul
    return df


def clean_churn_events(df: pd.DataFrame) -> pd.DataFrame:
    # Exclure les réactivations pour éviter le double comptage
    return df[df["is_reactivation"] == False].copy()


def clean_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Imputer satisfaction_score manquant par la médiane
    median_score = df["satisfaction_score"].median()
    df["satisfaction_score"] = df["satisfaction_score"].fillna(median_score)
    return df


def clean_feature_usage(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()
