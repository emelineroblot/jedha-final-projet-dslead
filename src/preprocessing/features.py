import pandas as pd


_PLAN_TIER_ORDER = {"basic": 0, "pro": 1, "enterprise": 2}

_DROP_COLS = [
    # Identifiants
    "account_id", "account_name",
    # Remplacé par tenure_days
    "signup_date",
    # Redondant avec mrr_amount * 12
    "arr_amount",
    # Remplacés par mrr_delta + mrr_growth_rate (mrr_amount courant déjà présent)
    "mrr_first", "mrr_last",
    # Data leakage : 339 comptes ont des events, seulement 110 ont churn_flag=True
    "churn_event_count",
    "churn_reason_pricing", "churn_reason_support", "churn_reason_budget",
    "churn_reason_features", "churn_reason_competitor",
    # Trop de modalités, aucun signal EDA
    "country",
]

_SUPPORT_COLS = [
    "ticket_count", "escalation_count", "avg_satisfaction_score",
    "avg_resolution_time", "avg_first_response_time",
]

_BOOL_COLS = ["auto_renew_flag", "is_active", "has_upgraded", "has_downgraded", "is_trial"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Temporel ---
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    ref_date = df["signup_date"].max()  # date fixe = reproductible sur le même dataset
    df["tenure_days"] = (ref_date - df["signup_date"]).dt.days

    # --- Usage ---
    df["error_rate"] = df["total_error_count"] / (df["total_sessions"] + 1)
    df["avg_session_duration"] = df["total_duration_secs"] / (df["total_sessions"] + 1)
    df["beta_feature_ratio"] = df["beta_sessions"] / (df["usage_rows"] + 1)
    df["sessions_per_seat"] = df["total_sessions"] / (df["seats"] + 1)

    # --- Financier ---
    df["mrr_delta"] = df["mrr_last"] - df["mrr_first"]
    df["mrr_growth_rate"] = df["mrr_delta"] / (df["mrr_first"] + 1)

    # --- Support (8 comptes sans ticket → imputation 0) ---
    df[_SUPPORT_COLS] = df[_SUPPORT_COLS].fillna(0)
    df["escalation_rate"] = df["escalation_count"] / (df["ticket_count"] + 1)
    df["tickets_per_seat"] = df["ticket_count"] / (df["seats"] + 1)

    # --- Encodage ordinal ---
    df["plan_tier"] = df["plan_tier"].str.lower().map(_PLAN_TIER_ORDER)
    df["billing_frequency"] = (df["billing_frequency"].str.lower() == "annual").astype(int)

    # --- Booléens → int ---
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- One-hot ---
    df = pd.get_dummies(df, columns=["industry", "referral_source"], drop_first=False, dtype=int)

    # --- Suppression des colonnes inutiles / leakage ---
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    return df
