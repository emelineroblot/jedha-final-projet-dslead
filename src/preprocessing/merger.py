import pandas as pd


def merge_all(
    accounts: pd.DataFrame,
    subscriptions: pd.DataFrame,
    feature_usage: pd.DataFrame,
    churn_events: pd.DataFrame,
    support_tickets: pd.DataFrame,
) -> pd.DataFrame:
    df = accounts.copy()

    # Agrégations subscriptions : état courant (dernière souscription) + historique
    sub_sorted = subscriptions.sort_values("start_date")
    sub_current = (
        sub_sorted.groupby("account_id")
        .last()
        .reset_index()[["account_id", "mrr_amount", "arr_amount", "billing_frequency", "auto_renew_flag", "is_active"]]
    )
    sub_history = (
        subscriptions.groupby("account_id")
        .agg(
            has_upgraded=("upgrade_flag", "any"),
            has_downgraded=("downgrade_flag", "any"),
            subscription_count=("subscription_id", "count"),
            mrr_first=("mrr_amount", "first"),
            mrr_last=("mrr_amount", "last"),
        )
        .reset_index()
    )
    df = df.merge(sub_current, on="account_id", how="left")
    df = df.merge(sub_history, on="account_id", how="left")

    # feature_usage : joint via subscription_id → account_id
    fu_with_account = feature_usage.merge(
        subscriptions[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left",
    )
    usage_agg = (
        fu_with_account.groupby("account_id")
        .agg(
            total_sessions=("usage_count", "sum"),
            total_duration_secs=("usage_duration_secs", "sum"),
            total_error_count=("error_count", "sum"),
            beta_sessions=("is_beta_feature", "sum"),
            usage_rows=("usage_id", "count"),
        )
        .reset_index()
    )
    df = df.merge(usage_agg, on="account_id", how="left")

    # churn_events : déjà filtrés (is_reactivation=False) par cleaner
    churn_agg = (
        churn_events.groupby("account_id")
        .agg(
            churn_event_count=("account_id", "count"),
            churn_reason_pricing=("reason_code", lambda x: (x == "pricing").sum()),
            churn_reason_support=("reason_code", lambda x: (x == "support").sum()),
            churn_reason_budget=("reason_code", lambda x: (x == "budget").sum()),
            churn_reason_features=("reason_code", lambda x: (x == "features").sum()),
            churn_reason_competitor=("reason_code", lambda x: (x == "competitor").sum()),
        )
        .reset_index()
    )
    df = df.merge(churn_agg, on="account_id", how="left")
    churn_cols = ["churn_event_count", "churn_reason_pricing", "churn_reason_support",
                  "churn_reason_budget", "churn_reason_features", "churn_reason_competitor"]
    df[churn_cols] = df[churn_cols].fillna(0)

    # support_tickets
    support_agg = (
        support_tickets.groupby("account_id")
        .agg(
            ticket_count=("account_id", "count"),
            escalation_count=("escalation_flag", "sum"),
            avg_satisfaction_score=("satisfaction_score", "mean"),
            avg_resolution_time=("resolution_time_hours", "mean"),
            avg_first_response_time=("first_response_time_minutes", "mean"),
        )
        .reset_index()
    )
    df = df.merge(support_agg, on="account_id", how="left")

    return df
