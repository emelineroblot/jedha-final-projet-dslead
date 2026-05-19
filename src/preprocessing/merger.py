import pandas as pd


def merge_all(
    accounts: pd.DataFrame,
    subscriptions: pd.DataFrame,
    feature_usage: pd.DataFrame,
    churn_events: pd.DataFrame,
    support_tickets: pd.DataFrame,
) -> pd.DataFrame:
    # Base : accounts (une ligne par account)
    df = accounts.copy()

    # Agréger subscriptions par account_id (dernière souscription active)
    sub_agg = (
        subscriptions.sort_values("start_date")
        .groupby("account_id")
        .last()
        .reset_index()
    )
    df = df.merge(sub_agg, on="account_id", how="left", suffixes=("", "_sub"))

    # Agréger feature_usage par account_id
    usage_agg = feature_usage.groupby("account_id").agg("sum").reset_index()
    df = df.merge(usage_agg, on="account_id", how="left")

    # Agréger churn_events par account_id
    churn_agg = (
        churn_events.groupby("account_id")
        .agg(churn_event_count=("account_id", "count"))
        .reset_index()
    )
    df = df.merge(churn_agg, on="account_id", how="left")
    df["churn_event_count"] = df["churn_event_count"].fillna(0)

    # Agréger support_tickets par account_id
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
