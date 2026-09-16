import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import clean
from src.preprocessing.features import FEATURE_COLUMNS, engineer_features


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Échantillon au format brut Kaggle (muhammadshahidazeem), avec une ligne malformée."""
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4],
        "Age": [30.0, 45.0, 28.0, np.nan],
        "Gender": ["Female", "Male", "male", np.nan],
        "Tenure": [12.0, 3.0, 0.0, np.nan],
        "Usage Frequency": [10.0, 25.0, 2.0, np.nan],
        "Support Calls": [1.0, 6.0, 0.0, np.nan],
        "Payment Delay": [0.0, 20.0, 5.0, np.nan],
        "Subscription Type": ["Basic", "Premium", "Premium", np.nan],
        "Contract Length": ["Annual", "Monthly", "Quarterly", np.nan],
        "Total Spend": [600.0, 150.0, 90.0, np.nan],
        "Last Interaction": [5.0, 28.0, 1.0, np.nan],
        "Churn": [0.0, 1.0, 1.0, np.nan],
    })


def test_clean_drops_malformed_row_and_id(raw_df):
    df = clean(raw_df)
    assert len(df) == 3
    assert "CustomerID" not in df.columns
    assert df.isna().sum().sum() == 0


def test_clean_encodes_gender_case_insensitive(raw_df):
    df = clean(raw_df)
    assert df["Gender"].tolist() == [0, 1, 1]


def test_engineer_features_columns_and_order(raw_df):
    df = engineer_features(clean(raw_df))
    assert list(df.columns) == FEATURE_COLUMNS + ["churn_flag"]
    assert len(FEATURE_COLUMNS) == 15


def test_engineer_features_derived_values(raw_df):
    df = engineer_features(clean(raw_df))
    row = df.iloc[1]  # Tenure=3, Support Calls=6, Total Spend=150, Payment Delay=20
    assert row["support_intensity"] == pytest.approx(6 / 4)
    assert row["spend_per_month"] == pytest.approx(150 / 4)
    assert row["payment_risk_score"] == pytest.approx(120)


def test_engineer_features_encodings(raw_df):
    df = engineer_features(clean(raw_df))
    assert df["Contract Length"].tolist() == [2, 0, 1]  # Annual, Monthly, Quarterly
    assert df["Subscription Type_Basic"].tolist() == [1, 0, 0]
    assert df["Subscription Type_Premium"].tolist() == [0, 1, 1]
    assert df["Subscription Type_Standard"].tolist() == [0, 0, 0]  # modalité absente → colonne garantie
    assert df["churn_flag"].dtype == int
    assert df["churn_flag"].tolist() == [0, 1, 1]


def test_engineer_features_handles_tenure_zero(raw_df):
    df = engineer_features(clean(raw_df))
    assert np.isfinite(df[["support_intensity", "spend_per_month"]].values).all()
