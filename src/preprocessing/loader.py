from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer_churn_dataset-training-master.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer_churn_dataset-testing-master.csv")
