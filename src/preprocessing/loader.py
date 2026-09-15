import pandas as pd

from src.paths import RAW_TEST_PATH, RAW_TRAIN_PATH


def load_train() -> pd.DataFrame:
    return pd.read_csv(RAW_TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return pd.read_csv(RAW_TEST_PATH)
