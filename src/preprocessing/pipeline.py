from pathlib import Path

import pandas as pd

from src.preprocessing.cleaner import clean
from src.preprocessing.features import engineer_features
from src.preprocessing.loader import load_test, load_train

OUTPUT_TRAIN = Path(__file__).parents[2] / "data" / "processed" / "features_engineered.csv"
OUTPUT_TEST = Path(__file__).parents[2] / "data" / "processed" / "features_engineered_test.csv"


def run() -> pd.DataFrame:
    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    train = engineer_features(clean(load_train()))
    test  = engineer_features(clean(load_test()))

    train.to_csv(OUTPUT_TRAIN, index=False)
    test.to_csv(OUTPUT_TEST, index=False)

    print(f"Train : {OUTPUT_TRAIN} ({len(train):,} lignes, {train.shape[1]} colonnes)")
    print(f"Test  : {OUTPUT_TEST} ({len(test):,} lignes, {test.shape[1]} colonnes)")
    print(f"Taux de churn — train : {train['churn_flag'].mean():.2%} | test : {test['churn_flag'].mean():.2%}")
    print(f"Colonnes : {list(train.columns)}")

    return train


if __name__ == "__main__":
    run()
