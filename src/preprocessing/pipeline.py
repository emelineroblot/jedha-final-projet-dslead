"""
Pipeline de preprocessing : CSV bruts Kaggle → features engineered.

Produit trois fichiers dans data/processed/ :
- features_engineered.csv       : référence d'entraînement (fichier train Kaggle, 440k lignes)
- features_incoming.csv         : 50 % du fichier test Kaggle — "nouvelles données de production" labellisées
- features_engineered_test.csv  : 50 % restants — hold-out d'évaluation, jamais entraîné

Le fichier test Kaggle ne suit pas la distribution du train (voir docs/dataset-report.md) :
c'est précisément la dérive que le pipeline de monitoring doit détecter et corriger.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import FEATURES_INCOMING_PATH, FEATURES_TEST_PATH, FEATURES_TRAIN_PATH, TARGET
from src.preprocessing.cleaner import clean
from src.preprocessing.features import engineer_features
from src.preprocessing.loader import load_test, load_train

SPLIT_SEED = 42


def run() -> pd.DataFrame:
    FEATURES_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train = engineer_features(clean(load_train()))
    kaggle_test = engineer_features(clean(load_test()))
    incoming, holdout = train_test_split(
        kaggle_test, test_size=0.5, stratify=kaggle_test[TARGET], random_state=SPLIT_SEED
    )

    train.to_csv(FEATURES_TRAIN_PATH, index=False)
    incoming.to_csv(FEATURES_INCOMING_PATH, index=False)
    holdout.to_csv(FEATURES_TEST_PATH, index=False)

    for label, df, path in (
        ("Train (référence)", train, FEATURES_TRAIN_PATH),
        ("Incoming (production)", incoming, FEATURES_INCOMING_PATH),
        ("Hold-out (évaluation)", holdout, FEATURES_TEST_PATH),
    ):
        print(f"{label:<24}: {path.name} — {len(df):,} lignes, churn {df[TARGET].mean():.2%}")
    print(f"Colonnes : {list(train.columns)}")

    return train


if __name__ == "__main__":
    run()
