"""
Injecte du bruit artificiel dans les nouvelles données de production (`features_incoming.csv`)
pour amplifier la dérive. Le fichier produit (`features_drifted.csv`) est repris par `check_drift()`
comme données "courantes" et par le DAG de réentraînement comme nouvelle fenêtre labellisée.

Note : la dérive naturelle entre le train et le fichier test Kaggle suffit déjà à déclencher
le pipeline — ce script sert à la démo (dérive plus visible dans le rapport Evidently).
Supprimer `features_drifted.csv` pour revenir à la dérive naturelle.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import FEATURES_DRIFTED_PATH, FEATURES_INCOMING_PATH, TARGET


def simulate_drift(
    noise_factor: float = 0.3,
    seed: int = 42,
    source: Path = FEATURES_INCOMING_PATH,
    output: Path = FEATURES_DRIFTED_PATH,
) -> Path:
    df = pd.read_csv(source)
    rng = np.random.default_rng(seed)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    for col in numeric_cols:
        std = df[col].std()
        df[col] = df[col] + rng.normal(0, std * noise_factor, size=len(df))

    df.to_csv(output, index=False)
    print(f"Données dérivées générées : {output} ({len(df):,} lignes, noise_factor={noise_factor})")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    simulate_drift(noise_factor=args.noise, seed=args.seed)
