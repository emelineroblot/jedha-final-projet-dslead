"""
Injecte du bruit artificiel dans les données de test pour simuler une dérive.
Utilisé lors de la démonstration jury.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parents[3] / "data" / "processed" / "features_engineered.csv"
OUTPUT_PATH = Path(__file__).parents[3] / "data" / "processed" / "features_drifted.csv"


def simulate_drift(noise_factor: float = 0.3, seed: int = 42) -> None:
    df = pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(seed)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "churn_flag"]

    for col in numeric_cols:
        std = df[col].std()
        df[col] = df[col] + rng.normal(0, std * noise_factor, size=len(df))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Données dérivées générées : {OUTPUT_PATH} (noise_factor={noise_factor})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.3)
    args = parser.parse_args()
    simulate_drift(noise_factor=args.noise)
