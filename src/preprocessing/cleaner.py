import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Supprime la ligne malformée (1 null par colonne dans le CSV source)
    df = df.dropna(subset=["Churn"])

    df = df.drop(columns=["CustomerID"])

    df["Gender"] = (df["Gender"].str.lower() == "male").astype(int)

    return df
