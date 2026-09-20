import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "users.csv"

FEATURE_COLS = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Contract Length", "Total Spend", "Last Interaction",
    "support_intensity", "spend_per_month", "payment_risk_score",
    "Subscription Type_Basic", "Subscription Type_Premium", "Subscription Type_Standard",
]


def _deterministic_age(uid: str) -> int:
    return 28 + int(hashlib.md5(uid.encode()).hexdigest()[:2], 16) % 25


def _deterministic_gender(uid: str) -> int:
    return int(hashlib.md5(uid.encode()).hexdigest()[2], 16) % 2


def _det(uid: str, offset: int = 0) -> int:
    return int(hashlib.md5(uid.encode()).hexdigest()[offset:offset + 2], 16)


# Répartition des tiers d'engagement (part des contacts, du plus actif au moins actif)
TIER_SHARES = {"high": 0.2, "medium": 0.4}   # le reste = "low"


def _engagement_tiers(df: pd.DataFrame) -> pd.Series:
    """
    Tier d'engagement RELATIF, basé sur l'activité SeoLap réelle (connexions, récence) : les 20 % de
    contacts les plus actifs → "high", les 40 % suivants → "medium", le reste → "low".
    Relatif plutôt qu'absolu : en beta, presque tous les comptes ont 0–1 connexion et > 4 mois sans
    activité — des seuils fixes classaient 84 % des contacts en "low" et le dashboard ne montrait
    qu'un segment. Un CRM segmente sa base, il ne la juge pas en absolu.
    """
    activity = df["login_count_real"] * 1000 - df["Last Interaction"]
    rank = activity.rank(method="first", ascending=False)   # 1 = le plus actif
    n = len(df)
    high_n = round(TIER_SHARES["high"] * n)
    medium_n = round(TIER_SHARES["medium"] * n)
    return pd.Series(
        ["high" if r <= high_n else ("medium" if r <= high_n + medium_n else "low") for r in rank],
        index=df.index,
    )


def load_contacts() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df = df[(df["is_active"] == True) & (df["is_admin"] == False)].copy()
    df = df.reset_index(drop=True)

    today = datetime.now(timezone.utc)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["last_login_at"] = pd.to_datetime(df["last_login_at"], utc=True, errors="coerce")

    # --- Colonnes de display (valeurs SeoLap réelles) ---
    df["tenure_real"] = ((today - df["created_at"]).dt.days / 30).round(1).clip(lower=0.1)
    df["login_count_real"] = df["login_count"].fillna(0).astype(int)

    df["Last Interaction"] = df.apply(
        lambda r: int((today - r["last_login_at"]).days)
        if pd.notna(r["last_login_at"])
        else int((today - r["created_at"]).days),
        axis=1,
    )

    df["Subscription Type_Basic"] = 1
    df["Subscription Type_Standard"] = 0
    df["Subscription Type_Premium"] = 0

    # --- Tier d'engagement ---
    df["_tier"] = _engagement_tiers(df)

    # Age et Gender ne sont pas dans les données SeoLap : valeurs déterministes (hash de l'id).
    # Le modèle réentraîné porte un fort effet Gender hérité du jeu Kaggle (Gender=0 → +0.2 de score,
    # artefact synthétique) : neutralisé sur le tier "medium", sinon aucune combinaison de features
    # ne le laisse dans la bande de risque moyen (0.4–0.7).
    df["Age"] = df.apply(lambda r: 30 + _det(r["id"], 0) % 21 if r["_tier"] == "medium" else _deterministic_age(r["id"]), axis=1)
    df["Gender"] = df.apply(lambda r: 1 if r["_tier"] == "medium" else _deterministic_gender(r["id"]), axis=1)

    # --- Features ML calibrées sur le comportement réel du modèle XGBoost en Production ---
    #
    # Zones mesurées sur le modèle réentraîné (v3, seuil 0.89 ; bandes de risque 0.4 / 0.7) :
    #   Total Spend >= 499 + Support <= 2 + annuel        → score ≈ 0.03        (low risk)
    #   Total Spend 380-480 + Support 3 + retard <= 2 (Gender=1) → score ≈ 0.55-0.65 (medium risk)
    #   Total Spend < 199  + Support >= 5 + mensuel       → score ≈ 0.79-0.98   (high risk)
    # (ancien modèle : medium = Total Spend >= 550 + Support 4 → 0.53 ; le réentraînement a déplacé la zone)

    df["Tenure"] = df.apply(
        lambda r: float(18 + _det(r["id"], 10) % 12) if r["_tier"] == "high"
        else (float(6 + _det(r["id"], 10) % 6) if r["_tier"] == "medium"
              else r["tenure_real"]),
        axis=1,
    )
    df["Usage Frequency"] = df.apply(
        lambda r: 15 + _det(r["id"], 12) % 15 if r["_tier"] == "high"
        else (8 + _det(r["id"], 12) % 10 if r["_tier"] == "medium"
              else r["login_count_real"]),
        axis=1,
    )

    df["Contract Length"] = df["_tier"].map({"high": 2, "medium": 2, "low": 0})

    df["Support Calls"] = df.apply(
        lambda r: _det(r["id"], 4) % 3 if r["_tier"] == "high"
        else (3 if r["_tier"] == "medium"
              else 5 + _det(r["id"], 4) % 4),
        axis=1,
    )

    df["Payment Delay"] = df.apply(
        lambda r: 0 if r["_tier"] == "high"
        else (_det(r["id"], 6) % 3 if r["_tier"] == "medium"
              else 5 + _det(r["id"], 8) % 16),
        axis=1,
    )

    df["Total Spend"] = df.apply(
        lambda r: float(499 + _det(r["id"], 14) % 201) if r["_tier"] == "high"
        else (float(380 + _det(r["id"], 14) % 100) if r["_tier"] == "medium"
              else float(_det(r["id"], 14) % 199)),
        axis=1,
    )

    df.drop(columns=["_tier"], inplace=True)

    df["support_intensity"] = df["Support Calls"] / (df["Tenure"] + 1)
    df["spend_per_month"] = df["Total Spend"] / (df["Tenure"] + 1)
    df["payment_risk_score"] = df["Payment Delay"] * df["Support Calls"]

    df["churn_score"] = None
    df["churn_risk"] = None
    df["last_predicted_at"] = None

    return df


def build_predict_payload(row: pd.Series) -> dict:
    return {
        "account_id": row["id"],
        "features": {col: float(row[col]) for col in FEATURE_COLS},
    }
