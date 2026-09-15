from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Exemple cohérent avec l'ordre des features du modèle
_EXAMPLE_FEATURES = {
    "Age": 35,
    "Gender": 1,
    "Tenure": 24,
    "Usage Frequency": 15,
    "Support Calls": 3,
    "Payment Delay": 10,
    "Contract Length": 1,
    "Total Spend": 800.0,
    "Last Interaction": 7,
    "support_intensity": 0.12,
    "spend_per_month": 32.0,
    "payment_risk_score": 30.0,
    "Subscription Type_Basic": 0,
    "Subscription Type_Premium": 0,
    "Subscription Type_Standard": 1,
}


class Features(BaseModel):
    """
    Les 15 features attendues par le modèle. Les noms avec espaces sont acceptés tels quels
    (alias) ou en snake_case (`usage_frequency`). Une feature manquante ou hors bornes → 422.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid", json_schema_extra={"example": _EXAMPLE_FEATURES})

    age: float = Field(alias="Age", ge=0, le=120, description="Âge du client")
    gender: int = Field(alias="Gender", ge=0, le=1, description="1 = homme, 0 = femme")
    tenure: float = Field(alias="Tenure", ge=0, description="Ancienneté en mois")
    usage_frequency: float = Field(alias="Usage Frequency", ge=0, description="Sessions par mois")
    support_calls: float = Field(alias="Support Calls", ge=0, description="Appels au support sur la période")
    payment_delay: float = Field(alias="Payment Delay", ge=0, description="Jours de retard de paiement")
    contract_length: int = Field(alias="Contract Length", ge=0, le=2, description="0 = Monthly, 1 = Quarterly, 2 = Annual")
    total_spend: float = Field(alias="Total Spend", ge=0, description="Dépense totale")
    last_interaction: float = Field(alias="Last Interaction", ge=0, description="Jours depuis la dernière interaction")
    support_intensity: float = Field(ge=0, description="Support Calls / (Tenure + 1)")
    spend_per_month: float = Field(ge=0, description="Total Spend / (Tenure + 1)")
    payment_risk_score: float = Field(ge=0, description="Payment Delay × Support Calls")
    subscription_basic: int = Field(alias="Subscription Type_Basic", ge=0, le=1, description="1 si offre Basic")
    subscription_premium: int = Field(alias="Subscription Type_Premium", ge=0, le=1, description="1 si offre Premium")
    subscription_standard: int = Field(alias="Subscription Type_Standard", ge=0, le=1, description="1 si offre Standard")

    def as_row(self) -> dict:
        """Dictionnaire avec les noms de colonnes du modèle (alias)."""
        return self.model_dump(by_alias=True)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"account_id": "ACC-001", "features": _EXAMPLE_FEATURES}})

    account_id: str = Field(description="Identifiant unique du compte", min_length=1)
    features: Features


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": "ACC-001",
                "churn_score": 0.1823,
                "churn_risk": "low",
                "churn_predicted": True,
                "model_version": "4",
            }
        }
    )

    account_id: str
    churn_score: float = Field(description="Probabilité de churn entre 0 et 1")
    churn_risk: str = Field(description="Bande métier : low (<0.4) | medium (0.4–0.7) | high (≥0.7)")
    churn_predicted: bool = Field(description="Score ≥ seuil de décision du modèle (optimisé sur validation)")
    model_version: Optional[str] = Field(default=None, description="Version du modèle dans le registry MLflow")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "accounts": [
                    {"account_id": "ACC-001", "features": _EXAMPLE_FEATURES},
                    {"account_id": "ACC-002", "features": {**_EXAMPLE_FEATURES, "Support Calls": 9, "Payment Delay": 25}},
                ]
            }
        }
    )

    accounts: list[PredictRequest] = Field(description="Liste de comptes à scorer", min_length=1, max_length=5000)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int
    latency_ms: float = Field(description="Temps de scoring côté serveur")


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "churnguard-model",
                "version": "4",
                "stage": "Production",
                "metrics": {"f1_score": 0.999, "auc_roc": 1.0, "precision": 0.999, "recall": 0.999},
                "decision_threshold": 0.15,
                "trained_at": "2026-05-23T06:42:24+00:00",
                "loaded_at": "2026-09-15T09:00:00+00:00",
                "feature_count": 15,
                "features": list(_EXAMPLE_FEATURES.keys()),
                "source": "mlflow",
            }
        }
    )

    model_name: str
    version: str
    stage: str
    metrics: dict[str, float] = Field(description="Métriques enregistrées dans MLflow au moment de l'entraînement")
    decision_threshold: float = Field(description="Seuil de décision (score ≥ seuil → churn_predicted)")
    trained_at: str = Field(description="Date d'enregistrement du modèle (ISO 8601 UTC)")
    loaded_at: str = Field(description="Date de chargement par l'API (ISO 8601 UTC)")
    feature_count: int
    features: list[str] = Field(description="Features attendues, dans l'ordre du modèle")
    source: str = Field(description="mlflow (registry) ou file (artefact standalone)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ReloadResponse(BaseModel):
    reloaded: bool
    previous_version: Optional[str]
    version: Optional[str]
    detail: str
