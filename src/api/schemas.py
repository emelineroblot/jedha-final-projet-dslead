from pydantic import BaseModel, ConfigDict, Field

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
    "Subscription Type_Basic": 0,
    "Subscription Type_Standard": 1,
    "Subscription Type_Premium": 0,
    "support_intensity": 0.12,
    "spend_per_month": 32.0,
    "payment_risk_score": 30.0,
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"account_id": "ACC-001", "features": _EXAMPLE_FEATURES}
        }
    )

    account_id: str = Field(description="Identifiant unique du compte")
    features: dict[str, float] = Field(
        description=(
            "15 features du modèle : Age, Gender (0/1), Tenure, Usage Frequency, "
            "Support Calls, Payment Delay, Contract Length (0=Monthly/1=Quarterly/2=Annual), "
            "Total Spend, Last Interaction, Subscription Type_Basic/Standard/Premium (one-hot), "
            "support_intensity, spend_per_month, payment_risk_score"
        )
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": "ACC-001",
                "churn_score": 0.1823,
                "churn_risk": "low",
            }
        }
    )

    account_id: str
    churn_score: float = Field(description="Probabilité de churn entre 0 et 1")
    churn_risk: str = Field(description="Niveau de risque : low (<0.4) | medium (0.4–0.7) | high (>0.7)")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "accounts": [
                    {"account_id": "ACC-001", "features": _EXAMPLE_FEATURES},
                    {"account_id": "ACC-002", "features": _EXAMPLE_FEATURES},
                ]
            }
        }
    )

    accounts: list[PredictRequest] = Field(description="Liste de comptes à scorer")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "churnguard-model",
                "version": "3",
                "stage": "Production",
                "metrics": {"f1_score": 0.999, "auc_roc": 1.0},
                "trained_at": "2025-05-18T14:32:00+00:00",
                "feature_count": 15,
            }
        }
    )

    model_name: str
    version: str
    stage: str
    metrics: dict[str, float] = Field(description="Métriques enregistrées dans MLflow au moment de l'entraînement")
    trained_at: str = Field(description="Date d'enregistrement du modèle (ISO 8601 UTC)")
    feature_count: int = Field(description="Nombre de features attendues par le modèle")
