from pydantic import BaseModel


class PredictRequest(BaseModel):
    account_id: str
    features: dict[str, float]


class PredictResponse(BaseModel):
    account_id: str
    churn_score: float
    churn_risk: str  # "low" | "medium" | "high"


class BatchPredictRequest(BaseModel):
    accounts: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    metrics: dict[str, float]
    trained_at: str
    feature_count: int
