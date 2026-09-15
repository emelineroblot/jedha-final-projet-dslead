from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import ModelState, app
from src.preprocessing.features import FEATURE_COLUMNS

SAMPLE_FEATURES = {
    "Age": 35,
    "Gender": 1,
    "Tenure": 24,
    "Usage Frequency": 15,
    "Support Calls": 2,
    "Payment Delay": 5,
    "Contract Length": 1,
    "Total Spend": 500.0,
    "Last Interaction": 10,
    "support_intensity": 0.08,
    "spend_per_month": 20.0,
    "payment_risk_score": 10.0,
    "Subscription Type_Basic": 1,
    "Subscription Type_Premium": 0,
    "Subscription Type_Standard": 0,
}


def _fake_state(version: str = "3", threshold: float = 0.15) -> ModelState:
    model = MagicMock()
    # une ligne de proba par ligne d'entrée
    model.predict_proba.side_effect = lambda X: np.tile([0.2, 0.8], (len(X), 1))
    model.feature_names_in_ = np.array(FEATURE_COLUMNS)
    model.n_features_in_ = 15
    return ModelState(
        model=model,
        version=version,
        decision_threshold=threshold,
        trained_at="2026-05-23T06:42:24+00:00",
        metrics={"f1_score": 0.999, "auc_roc": 1.0},
        source="mlflow",
    )


@pytest.fixture()
def client():
    with patch("src.api.main.load_model_state", return_value=_fake_state()):
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "3"


def test_predict_endpoint(client):
    resp = client.post("/predict", json={"account_id": "acc_001", "features": SAMPLE_FEATURES})
    assert resp.status_code == 200
    data = resp.json()
    assert data["account_id"] == "acc_001"
    assert 0.0 <= data["churn_score"] <= 1.0
    assert data["churn_risk"] == "high"
    assert data["churn_predicted"] is True  # 0.8 ≥ seuil 0.15
    assert data["model_version"] == "3"
    assert "X-Process-Time-Ms" in resp.headers


def test_predict_accepts_snake_case_aliases(client):
    features = {
        "age": 35, "gender": 1, "tenure": 24, "usage_frequency": 15, "support_calls": 2,
        "payment_delay": 5, "contract_length": 1, "total_spend": 500.0, "last_interaction": 10,
        "support_intensity": 0.08, "spend_per_month": 20.0, "payment_risk_score": 10.0,
        "subscription_basic": 1, "subscription_premium": 0, "subscription_standard": 0,
    }
    resp = client.post("/predict", json={"account_id": "acc_002", "features": features})
    assert resp.status_code == 200


def test_predict_rejects_missing_feature(client):
    features = {k: v for k, v in SAMPLE_FEATURES.items() if k != "Support Calls"}
    resp = client.post("/predict", json={"account_id": "acc_001", "features": features})
    assert resp.status_code == 422
    assert "Support Calls" in resp.text or "support_calls" in resp.text


def test_predict_rejects_unknown_feature(client):
    resp = client.post("/predict", json={"account_id": "acc_001", "features": {**SAMPLE_FEATURES, "Foo": 1}})
    assert resp.status_code == 422


def test_predict_rejects_out_of_range(client):
    resp = client.post("/predict", json={"account_id": "acc_001", "features": {**SAMPLE_FEATURES, "Gender": 2}})
    assert resp.status_code == 422


def test_predict_batch_endpoint(client):
    payload = {
        "accounts": [
            {"account_id": "acc_001", "features": SAMPLE_FEATURES},
            {"account_id": "acc_002", "features": SAMPLE_FEATURES},
        ]
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["account_id"] == "acc_001"
    assert data["results"][1]["account_id"] == "acc_002"
    assert data["latency_ms"] >= 0


def test_predict_batch_is_vectorized(client):
    """Un seul appel predict_proba pour tout le batch."""
    from src.api import main as api_main

    model = api_main._state.model
    model.predict_proba.reset_mock()
    payload = {"accounts": [{"account_id": f"acc_{i}", "features": SAMPLE_FEATURES} for i in range(50)]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    assert model.predict_proba.call_count == 1


def test_model_info_endpoint(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "churnguard-model"
    assert data["version"] == "3"
    assert data["stage"] == "Production"
    assert data["metrics"]["f1_score"] == pytest.approx(0.999)
    assert data["decision_threshold"] == pytest.approx(0.15)
    assert data["feature_count"] == 15
    assert data["features"] == FEATURE_COLUMNS


def test_model_reload_switches_version(client):
    with patch("src.api.main.load_model_state", return_value=_fake_state(version="4")):
        resp = client.post("/model/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["reloaded"] is True
    assert data["previous_version"] == "3"
    assert data["version"] == "4"
    assert client.get("/health").json()["model_version"] == "4"


def test_model_reload_keeps_current_on_failure(client):
    with patch("src.api.main.load_model_state", side_effect=RuntimeError("mlflow down")):
        resp = client.post("/model/reload")
    assert resp.status_code == 200
    assert resp.json()["reloaded"] is False
    assert client.get("/health").json()["model_loaded"] is True


def test_metrics_endpoint_exposes_prometheus(client):
    client.post("/predict", json={"account_id": "acc_001", "features": SAMPLE_FEATURES})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "churnguard_prediction_latency_ms" in resp.text
    assert "churnguard_predictions_total" in resp.text


def test_predict_returns_503_without_model():
    with patch("src.api.main.load_model_state", side_effect=RuntimeError("no model")):
        with TestClient(app) as c:
            assert c.get("/health").json()["model_loaded"] is False
            resp = c.post("/predict", json={"account_id": "acc_001", "features": SAMPLE_FEATURES})
            assert resp.status_code == 503
