from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

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
    "Subscription Type_Basic": 1.0,
    "Subscription Type_Standard": 0.0,
    "Subscription Type_Premium": 0.0,
    "support_intensity": 0.08,
    "spend_per_month": 20.0,
    "payment_risk_score": 10.0,
}


@pytest.fixture()
def mock_model():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.2, 0.8]])
    m.n_features_in_ = 15
    return m


@pytest.fixture()
def client(mock_model):
    with patch("src.api.main._load_from_registry", return_value=mock_model):
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    resp = client.post("/predict", json={"account_id": "acc_001", "features": SAMPLE_FEATURES})
    assert resp.status_code == 200
    data = resp.json()
    assert data["account_id"] == "acc_001"
    assert 0.0 <= data["churn_score"] <= 1.0
    assert data["churn_risk"] == "high"


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
    assert len(data["results"]) == 2
    assert data["results"][0]["account_id"] == "acc_001"
    assert data["results"][1]["account_id"] == "acc_002"


def test_model_info_endpoint(client, mock_model):
    mock_version = MagicMock()
    mock_version.version = "3"
    mock_version.run_id = "abc123"
    mock_version.creation_timestamp = 1700000000000

    mock_run = MagicMock()
    mock_run.data.metrics = {"f1_score": 0.999, "auc_roc": 1.0}

    mock_client_instance = MagicMock()
    mock_client_instance.get_latest_versions.return_value = [mock_version]
    mock_client_instance.get_run.return_value = mock_run

    with patch("src.api.main.MlflowClient", return_value=mock_client_instance):
        resp = client.get("/model/info")

    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "churnguard-model"
    assert data["version"] == "3"
    assert data["stage"] == "Production"
    assert data["metrics"]["f1_score"] == pytest.approx(0.999)
    assert data["feature_count"] == 15
