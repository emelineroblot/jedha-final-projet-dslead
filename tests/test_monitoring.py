from unittest.mock import patch

import pytest

from src.monitoring import notify
from src.monitoring.alert import DRIFT_THRESHOLD, _evaluate_report


def _report(drift_share: float, f1_ref=None, f1_cur=None) -> dict:
    metrics = [
        {
            "metric": "DatasetDriftMetric",
            "result": {
                "drift_share": 0.5,  # seuil Evidently, PAS la part observée
                "number_of_columns": 15,
                "number_of_drifted_columns": int(round(drift_share * 15)),
                "share_of_drifted_columns": drift_share,
                "dataset_drift": drift_share >= 0.5,
            },
        },
        {"metric": "ColumnDriftMetric", "result": {"column_name": "Age", "drift_detected": drift_share > 0.5}},
    ]
    if f1_ref is not None:
        metrics.append({
            "metric": "ClassificationQualityMetric",
            "result": {"current": {"f1": f1_cur}, "reference": {"f1": f1_ref}},
        })
    return {"metrics": metrics}


def test_no_drift_below_threshold():
    result = _evaluate_report(_report(0.1))
    assert result.drifted is False
    assert result.drift_share == 0.1
    assert "pas de dérive" in result.reason


def test_drift_above_threshold():
    result = _evaluate_report(_report(0.6), source="features_drifted.csv", n_current=1000)
    assert result.drifted is True
    assert result.n_drifted_features == 9
    assert result.current_source == "features_drifted.csv"
    assert "60%" in result.reason


def test_threshold_is_strict():
    assert _evaluate_report(_report(DRIFT_THRESHOLD)).drifted is False


def test_evidently_drift_share_key_is_not_the_observed_value():
    """Régression : `drift_share` = seuil Evidently (0.5) — ne doit jamais déclencher seul."""
    report = {"metrics": [{"metric": "DatasetDriftMetric", "result": {
        "drift_share": 0.5, "number_of_columns": 15, "number_of_drifted_columns": 1, "share_of_drifted_columns": 1 / 15,
    }}]}
    result = _evaluate_report(report)
    assert result.drifted is False
    assert result.drift_share == pytest.approx(1 / 15)


def test_f1_drop_triggers_alert():
    result = _evaluate_report(_report(0.05, f1_ref=0.99, f1_cur=0.90))
    assert result.drifted is True
    assert result.f1_reference == 0.99
    assert "F1" in result.reason


def test_small_f1_drop_is_tolerated():
    assert _evaluate_report(_report(0.05, f1_ref=0.99, f1_cur=0.96)).drifted is False


def test_legacy_evidently_keys_are_supported():
    report = {"metrics": [{
        "metric": "ClassificationQualityMetric",
        "result": {"reference_metrics": {"f1": 0.95}, "current_metrics": {"f1": 0.80}},
    }]}
    assert _evaluate_report(report).drifted is True


def test_send_alert_without_webhook_logs_only():
    with patch.object(notify, "WEBHOOK_URL", None):
        assert notify.send_alert("test", "message") is False


def test_send_alert_posts_to_webhook():
    with patch.object(notify, "WEBHOOK_URL", "https://hooks.example/abc"), patch("requests.post") as post:
        post.return_value.raise_for_status.return_value = None
        assert notify.send_alert("Dérive détectée", "60% des features", level="warning") is True
        payload = post.call_args.kwargs["json"]
        assert "Dérive détectée" in payload["content"]
        assert payload["text"] == payload["content"]


def test_store_is_noop_without_database_url():
    from src.monitoring import store

    with patch.object(store, "DATABASE_URL", None):
        assert store.is_enabled() is False
        store.init_db()
        store.log_predictions([{"account_id": "x"}])
        assert store.fetch_recent_features() is None
        assert store.count_since(None) == 0
