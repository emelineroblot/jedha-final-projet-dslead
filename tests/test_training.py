from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features import FEATURE_COLUMNS
from src.training.evaluate import compute_metrics, feature_importance, find_optimal_threshold
from src.training.train import _dvc_md5, load_training_data

FIXTURE = Path(__file__).parent / "fixtures" / "sample_test.csv"


def test_find_optimal_threshold_maximizes_f1():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    t = find_optimal_threshold(y, proba)
    assert 0.3 < t <= 0.35
    assert compute_metrics(y, proba, t)["f1_score"] == 1.0


def test_compute_metrics_keys_and_ranges():
    y = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([0.2, 0.9, 0.4, 0.6, 0.3, 0.1])
    m = compute_metrics(y, proba, 0.5)
    assert set(m) == {"f1_score", "precision", "recall", "auc_roc", "auc_pr", "decision_threshold"}
    assert all(0.0 <= m[k] <= 1.0 for k in m)
    assert m["recall"] == pytest.approx(2 / 3)


def test_load_training_data_concatenates_extra_window(tmp_path):
    base = pd.read_csv(FIXTURE)
    ref, extra = base.iloc[:300], base.iloc[300:]
    ref_path, extra_path = tmp_path / "ref.csv", tmp_path / "extra.csv"
    ref.to_csv(ref_path, index=False)
    extra.assign(unexpected_col=1).to_csv(extra_path, index=False)  # colonne parasite ignorée

    df = load_training_data(ref_path, extra_path)
    assert len(df) == 500
    assert list(df.columns) == FEATURE_COLUMNS + ["churn_flag"]


def test_load_training_data_without_extra():
    df = load_training_data(FIXTURE, None)
    assert len(df) == 500


def test_dvc_md5_reads_pointer(tmp_path):
    data = tmp_path / "x.csv"
    data.write_text("a\n1\n")
    assert _dvc_md5(data) == "untracked"
    (tmp_path / "x.csv.dvc").write_text("outs:\n- md5: abc123\n  path: x.csv\n")
    assert _dvc_md5(data) == "abc123"


def test_feature_importance_sums_to_one():
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv(FIXTURE)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(df[FEATURE_COLUMNS], df["churn_flag"])
    fi = feature_importance(model, FEATURE_COLUMNS)
    assert fi["importance"].sum() == pytest.approx(1.0)
    assert fi.iloc[0]["importance"] >= fi.iloc[-1]["importance"]


def test_fixture_sample_is_valid():
    df = pd.read_csv(FIXTURE)
    assert len(df) == 500
    assert set(FEATURE_COLUMNS + ["churn_flag"]) <= set(df.columns)
    assert 0.3 < df["churn_flag"].mean() < 0.7
