from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

REPORTS_DIR = Path(__file__).parents[2] / "reports"


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    report["auc_roc"] = roc_auc_score(y_test, y_proba)

    return report
