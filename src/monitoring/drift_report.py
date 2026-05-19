from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.report import Report

REPORTS_DIR = Path(__file__).parents[2] / "reports"


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str = "churn_flag",
    prediction_column: str = "prediction",
) -> dict:
    column_mapping = ColumnMapping(
        target=target_column,
        prediction=prediction_column,
    )

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    REPORTS_DIR.mkdir(exist_ok=True)
    report.save_html(str(REPORTS_DIR / "drift_report.html"))

    report_dict = report.as_dict()

    import json
    with open(REPORTS_DIR / "drift_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    return report_dict
