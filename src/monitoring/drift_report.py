import json
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.legacy.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

REPORTS_DIR = Path(__file__).parents[2] / "reports"


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: Optional[str] = "churn_flag",
    prediction_column: Optional[str] = None,
) -> dict:
    column_mapping = ColumnMapping(
        target=target_column,
        prediction=prediction_column,
    )

    metrics = [DataDriftPreset()]
    if prediction_column is not None:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    REPORTS_DIR.mkdir(exist_ok=True)
    report.save_html(str(REPORTS_DIR / "drift_report.html"))

    report_dict = report.as_dict()
    with open(REPORTS_DIR / "drift_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    return report_dict
