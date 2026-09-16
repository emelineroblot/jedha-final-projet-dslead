import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.legacy.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from src.paths import REPORTS_DIR


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: Optional[str] = "churn_flag",
    prediction_column: Optional[str] = None,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """
    Génère le rapport Evidently (JSON + HTML) dans `reports_dir` et une copie horodatée
    dans `reports_dir/history/` pour conserver l'historique des contrôles.

    DataDriftPreset est toujours calculé ; ClassificationPreset seulement si une
    colonne de prédiction est fournie (labels connus → suivi de la performance).
    """
    column_mapping = ColumnMapping(target=target_column, prediction=prediction_column)

    metrics = [DataDriftPreset()]
    if prediction_column is not None:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    reports_dir.mkdir(parents=True, exist_ok=True)
    history_dir = reports_dir / "history"
    history_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report.save_html(str(reports_dir / "drift_report.html"))
    report_dict = report.as_dict()
    report_dict["generated_at"] = datetime.now(timezone.utc).isoformat()
    report_dict["n_reference"] = int(len(reference))
    report_dict["n_current"] = int(len(current))

    for path in (reports_dir / "drift_report.json", history_dir / f"drift_{stamp}.json"):
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

    return report_dict
