import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

REPORTS_DIR = Path(__file__).parents[2] / "reports"
REPORT_PATH = REPORTS_DIR / "drift_report.json"
REFERENCE_PATH = Path(__file__).parents[2] / "data" / "processed" / "features_engineered.csv"
DRIFTED_PATH = Path(__file__).parents[2] / "data" / "processed" / "features_drifted.csv"
CURRENT_PATH = Path(__file__).parents[2] / "data" / "processed" / "features_engineered_test.csv"
DRIFT_THRESHOLD = 0.2
F1_DROP_THRESHOLD = 0.05

logger = logging.getLogger(__name__)


def _generate_report_from_defaults() -> None:
    from src.monitoring.drift_report import generate_drift_report

    reference = pd.read_csv(REFERENCE_PATH)
    current_source = DRIFTED_PATH if DRIFTED_PATH.exists() else CURRENT_PATH
    current = pd.read_csv(current_source)

    feature_cols = [c for c in reference.columns if c != "churn_flag"]
    ref_features = reference[feature_cols]
    cur_features = current[[c for c in feature_cols if c in current.columns]]

    generate_drift_report(ref_features, cur_features, target_column=None, prediction_column=None)
    logger.info("Rapport généré depuis %s", current_source.name)


def check_drift(report_path: Optional[Path] = None) -> bool:
    if report_path is None:
        report_path = REPORT_PATH

    if not report_path.exists():
        logger.warning("Rapport introuvable — génération automatique")
        _generate_report_from_defaults()

    with open(report_path) as f:
        report = json.load(f)

    for metric_info in report.get("metrics", []):
        metric_name = metric_info.get("metric", "")
        result = metric_info.get("result", {})

        if "DatasetDrift" in metric_name:
            drift_share = result.get("drift_share", 0.0)
            if drift_share > DRIFT_THRESHOLD:
                logger.warning(
                    "ALERTE dérive : %.0f%% des features ont drivé (seuil %.0f%%)",
                    drift_share * 100,
                    DRIFT_THRESHOLD * 100,
                )
                return True

        if "ClassificationQuality" in metric_name:
            ref_f1 = result.get("reference_metrics", {}).get("f1")
            cur_f1 = result.get("current_metrics", {}).get("f1")
            if ref_f1 is not None and cur_f1 is not None:
                f1_drop = ref_f1 - cur_f1
                if f1_drop > F1_DROP_THRESHOLD:
                    logger.warning(
                        "ALERTE F1 drop : %.3f → %.3f (seuil %.2f)",
                        ref_f1,
                        cur_f1,
                        F1_DROP_THRESHOLD,
                    )
                    return True

    logger.info("Monitoring OK — pas de dérive significative")
    return False
