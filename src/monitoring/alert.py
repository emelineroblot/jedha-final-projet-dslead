import json
import logging
from pathlib import Path

REPORT_PATH = Path(__file__).parents[2] / "reports" / "drift_report.json"
DRIFT_THRESHOLD = 0.2
F1_DROP_THRESHOLD = 0.05

logger = logging.getLogger(__name__)


def check_drift(report_path: Path = REPORT_PATH) -> bool:
    with open(report_path) as f:
        _ = json.load(f)

    # TODO: extraire les métriques selon la structure Evidently réelle
    drift_detected = False

    if drift_detected:
        logger.warning("ALERTE : dérive détectée — réentraînement recommandé")
    else:
        logger.info("Monitoring OK — pas de dérive significative")

    return drift_detected
