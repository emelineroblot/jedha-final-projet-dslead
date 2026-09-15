"""
Contrôle de dérive : génère (toujours) un rapport Evidently puis applique les seuils d'alerte.

Source des données "courantes", par ordre de priorité :
1. chemin explicite passé en argument ;
2. prédictions réellement reçues par l'API (table `predictions`, N derniers jours) ;
3. `features_drifted.csv` s'il existe (démo : `simulate_drift.py`, bruit supplémentaire) ;
4. `features_incoming.csv` — nouvelles données de production labellisées (dérive naturelle
   du fichier test Kaggle par rapport au train) ;
5. `features_engineered_test.csv` (hold-out, fallback).
"""
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.monitoring import store
from src.monitoring.notify import send_alert
from src.paths import (
    FEATURES_DRIFTED_PATH,
    FEATURES_INCOMING_PATH,
    FEATURES_TEST_PATH,
    FEATURES_TRAIN_PATH,
    REPORTS_DIR,
    TARGET,
)
from src.preprocessing.features import FEATURE_COLUMNS

REPORT_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_THRESHOLD = 0.2  # part des features en dérive
F1_DROP_THRESHOLD = 0.05  # chute de F1 tolérée (référence − courant)
REFERENCE_SAMPLE = 50_000  # lignes de référence échantillonnées (vitesse du rapport)

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    drifted: bool
    reason: str
    drift_share: float
    n_drifted_features: int
    f1_reference: Optional[float]
    f1_current: Optional[float]
    current_source: str
    n_current: int
    report_path: str

    def as_dict(self) -> dict:
        return asdict(self)


def _load_current(current_path: Optional[Path], days: int) -> tuple[pd.DataFrame, str]:
    if current_path is not None:
        return pd.read_csv(current_path), Path(current_path).name

    live = store.fetch_recent_features(days=days)
    if live is not None:
        return live, f"predictions_db_last_{days}d"

    for path in (FEATURES_DRIFTED_PATH, FEATURES_INCOMING_PATH, FEATURES_TEST_PATH):
        if path.exists():
            return pd.read_csv(path), path.name
    raise FileNotFoundError("Aucune donnée courante disponible (lancer le preprocessing ou dvc pull)")


def run_drift_check(
    current_path: Optional[Path] = None,
    reference_path: Path = FEATURES_TRAIN_PATH,
    days: int = 7,
    notify: bool = True,
) -> DriftResult:
    from src.monitoring.drift_report import generate_drift_report

    reference = pd.read_csv(reference_path)
    if len(reference) > REFERENCE_SAMPLE:
        reference = reference.sample(REFERENCE_SAMPLE, random_state=42)
    current, source = _load_current(current_path, days)

    feature_cols = [c for c in FEATURE_COLUMNS if c in current.columns]
    ref = reference[feature_cols].copy()
    cur = current[feature_cols].copy()

    # Suivi de performance si les labels ET les scores sont connus (prédictions stockées + labels différés)
    prediction_column = None
    label_col = TARGET if TARGET in current.columns else ("actual_churn" if "actual_churn" in current.columns else None)
    if label_col and "churn_score" in current.columns and current[label_col].notna().mean() > 0.5:
        ref[TARGET] = reference[TARGET].values
        ref["prediction"] = reference[TARGET].astype(float).values  # référence : labels parfaits
        cur[TARGET] = current[label_col].fillna(0).astype(int).values
        cur["prediction"] = current["churn_score"].values
        prediction_column = "prediction"

    report = generate_drift_report(
        ref,
        cur,
        target_column=TARGET if prediction_column else None,
        prediction_column=prediction_column,
    )
    result = _evaluate_report(report, source=source, n_current=len(current))
    logger.info("Rapport généré depuis %s (%d lignes) — dérive : %s", source, len(current), result.drifted)

    if notify and result.drifted:
        send_alert(
            "Dérive détectée",
            f"{result.reason}\nSource : {source} ({result.n_current} lignes) — rapport : {result.report_path}",
            level="warning",
        )
    return result


def _evaluate_report(report: dict, source: str = "", n_current: int = 0) -> DriftResult:
    drift_share, n_drifted = 0.0, 0
    f1_ref, f1_cur = None, None

    for metric_info in report.get("metrics", []):
        metric_name = metric_info.get("metric", "")
        result = metric_info.get("result", {})
        # Le nom exact varie selon la version d'Evidently → test en substring.
        # Attention : `drift_share` est le SEUIL paramétré d'Evidently (0.5 par défaut),
        # la part observée est `share_of_drifted_columns` (ou drifted / total).
        if "DatasetDrift" in metric_name:
            n_drifted = int(result.get("number_of_drifted_columns", 0))
            n_cols = int(result.get("number_of_columns", 0))
            if "share_of_drifted_columns" in result:
                drift_share = float(result["share_of_drifted_columns"])
            elif n_cols:
                drift_share = n_drifted / n_cols
            else:
                drift_share = float(result.get("drift_share", 0.0))
        if "ClassificationQuality" in metric_name:
            f1_ref = result.get("reference", {}).get("f1", result.get("reference_metrics", {}).get("f1"))
            f1_cur = result.get("current", {}).get("f1", result.get("current_metrics", {}).get("f1"))

    reasons = []
    if drift_share > DRIFT_THRESHOLD:
        reasons.append(f"{drift_share:.0%} des features en dérive (seuil {DRIFT_THRESHOLD:.0%})")
    if f1_ref is not None and f1_cur is not None and (f1_ref - f1_cur) > F1_DROP_THRESHOLD:
        reasons.append(f"F1 {f1_ref:.3f} → {f1_cur:.3f} (chute > {F1_DROP_THRESHOLD})")

    return DriftResult(
        drifted=bool(reasons),
        reason=" ; ".join(reasons) or "pas de dérive significative",
        drift_share=drift_share,
        n_drifted_features=n_drifted,
        f1_reference=f1_ref,
        f1_current=f1_cur,
        current_source=source,
        n_current=n_current,
        report_path=str(REPORT_PATH),
    )


def check_drift(report_path: Optional[Path] = None, force: bool = True) -> bool:
    """
    Interface booléenne utilisée par le DAG et la CLI.
    `force=True` (défaut) régénère toujours le rapport ; `force=False` relit un rapport existant.
    """
    path = Path(report_path) if report_path else REPORT_PATH
    if force or not path.exists():
        return run_drift_check().drifted

    with open(path) as f:
        report = json.load(f)
    result = _evaluate_report(report, source=path.name)
    if result.drifted:
        logger.warning("ALERTE dérive : %s", result.reason)
    else:
        logger.info("Monitoring OK — %s", result.reason)
    return result.drifted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Contrôle de dérive Evidently")
    parser.add_argument("--current", type=Path, default=None, help="CSV de données courantes")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()
    res = run_drift_check(current_path=args.current, notify=not args.no_notify)
    print(json.dumps(res.as_dict(), indent=2))
