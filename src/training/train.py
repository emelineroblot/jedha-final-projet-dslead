"""
Entraînement, comparaison et enregistrement des modèles dans MLflow.

Protocole d'évaluation :
- split train / validation (80/20 stratifié) sur le jeu d'entraînement ;
- le seuil de décision est optimisé sur la VALIDATION — en réentraînement, la validation est tirée
  de la fenêtre RÉCENTE (extra_data), et la référence est sous-échantillonnée (reference_rows) :
  le modèle et son seuil sont calibrés sur ce à quoi ressemble la production aujourd'hui ;
- les métriques finales sont calculées sur le HOLD-OUT (`features_engineered_test.csv`, moitié du
  fichier test Kaggle), jamais vu pendant l'entraînement ni le choix du seuil. Le fichier test Kaggle
  ne suit pas la distribution du train : un modèle entraîné sur la seule référence y obtient ~0.66 de F1 ;
  réentraîné avec la fenêtre `features_incoming.csv` (autre moitié), il remonte — c'est la boucle MLOps.

Chaque run MLflow logue : hyperparamètres, métriques (F1, précision, rappel, AUC-ROC, AUC-PR),
seuil, signature du modèle, importance des features, matrice de confusion, courbe ROC,
et le lineage (git sha, hash DVC des données, dataset).
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.paths import FEATURES_TEST_PATH, FEATURES_TRAIN_PATH, ROOT, TARGET
from src.preprocessing.features import FEATURE_COLUMNS
from src.training.evaluate import (
    compute_metrics,
    feature_importance,
    find_optimal_threshold,
    save_figures,
)
from src.training.registry import promote_model

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "churnguard"
MODEL_NAME = "churnguard-model"
DATASET_NAME = "kaggle/muhammadshahidazeem/customer-churn-dataset"
F1_PROMOTION_THRESHOLD = 0.70
BEST_PARAMS_PATH = Path(__file__).parent / "best_params.json"
# Lignes de référence conservées lors d'un réentraînement avec fenêtre récente (0 = toutes)
RETRAIN_REFERENCE_ROWS = int(os.getenv("RETRAIN_REFERENCE_ROWS", "100000"))
VAL_SIZE = 0.2

XGB_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _xgb_params() -> dict:
    """Hyperparamètres XGBoost : ceux trouvés par tuner.py s'ils existent, sinon les défauts."""
    params = dict(XGB_DEFAULT_PARAMS)
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            params.update(json.load(f).get("params", {}))
        logger.info("Hyperparamètres XGBoost chargés depuis %s", BEST_PARAMS_PATH.name)
    return params


def _build_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=42, class_weight="balanced", n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            **_xgb_params(),
            eval_metric="logloss", random_state=42,
            scale_pos_weight=scale_pos_weight, n_jobs=-1,
        ),
    }


def _params_of(model) -> dict:
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    return {k: v for k, v in est.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None}


def _log_model(name: str, model, X_sample: pd.DataFrame) -> None:
    signature = infer_signature(X_sample, model.predict_proba(X_sample)[:, 1])
    # artifact_path="model" → URI runs:/<run_id>/model (utilisé par registry.py et les DAGs)
    kwargs = dict(artifact_path="model", signature=signature, input_example=X_sample.head(3))
    if name == "xgboost":
        mlflow.xgboost.log_model(model, **kwargs)
    else:
        mlflow.sklearn.log_model(model, **kwargs)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _dvc_md5(data_path: Path) -> str:
    """
    Hash DVC du fichier de données → lineage données ↔ modèle.
    Lu dans le pointer `<fichier>.dvc` (dvc add) ou, à défaut, dans `dvc.lock` (sortie d'un stage dvc.yaml).
    """
    pointer = data_path.with_suffix(data_path.suffix + ".dvc")
    if pointer.exists():
        for line in pointer.read_text().splitlines():
            line = line.strip().lstrip("- ")
            if line.startswith("md5:"):
                return line.split("md5:")[1].strip()
        return "unknown"

    lock = ROOT / "dvc.lock"
    if lock.exists():
        try:
            rel = Path(data_path).resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            rel = Path(data_path).name
        lines = lock.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip().lstrip("- ").startswith("path:") and line.split("path:")[1].strip() == rel:
                for nxt in lines[i + 1 : i + 4]:
                    if nxt.strip().startswith("md5:"):
                        return nxt.split("md5:")[1].strip()
    return "untracked"


def load_training_data(
    data_path: Path = FEATURES_TRAIN_PATH,
    extra_data_path: Optional[Path] = None,
    reference_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Jeu d'entraînement = référence (éventuellement sous-échantillonnée) + nouvelle fenêtre labellisée.
    Le DAG de réentraînement passe en `extra_data_path` les données courantes
    (`features_incoming.csv` ou `features_drifted.csv`) pour que le candidat apprenne la nouvelle distribution.
    """
    df = pd.read_csv(data_path)
    if reference_rows and len(df) > reference_rows:
        df = df.sample(reference_rows, random_state=42)
    if extra_data_path is not None and Path(extra_data_path).exists():
        extra = pd.read_csv(extra_data_path)
        extra = extra[[c for c in df.columns if c in extra.columns]]
        df = pd.concat([df, extra], ignore_index=True)
        logger.info("Données supplémentaires ajoutées : %s (%d lignes)", Path(extra_data_path).name, len(extra))
    return df[FEATURE_COLUMNS + [TARGET]]


def _split(data_path: Path, extra_data_path: Optional[Path], reference_rows: Optional[int]):
    """
    Sans fenêtre récente : validation = 20 % de la référence.
    Avec fenêtre récente : validation = 20 % de la fenêtre récente (seuil calibré sur la production
    actuelle), entraînement = référence échantillonnée + 80 % de la fenêtre.
    """
    has_extra = extra_data_path is not None and Path(extra_data_path).exists()
    if not has_extra:
        df = load_training_data(data_path, None, reference_rows)
        return train_test_split(
            df[FEATURE_COLUMNS], df[TARGET].astype(int), test_size=VAL_SIZE, stratify=df[TARGET], random_state=42
        )

    extra = pd.read_csv(extra_data_path)[FEATURE_COLUMNS + [TARGET]]
    extra_train, extra_val = train_test_split(extra, test_size=VAL_SIZE, stratify=extra[TARGET], random_state=42)
    ref = load_training_data(data_path, None, reference_rows)
    train_df = pd.concat([ref, extra_train], ignore_index=True)
    return (
        train_df[FEATURE_COLUMNS],
        extra_val[FEATURE_COLUMNS],
        train_df[TARGET].astype(int),
        extra_val[TARGET].astype(int),
    )


def train(
    auto_promote: bool = True,
    data_path: Path = FEATURES_TRAIN_PATH,
    extra_data_path: Optional[Path] = None,
    holdout_path: Path = FEATURES_TEST_PATH,
    model_names: Optional[list[str]] = None,
    experiment: str = MLFLOW_EXPERIMENT,
    reference_rows: Optional[int] = None,
) -> Optional[str]:
    """Entraîne les modèles, logue tout dans MLflow, retourne le run_id du meilleur (par F1 hold-out)."""
    has_extra = extra_data_path is not None and Path(extra_data_path).exists()
    if has_extra and reference_rows is None:
        reference_rows = RETRAIN_REFERENCE_ROWS
    X_train, X_val, y_train, y_val = _split(data_path, extra_data_path, reference_rows)

    # scale_pos_weight calculé dynamiquement sur le dataset actif
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    holdout = None
    if holdout_path is not None and Path(holdout_path).exists():
        df_hold = pd.read_csv(holdout_path)
        holdout = (df_hold[FEATURE_COLUMNS], df_hold[TARGET].astype(int))

    lineage = {
        "git_sha": _git_sha(),
        "dataset": DATASET_NAME,
        "data_path": str(Path(data_path).name),
        "data_dvc_md5": _dvc_md5(Path(data_path)),
        "extra_data": str(Path(extra_data_path).name) if has_extra else "none",
        "reference_rows": str(reference_rows or "all"),
        "threshold_calibrated_on": "recent_window" if has_extra else "reference_split",
        "holdout": Path(holdout_path).name if holdout else "validation_split",
    }

    models = _build_models(scale_pos_weight)
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}

    mlflow.set_experiment(experiment)
    best_f1, best_run_id = -1.0, None
    val_source = "fenêtre récente" if has_extra else "référence"
    print(
        f"Entraînement : {len(X_train):,} lignes | validation : {len(X_val):,} ({val_source}) "
        f"| churn rate : {y_train.mean():.2%} | scale_pos_weight : {scale_pos_weight:.3f}"
    )

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)

            proba_val = model.predict_proba(X_val)[:, 1]
            threshold = find_optimal_threshold(y_val.values, proba_val)
            val_metrics = compute_metrics(y_val.values, proba_val, threshold)

            if holdout:
                proba_eval = model.predict_proba(holdout[0])[:, 1]
                y_eval = holdout[1].values
            else:
                proba_eval, y_eval = proba_val, y_val.values
            metrics = compute_metrics(y_eval, proba_eval, threshold)

            mlflow.log_param("model_type", name)
            mlflow.log_params(_params_of(model))
            mlflow.log_param("decision_threshold", round(threshold, 3))
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_val_rows", len(X_val))
            mlflow.set_tags(lineage)
            mlflow.log_metrics(metrics)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if k != "decision_threshold"})

            with tempfile.TemporaryDirectory() as tmp:
                for _, path in save_figures(y_eval, proba_eval, threshold, Path(tmp)).items():
                    mlflow.log_artifact(str(path), artifact_path="evaluation")
                fi = feature_importance(model, FEATURE_COLUMNS)
                if fi is not None:
                    fi_path = Path(tmp) / "feature_importance.csv"
                    fi.to_csv(fi_path, index=False)
                    mlflow.log_artifact(str(fi_path), artifact_path="evaluation")

            _log_model(name, model, X_train.head(100))

            print(
                f"{name}: F1={metrics['f1_score']:.3f} | precision={metrics['precision']:.3f} "
                f"| recall={metrics['recall']:.3f} | AUC={metrics['auc_roc']:.3f} | threshold={threshold:.2f}"
            )

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_id = run.info.run_id

    if best_run_id is None:
        return None

    if best_f1 >= F1_PROMOTION_THRESHOLD and auto_promote:
        promote_model(run_id=best_run_id, stage="Production")
        print(f"\nModèle promu en Production (F1={best_f1:.3f})")
    else:
        mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
        reason = "auto_promote=False" if best_f1 >= F1_PROMOTION_THRESHOLD else f"F1 < seuil {F1_PROMOTION_THRESHOLD}"
        print(f"\nModèle enregistré sans promotion ({reason}, F1={best_f1:.3f})")

    return best_run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement ChurnGuard")
    parser.add_argument("--no-promote", action="store_true", help="Enregistre sans promouvoir en Production")
    parser.add_argument("--extra-data", type=Path, default=None, help="CSV labellisé à ajouter à l'entraînement")
    parser.add_argument("--models", nargs="*", default=None, help="Sous-ensemble : logistic_regression random_forest xgboost")
    parser.add_argument("--reference-rows", type=int, default=None, help="Lignes de référence conservées (défaut : toutes, ou RETRAIN_REFERENCE_ROWS avec --extra-data)")
    args = parser.parse_args()
    train(
        auto_promote=not args.no_promote,
        extra_data_path=args.extra_data,
        model_names=args.models,
        reference_rows=args.reference_rows,
    )
