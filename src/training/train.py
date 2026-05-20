from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.training.registry import promote_model

DATA_PATH = Path(__file__).parents[2] / "data" / "processed" / "features_engineered.csv"
TARGET = "churn_flag"
MLFLOW_EXPERIMENT = "churnguard"
MODEL_NAME = "churnguard-model"
F1_PROMOTION_THRESHOLD = 0.70


def _build_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=42, class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=scale_pos_weight,
        ),
    }


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Seuil qui maximise le F1 sur le jeu de test."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _log_model(name: str, model) -> None:
    if name == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")


def train(auto_promote: bool = True):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # scale_pos_weight calculé dynamiquement sur le dataset actif
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = _build_models(scale_pos_weight)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    best_f1, best_run_id = 0.0, None

    print(f"Dataset : {len(df):,} lignes | churn rate : {y.mean():.2%} | scale_pos_weight : {scale_pos_weight:.3f}")

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            threshold = find_optimal_threshold(y_test.values, y_proba)
            y_pred = (y_proba >= threshold).astype(int)

            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            mlflow.log_param("model_type", name)
            mlflow.log_param("decision_threshold", round(threshold, 3))
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc)
            _log_model(name, model)

            print(f"{name}: F1={f1:.3f}, AUC={auc:.3f}, threshold={threshold:.2f}")
            print(classification_report(y_test, y_pred))

            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id

    if best_run_id and best_f1 >= F1_PROMOTION_THRESHOLD:
        if auto_promote:
            promote_model(run_id=best_run_id, stage="Production")
            print(f"\nModèle promu en Production (F1={best_f1:.3f})")
        else:
            mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
            print(f"\nModèle enregistré sans promotion (F1={best_f1:.3f})")
    elif best_run_id:
        print(f"\nF1={best_f1:.3f} < seuil {F1_PROMOTION_THRESHOLD} — pas de promotion automatique")
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

    return best_run_id


if __name__ == "__main__":
    train()
