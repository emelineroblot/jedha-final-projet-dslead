from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parents[2] / "data" / "processed" / "features_engineered.csv"
TARGET = "churn_flag"
MLFLOW_EXPERIMENT = "churnguard"
MODEL_NAME = "churnguard-model"

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
}


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    best_f1, best_run_id = 0.0, None

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_roc", auc)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"{name}: F1={f1:.3f}, AUC={auc:.3f}")
            print(classification_report(y_test, y_pred))

            if f1 > best_f1:
                best_f1 = f1
                best_run_id = mlflow.active_run().info.run_id

    # Enregistrer le meilleur modèle dans le Registry
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)
        print(f"\nMeilleur modèle enregistré (F1={best_f1:.3f}) : {model_uri}")


if __name__ == "__main__":
    train()
