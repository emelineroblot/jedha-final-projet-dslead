"""
Optimisation des hyperparamètres XGBoost (RandomizedSearchCV, validation croisée stratifiée).

Chaque configuration testée est loguée comme run imbriqué dans l'expérience MLflow
`churnguard-tuning` (comparable dans l'UI : parallel coordinates). Le meilleur jeu de
paramètres est écrit dans `src/training/best_params.json`, repris automatiquement par train.py.

    python -m src.training.tuner --sample 100000 --n-iter 20 --cv 3
"""
import argparse
import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.paths import FEATURES_TRAIN_PATH, TARGET
from src.preprocessing.features import FEATURE_COLUMNS
from src.training.train import BEST_PARAMS_PATH

TUNING_EXPERIMENT = "churnguard-tuning"

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(3, 9),
    "learning_rate": uniform(0.01, 0.29),  # [0.01, 0.30]
    "subsample": uniform(0.6, 0.4),  # [0.6, 1.0]
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0.0, 5.0),
    "reg_lambda": uniform(0.5, 4.5),
}
INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight"}


def tune(sample: int = 100_000, n_iter: int = 20, cv: int = 3, seed: int = 42) -> dict:
    df = pd.read_csv(FEATURES_TRAIN_PATH)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    X, y = df[FEATURE_COLUMNS], df[TARGET].astype(int)
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    base = XGBClassifier(
        eval_metric="logloss", random_state=seed, n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    search = RandomizedSearchCV(
        base,
        PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,  # XGBoost parallélise déjà sur les cœurs
        verbose=1,
        return_train_score=True,
    )

    mlflow.set_experiment(TUNING_EXPERIMENT)
    with mlflow.start_run(run_name=f"random_search_{datetime.now(timezone.utc):%Y%m%d_%H%M}") as parent:
        mlflow.log_params({"n_iter": n_iter, "cv": cv, "sample_rows": len(df), "scoring": "f1"})
        search.fit(X, y)

        results = pd.DataFrame(search.cv_results_)
        for i, row in results.iterrows():
            with mlflow.start_run(run_name=f"config_{i:02d}", nested=True):
                mlflow.log_params({k: row[f"param_{k}"] for k in PARAM_DISTRIBUTIONS})
                mlflow.log_metrics({
                    "cv_f1_mean": float(row["mean_test_score"]),
                    "cv_f1_std": float(row["std_test_score"]),
                    "train_f1_mean": float(row["mean_train_score"]),
                    "fit_time_s": float(row["mean_fit_time"]),
                })

        best = {
            k: (int(v) if k in INT_PARAMS else round(float(v), 4))
            for k, v in search.best_params_.items()
        }
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        print(f"\nMeilleur F1 CV : {search.best_score_:.4f}\nParamètres : {best}")

        payload = {
            "params": best,
            "cv_f1": float(search.best_score_),
            "n_iter": n_iter,
            "cv": cv,
            "sample_rows": len(df),
            "mlflow_run_id": parent.info.run_id,
            "tuned_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        mlflow.log_artifact(str(BEST_PARAMS_PATH))
        print(f"Paramètres sauvegardés dans {BEST_PARAMS_PATH}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning XGBoost (RandomizedSearchCV)")
    parser.add_argument("--sample", type=int, default=100_000, help="Lignes échantillonnées (0 = tout)")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()
    tune(sample=args.sample, n_iter=args.n_iter, cv=args.cv)
