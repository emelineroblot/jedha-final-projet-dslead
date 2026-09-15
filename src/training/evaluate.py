"""
Évaluation d'un classifieur binaire : métriques + figures (matrice de confusion, ROC, PR).

Utilisé par train.py (log MLflow), par validate.py (gate CI) et par le DAG de réentraînement.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Seuil de décision qui maximise le F1 (à calculer sur un jeu de VALIDATION, jamais sur le test)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "decision_threshold": float(threshold),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X)[:, 1]
    return compute_metrics(np.asarray(y), y_proba, threshold)


def save_figures(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, out_dir: Path, prefix: str = ""
) -> dict[str, Path]:
    """Matrice de confusion + courbe ROC en PNG. Retourne {nom: chemin}. Matplotlib optionnel."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # environnement API/CI sans matplotlib
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["no churn", "churn"])
    ax.set_yticks([0, 1], ["no churn", "churn"])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de confusion (seuil {threshold:.2f})")
    p = out_dir / f"{prefix}confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["confusion_matrix"] = p

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    p = out_dir / f"{prefix}roc_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["roc_curve"] = p

    return paths


def feature_importance(model, feature_names: list[str]) -> Optional[pd.DataFrame]:
    """Importance des features (arbres) ou coefficients (linéaire), triés décroissants."""
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_[0])
    else:
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": values})
    df["importance"] = df["importance"] / df["importance"].sum()
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
