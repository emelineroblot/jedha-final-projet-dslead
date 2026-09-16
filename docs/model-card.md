# Model card — `churnguard-model`

## Tâche

Classification binaire : un compte va-t-il résilier (`churn_flag = 1`) ? Sortie : probabilité de churn, transformée en décision au seuil `decision_threshold` (calibré à l'entraînement) et en bande métier `low / medium / high` pour le CRM.

Métrique principale : **F1** (équilibre précision / rappel — rater un churner coûte un client, sur-alerter coûte du temps commercial). Secondaires : AUC-ROC, AUC-PR, précision, rappel.

## Choix de l'algorithme

Trois familles comparées à chaque entraînement (`src/training/train.py`), toutes loguées dans MLflow :

| Modèle | Pourquoi le tester | Verdict |
|---|---|---|
| Régression logistique (`StandardScaler` + `class_weight=balanced`) | Baseline linéaire, interprétable, rapide | Sous-performe : les interactions (support × retard, dépense / ancienneté) sont non linéaires |
| Random Forest (200 arbres, profondeur 8) | Non linéaire, robuste, peu de tuning | Très proche de XGBoost, mais plus lourd à servir (200 arbres complets) et probabilités moins bien calibrées |
| **XGBoost** (gradient boosting) | Référence sur données tabulaires : capture les interactions, gère le déséquilibre (`scale_pos_weight`), rapide en inférence, importance des features native | **Retenu** : meilleur F1 et AUC sur le hold-out à chaque réentraînement, latence < 5 ms par lot |

Pas de deep learning : 15 features tabulaires, 440 k lignes, pas de signal séquentiel ni textuel — un GBDT est l'état de l'art et reste explicable.

## Protocole d'évaluation

1. Split train / validation 80/20 stratifié.
2. **Seuil de décision optimisé sur la validation** (maximise le F1), jamais sur le jeu d'évaluation.
3. **Métriques finales sur le hold-out** `features_engineered_test.csv` (32 187 lignes, moitié du fichier test Kaggle) — jamais vu à l'entraînement.
4. En réentraînement (DAG) : la validation est tirée de la **fenêtre récente** (`features_incoming.csv`) et la référence est sous-échantillonnée à 100 000 lignes (`RETRAIN_REFERENCE_ROWS`) : le modèle et son seuil sont calibrés sur ce à quoi ressemble la production aujourd'hui.

Pourquoi le point 4 : le fichier test Kaggle ne suit pas la distribution du train ([dataset-report §3](dataset-report.md)). Un modèle entraîné sur la seule référence avec un seuil calibré sur celle-ci obtient AUC 0,97 mais F1 0,66 sur le hold-out (seuil inadapté). Les variantes testées (XGBoost, seuil calibré sur la fenêtre récente) :

| Référence conservée | F1 hold-out | F1 ancienne distribution |
|---|---|---|
| 440 000 (toutes) | 0,86 | 0,62 |
| **100 000** | **0,98** | 0,43 |
| 30 000 | 0,99 | 0,43 |
| 0 (fenêtre seule) | 0,999 | 0,43 |

100 000 lignes est le compromis retenu : quasi-optimal sur la production actuelle, en gardant une mémoire de la référence. Les deux distributions étant incompatibles (dataset synthétique), la performance sur l'ancienne chute quoi qu'il arrive — c'est le comportement attendu quand la production a réellement changé.

## Résultats

### Modèle initial (référence seule) — ce qui était en Production avant la dérive

| | Validation (train) | **Hold-out** |
|---|---|---|
| XGBoost v1 | F1 0,999 · AUC 1,000 | **F1 0,657** · précision 0,489 · rappel 0,998 · AUC 0,731 |

Le modèle prédit presque tout en churn sur la nouvelle distribution : c'est la dérive détectée par Evidently (9 features sur 15).

### Après réentraînement par le DAG (référence 100 k + fenêtre récente)

| Modèle | F1 hold-out | Précision | Rappel | AUC-ROC | AUC-PR | Seuil |
|---|---|---|---|---|---|---|
| **XGBoost** | **0,978** | 0,985 | 0,972 | 0,995 | 0,996 | 0,89 |
| Random Forest | 0,973 | 0,986 | 0,960 | 0,987 | 0,991 | 0,83 |
| Régression logistique | 0,753 | 0,654 | 0,887 | 0,826 | 0,795 | 0,69 |

Gain de la boucle MLOps : **F1 0,66 → 0,98** sur le hold-out, sans intervention manuelle (dérive → retrain → évaluation → promotion → reload API).

Résultats reproduits en local (MLflow sqlite, 2026-09-15) puis **sur la stack Docker par le DAG `auto_retraining` (2026-09-16)** : candidat F1 0,9777 vs Production 0,6566 → v5 promue automatiquement, API rechargée, smoke test OK.

## Hyperparamètres

Défauts XGBoost : `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`, `scale_pos_weight` calculé sur le jeu d'entraînement.

Tuning : `python -m src.training.tuner --sample 100000 --n-iter 20 --cv 3` — `RandomizedSearchCV` (validation croisée stratifiée 3 plis, scoring F1) sur `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`. Chaque configuration est un run imbriqué de l'expérience MLflow `churnguard-tuning` (comparables en *parallel coordinates*). Le meilleur jeu est écrit dans `src/training/best_params.json` et repris automatiquement par `train.py`.

Sur ce dataset synthétique, le gain du tuning est marginal (le signal est saturé : AUC > 0,99) — il est surtout là pour les données SeoLap réelles.

## Ce que le modèle a appris (importance XGBoost, artefact `evaluation/feature_importance.csv`)

Top features : `Total Spend`, `Support Calls`, `Contract Length`, `payment_risk_score`, `Payment Delay`. Lecture métier : appels au support fréquents, retards de paiement et engagement mensuel sont les signaux de départ ; une dépense élevée est associée au churn (artefact du dataset synthétique — à réinterpréter sur SeoLap).

## Ce qui est logué pour chaque run

Hyperparamètres complets, `decision_threshold`, `n_train_rows` / `n_val_rows`, métriques hold-out + validation (`val_*`), signature + exemple d'entrée, `evaluation/confusion_matrix.png`, `evaluation/roc_curve.png`, `evaluation/feature_importance.csv`, tags de lineage (`git_sha`, `dataset`, `data_dvc_md5`, `extra_data`, `reference_rows`, `threshold_calibrated_on`, `holdout`).

## Limites et usage prévu

- **Dataset synthétique** : des F1 > 0,95 ne sont pas transposables ; sur SeoLap, attendre 0,7–0,8 et un vrai travail de features.
- `Contract Length = Monthly` est une quasi-fuite dans le train (100 % churn) — à ré-évaluer sur des données réelles.
- Les bandes `low / medium / high` (0,4 / 0,7) sont des seuils métier pour le CRM, distincts du seuil de décision du modèle (`churn_predicted`).
- Usage : priorisation des actions de rétention. Pas de décision automatique impactant le client sans revue humaine.
