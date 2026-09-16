# Rapport dataset & preprocessing

## 1. Choix du dataset

**Contexte métier** : SeoLap (SaaS de monitoring SEO, en beta) n'a pas encore assez d'historique pour entraîner un modèle de churn. Il faut un dataset public dont les variables se mappent sur celles qu'on observera sur une plateforme SaaS : usage, support, paiement, ancienneté, offre.

**Premier essai — rivalytics (abandonné)** : 500 comptes fictifs, 5 tables relationnelles. Pipeline complet implémenté, puis F1 plafonné à **0,52** : toutes les corrélations feature ↔ cible sont < 0,09. Données synthétiques sans signal. Décision : changer de dataset (détail dans `contexte/problematiques-rencontrees.md`, P1).

**Dataset retenu — [muhammadshahidazeem / customer-churn-dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) (Kaggle)** : service par abonnement générique, 2 fichiers fournis.

| Fichier | Lignes | Colonnes | Taux de churn |
|---|---|---|---|
| `customer_churn_dataset-training-master.csv` | 440 833 | 12 | 56,7 % |
| `customer_churn_dataset-testing-master.csv` | 64 374 | 12 | 47,4 % |

Colonnes : `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`, `Churn`.

## 2. Qualité des données (EDA — `notebooks/02_eda_new_dataset.ipynb`)

- **Valeurs manquantes** : une seule ligne entièrement nulle dans le fichier train (artefact d'export) → supprimée. Aucune autre valeur manquante.
- **Doublons** : `CustomerID` unique, aucun doublon complet.
- **Types** : numériques déjà en float ; 3 catégorielles (`Gender`, `Subscription Type`, `Contract Length`).
- **Outliers** : distributions bornées et plausibles (Age 18–65, Tenure 1–60 mois, Payment Delay 0–30 j, Total Spend 100–1 000). Aucun traitement d'outliers nécessaire — les modèles retenus (arbres) y sont insensibles.
- **Déséquilibre** : 56,7 % de churn — léger, dans le sens inhabituel (majorité de churners). Géré par `scale_pos_weight` (XGBoost) et `class_weight="balanced"` (LogReg, RF), pas de ré-échantillonnage.

### Signal par feature (fichier train)

| Feature | Corrélation avec Churn | Lecture |
|---|---|---|
| `Support Calls` | **0,57** | Feature la plus discriminante : médiane 5 appels (churn) vs 1 (non-churn) |
| `Total Spend` | 0,43 (négative) | Les churners dépensent moins |
| `Payment Delay` | 0,31 | Retards de paiement → risque |
| `Age` | 0,21 | Effet modéré |
| `Last Interaction` | 0,10 | Faible |
| `Tenure`, `Usage Frequency` | < 0,06 | Quasi nul malgré des p-values significatives (effet du volume) |

Catégorielles : `Contract Length = Monthly` → **100 % de churn** dans le train (fuite de données évidente, voir §4) ; `Subscription Type` non discriminant (55,9–58,2 % sur les 3 offres) ; `Gender` légèrement discriminant (femmes 66 % vs hommes 50 %).

## 3. Le point clé : train et test ne suivent pas la même distribution

En validant le modèle sur le fichier test officiel, on observe une chute brutale : **F1 0,999 sur un split du train → 0,66 sur le fichier test** (AUC 0,73, le modèle prédit presque tout en churn). Les moyennes par classe le confirment :

| | Support Calls (non-churn / churn) | Payment Delay (non-churn / churn) | Total Spend (non-churn / churn) |
|---|---|---|---|
| Train | 1,6 / 5,2 | 10,0 / 15,3 | 749 / 544 |
| Test | **4,5** / 6,4 | **12,5** / 22,3 | **561** / 519 |

Les non-churners du fichier test ressemblent aux churners du fichier train. Les deux fichiers ont été générés différemment (dataset synthétique). Evidently mesure **9 features sur 15 en dérive** entre les deux.

**Décision** — plutôt que de masquer le problème (re-mélanger les deux fichiers), on l'exploite comme scénario MLOps réaliste : le fichier test **est** la production qui a dérivé.

- `features_engineered.csv` (440 k) : référence d'entraînement.
- Fichier test scindé en deux moitiés stratifiées (seed 42) :
  - `features_incoming.csv` (32 187) : « nouvelles données de production » labellisées — scoring quotidien, contrôle de dérive, réentraînement ;
  - `features_engineered_test.csv` (32 187) : **hold-out** d'évaluation, jamais utilisé pour entraîner ni calibrer un seuil.

Le pipeline complet (dérive détectée → réentraînement sur référence + fenêtre récente → promotion si meilleur sur le hold-out) est ainsi démontrable sur une dérive **réelle**, pas seulement simulée. Voir `docs/model-card.md` pour les résultats.

## 4. Preprocessing (`src/preprocessing/`)

| Étape | Module | Transformation | Justification |
|---|---|---|---|
| Nettoyage | `cleaner.py` | Suppression de la ligne nulle, drop `CustomerID`, `Gender` → 1 (homme) / 0 (femme), insensible à la casse | Identifiant sans valeur prédictive ; binaire suffit |
| Encodage ordinal | `features.py` | `Contract Length` : Monthly = 0, Quarterly = 1, Annual = 2 | Ordre naturel d'engagement. Monthly = 100 % churn dans le train, mais l'importance XGBoost reste à 12 % (non dominante) → conservée, surveillée |
| One-hot | `features.py` | `Subscription Type` → 3 colonnes (sans `drop_first`), colonnes garanties même si une modalité est absente du batch | Non ordinal ; robustesse sur petits lots (données SeoLap) |
| Features dérivées | `features.py` | `support_intensity = Support Calls / (Tenure + 1)` · `spend_per_month = Total Spend / (Tenure + 1)` · `payment_risk_score = Payment Delay × Support Calls` | Normaliser le support et la dépense par l'ancienneté ; interaction des deux signaux de risque les plus forts. `+1` évite la division par zéro |
| Cible | `features.py` | `Churn` → `churn_flag` (int) | |
| Ordre des colonnes | `features.py` | `FEATURE_COLUMNS` (15) figé = ordre attendu par le modèle et l'API | Évite les `feature_names mismatch` XGBoost |
| Split | `pipeline.py` | Test Kaggle → incoming / hold-out (50/50 stratifié) | §3 |

Pas de scaling pour les arbres ; `StandardScaler` intégré dans le `Pipeline` de la régression logistique uniquement.

## 5. Versioning

Données brutes, features et modèle exporté sont versionnés avec **DVC** (remote DagsHub) ; `dvc.yaml` décrit les stages `preprocess` → `train`, `dvc repro` rejoue la chaîne si les entrées changent. Le hash DVC du fichier d'entraînement est tagué sur chaque run MLflow (`data_dvc_md5`) — chaque modèle sait sur quelles données il a été entraîné.

## 6. Limites

- Dataset synthétique : les relations feature ↔ cible sont plus nettes que dans la réalité (F1 > 0,95 sur une distribution stable n'est pas transposable à SeoLap).
- Les 32 k lignes « incoming » jouent le rôle de données de production labellisées ; en réalité les labels de churn arrivent avec un délai (30–90 j) — le DAG hebdomadaire réentraîne sur la fenêtre labellisée disponible.
- `Contract Length = Monthly` reste une quasi-fuite ; sur les données SeoLap réelles, cette feature devra être ré-évaluée.
