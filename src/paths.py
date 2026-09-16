"""
Chemins du projet, surchargeables par la variable d'environnement CHURNGUARD_ROOT.

En local, la racine est déduite de l'emplacement de ce fichier. Dans les conteneurs
(Airflow, API), CHURNGUARD_ROOT pointe vers le dossier où src/, data/ et reports/
sont montés — le même code fonctionne dans les deux contextes.
"""
import os
from pathlib import Path

ROOT = Path(os.getenv("CHURNGUARD_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MODEL_ARTIFACTS_DIR = ROOT / "model_artifacts"

RAW_TRAIN_PATH = DATA_DIR / "customer_churn_dataset-training-master.csv"
RAW_TEST_PATH = DATA_DIR / "customer_churn_dataset-testing-master.csv"
# Référence d'entraînement (fichier train Kaggle, 440k lignes)
FEATURES_TRAIN_PATH = PROCESSED_DIR / "features_engineered.csv"
# Le fichier test Kaggle suit une distribution différente du train (dérive naturelle).
# Il est scindé en deux moitiés stratifiées :
#   - incoming : "nouvelles données de production" labellisées (scoring, contrôle de dérive, réentraînement)
#   - test     : hold-out d'évaluation, jamais utilisé pour entraîner ni choisir un seuil
FEATURES_INCOMING_PATH = PROCESSED_DIR / "features_incoming.csv"
FEATURES_TEST_PATH = PROCESSED_DIR / "features_engineered_test.csv"
# Dérive artificielle supplémentaire (simulate_drift.py) appliquée à incoming
FEATURES_DRIFTED_PATH = PROCESSED_DIR / "features_drifted.csv"

TARGET = "churn_flag"
