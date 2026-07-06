"""
Configuration — minimal, matching notebook parameters.
"""

import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'IncidentData.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# === Data Settings ===
MIN_YEAR = 2023
MAX_YEAR = 2025
PREDICTION_YEAR = 2026

# === Model Parameters (notebook cell 12 — hardcoded, no tuning) ===
GB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 3,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'min_samples_split': 5,
    'random_state': 42,
}

# === Feature Selection ===
N_TOP_FEATURES = 20
