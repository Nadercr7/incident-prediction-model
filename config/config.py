"""
Configuration Module
====================
Central configuration for the Incident Prediction Model.

All hyperparameters, feature definitions, and paths are managed here.
Adjust these values to experiment with different model configurations.

Author: Nader Mohamed
Date: February 2026
"""

import os
from pathlib import Path


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Default data file (users should provide their own)
DATA_FILE = DATA_DIR / "incidents.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "predictions.csv"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)


# =============================================================================
# COLUMN CONFIGURATION
# =============================================================================

# Required columns in input data
REQUIRED_COLUMNS = [
    'g_date',           # Date column (YYYY/MM/DD)
    'location_name',    # Location identifier
]

# Optional columns used for binary flags
OPTIONAL_COLUMNS = [
    'injury_type_name',
    'vehicle_damage_name',
    'vehicle_collision_type_name',
]

# Target column (auto-generated from aggregation)
TARGET_COL = 'incident_count'

# Log-transformed target
TARGET_COL_LOG = 'incident_count_log'


# =============================================================================
# FEATURE CONFIGURATION (56 engineered → 20 selected)
# =============================================================================

# --- Lag Features ---
LAG_FEATURES = [
    'incident_count_lag1', 'incident_count_lag2',
    'injury_count_lag1', 'damage_count_lag1',
    'lag1_log', 'lag1_sqrt',
]

# --- Trend Features ---
TREND_FEATURES = [
    'incident_trend', 'trend_acceleration',
    'yoy_pct_change',
]

# --- Interaction Features (key breakthrough) ---
INTERACTION_FEATURES = [
    'weighted_trend', 'trend_magnitude',
    'trend_x_lag1', 'lag1_x_mean',
    'lag1_to_mean_ratio', 'lag1_to_max_ratio',
]

# --- Seasonal Patterns ---
SEASONAL_FEATURES = [
    'peak_quarter_count', 'seasonal_variance',
    'h1_proportion', 'q4_proportion',
]

# --- Historical Statistics ---
HISTORICAL_FEATURES = [
    'hist_mean_incidents', 'hist_std_incidents',
    'hist_max_incidents', 'hist_min_incidents',
    'incident_rolling_avg_2y', 'incident_rolling_avg_3y',
    'cv',
]

# --- All Selected Features (20 total, from 56 engineered) ---
SELECTED_FEATURE_COLS = (
    LAG_FEATURES + TREND_FEATURES + INTERACTION_FEATURES + HISTORICAL_FEATURES
)

# Complete engineered features (56 total, before selection)
ALL_ENGINEERED_FEATURES = SELECTED_FEATURE_COLS + SEASONAL_FEATURES + [
    'year',
    'incident_count_lag3', 'injury_count_lag2', 'injury_count_lag3',
    'damage_count_lag2', 'damage_count_lag3',
    'collision_count_lag1', 'collision_count_lag2', 'collision_count_lag3',
    'lag1_squared', 'lag1_cubed', 'lag2_log', 'lag3_log',
    'lag2_sqrt', 'lag3_sqrt',
    'hist_total_incidents',
    'hist_avg_injury_rate', 'hist_avg_damage_rate', 'hist_avg_collision_rate',
    'peak_month', 'peak_day',
    'q1_count', 'q2_count', 'q3_count', 'q4_count',
    'h2_proportion',
    'lag_diff_12', 'lag_diff_23',
    'injury_rate_lag1', 'damage_rate_lag1', 'collision_rate_lag1',
    'high_volume_flag', 'low_volume_flag',
    'trend_positive_flag', 'trend_negative_flag',
]


# =============================================================================
# MODEL HYPERPARAMETERS (Final: Pure Gradient Boosting)
# =============================================================================

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 150,
    'max_depth': 3,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'min_samples_leaf': 4,
    'random_state': 42,
}


# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================

FEATURE_SELECTION = {
    'method': 'averaged_importance',     # RF + ExtraTrees averaged
    'n_features_to_select': 20,
    'rf_n_estimators': 200,
    'et_n_estimators': 200,
    'random_state': 42,
}


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

MAX_LAG_YEARS = 3
PREDICTION_YEAR = 2026
USE_LOG_TRANSFORM = True        # Log-transform target (key technique)
USE_FEATURE_SELECTION = True     # Select top-20 from 56 features


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

FIGURE_DPI = 150
FIGURE_SIZE_LARGE = (16, 12)
FIGURE_SIZE_MEDIUM = (14, 10)
FIGURE_SIZE_SMALL = (10, 8)

COLORS = {
    'primary': 'steelblue',
    'secondary': 'coral',
    'success': 'seagreen',
    'warning': 'gold',
    'danger': 'indianred',
    'neutral': 'gray',
}


# =============================================================================
# UTILITY
# =============================================================================

def print_config():
    """Print current configuration settings."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SETTINGS")
    print("=" * 60)
    print(f"\nPaths:")
    print(f"  Data file:  {DATA_FILE}")
    print(f"  Models dir: {MODELS_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"\nFeatures:")
    print(f"  Engineered: {len(ALL_ENGINEERED_FEATURES)}")
    print(f"  Selected:   {len(SELECTED_FEATURE_COLS)}")
    print(f"\nModel: Gradient Boosting")
    for k, v in GRADIENT_BOOSTING_PARAMS.items():
        print(f"  {k}: {v}")
    print(f"\nSettings:")
    print(f"  Log transform:     {USE_LOG_TRANSFORM}")
    print(f"  Feature selection: {USE_FEATURE_SELECTION}")
    print(f"  Prediction year:   {PREDICTION_YEAR}")


if __name__ == "__main__":
    print_config()
