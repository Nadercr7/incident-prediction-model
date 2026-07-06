"""
Model Training — matches notebook cell 12 exactly.
GradientBoosting with hardcoded params (no tuning), log-transformed target.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


def train_model(
    df_features: pd.DataFrame,
    selected_features: List[str],
    train_years: List[int],
    test_year: int,
) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """
    Train GB model on log1p(incident_count), exactly matching notebook cell 12.

    Returns (gb_model, train_data, test_data)
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60 + "\n")

    train_data = df_features[df_features['year'].isin(train_years)].copy()
    test_data = df_features[df_features['year'] == test_year].copy()

    print(f"    Train years: {train_years} ({len(train_data)} samples)")
    print(f"    Test year:   {test_year} ({len(test_data)} samples)")

    X_train = train_data[selected_features]
    y_train = train_data['incident_count']
    y_train_log = np.log1p(y_train)

    X_test = test_data[selected_features]

    # ---- Gradient Boosting (primary model) ----
    gb_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_split=5,
        random_state=42,
    )
    gb_model.fit(X_train, y_train_log)

    # ---- Random Forest (secondary — trained for completeness) ----
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train_log)

    # ---- Predictions (GB only, matching notebook's best_weight=0.0) ----
    gb_pred = np.expm1(gb_model.predict(X_test))
    rf_pred = np.expm1(rf_model.predict(X_test))

    # Notebook optimal_weight = 0.0 → pure GB
    final_pred = np.maximum(gb_pred, 0).round()

    test_data['predicted'] = final_pred

    # Feature importance
    fi = pd.Series(gb_model.feature_importances_, index=selected_features).sort_values(ascending=False)
    print("\n    Top features (GB):")
    for feat, imp in fi.head(10).items():
        print(f"      {feat}: {imp:.4f}")

    return gb_model, train_data, test_data
