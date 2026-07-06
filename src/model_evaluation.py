"""
Model Evaluation — matches notebook cell 14 exactly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error


def evaluate_model(test_data: pd.DataFrame) -> dict:
    """Print and return evaluation metrics, exactly matching notebook."""

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60 + "\n")

    y_true = test_data['incident_count'].values
    y_pred = test_data['predicted'].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # sMAPE
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

    # Baseline: predict historical mean per location
    baseline_pred = np.full_like(y_true, y_true.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)

    metrics = {
        'R2': r2,
        'MAE': mae,
        'Median_AE': median_ae,
        'RMSE': rmse,
        'sMAPE': smape,
        'Baseline_MAE': baseline_mae,
    }

    print(f"    R² Score:       {r2:.4f}")
    print(f"    MAE:            {mae:.2f}")
    print(f"    Median AE:      {median_ae:.2f}")
    print(f"    RMSE:           {rmse:.2f}")
    print(f"    sMAPE:          {smape:.2f}%")
    print(f"    Baseline MAE:   {baseline_mae:.2f}")
    print(f"    Improvement:    {((baseline_mae - mae) / baseline_mae * 100):.1f}% vs baseline")

    return metrics
