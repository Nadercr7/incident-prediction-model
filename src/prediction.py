"""
2026 Prediction — matches notebook cell 19 exactly.
"""

import numpy as np
import pandas as pd
from typing import List


def predict_2026(
    df_features: pd.DataFrame,
    gb_model,
    selected_features: List[str],
    output_path: str = None,
) -> pd.DataFrame:
    """Generate 2026 predictions using ONLY gb_model (best_weight=0.0)."""

    print("\n" + "=" * 60)
    print("  2026 PREDICTIONS")
    print("=" * 60 + "\n")

    # Latest year data
    data_2025 = df_features[df_features['year'] == 2025].copy()

    # Create 2026 prediction dataframe
    pred_2026 = data_2025[['location_name']].copy()

    # For lag features: shift values forward
    # 2025 actual → 2026 lag1, 2024 actual → 2026 lag2
    for col in selected_features:
        if 'lag2' in col:
            lag1_col = col.replace('lag2', 'lag1')
            if lag1_col in data_2025.columns:
                pred_2026[col] = data_2025[lag1_col].values
            else:
                pred_2026[col] = 0
        elif '_lag1' in col:
            base_col = col.replace('_lag1', '')
            if base_col in data_2025.columns:
                pred_2026[col] = data_2025[base_col].values
            elif col in data_2025.columns:
                pred_2026[col] = data_2025[col].values
            else:
                pred_2026[col] = 0
        else:
            if col in data_2025.columns:
                pred_2026[col] = data_2025[col].values
            else:
                pred_2026[col] = 0

    # === Recompute derived features for 2026 ===

    # Rate features
    pred_2026['damage_rate_lag1'] = data_2025['damage_count'].values / (data_2025['incident_count'].values + 1)
    pred_2026['injury_rate_lag1'] = data_2025['injury_count'].values / (data_2025['incident_count'].values + 1)
    pred_2026['collision_rate_lag1'] = data_2025['collision_count'].values / (data_2025['incident_count'].values + 1)

    # Growth rate for 2026
    data_2024_inc = df_features[df_features['year'] == 2024].set_index('location_name')['incident_count']
    data_2025_inc = data_2025.set_index('location_name')['incident_count']
    growth = (data_2025_inc - data_2024_inc) / (data_2024_inc + 1)
    pred_2026['growth_rate'] = pred_2026['location_name'].map(growth).fillna(0).values

    # Trend for 2026
    trend_2026 = data_2025_inc.subtract(data_2024_inc, fill_value=0)
    pred_2026['trend'] = pred_2026['location_name'].map(trend_2026).fillna(0).values

    # Historical stats (2023-2025)
    location_hist_mean = df_features.groupby('location_name')['incident_count'].mean()
    pred_2026['hist_mean'] = pred_2026['location_name'].map(location_hist_mean).values
    pred_2026['hist_mean_log'] = np.log1p(pred_2026['hist_mean'].values)

    location_hist_max = df_features.groupby('location_name')['incident_count'].max()
    pred_2026['hist_max'] = pred_2026['location_name'].map(location_hist_max).values

    location_hist_min = df_features.groupby('location_name')['incident_count'].min()
    pred_2026['hist_min'] = pred_2026['location_name'].map(location_hist_min).values

    location_hist_std = df_features.groupby('location_name')['incident_count'].std().fillna(0)
    pred_2026['hist_std'] = pred_2026['location_name'].map(location_hist_std).values

    # Rolling mean 2y
    rolling_2y = (data_2024_inc.add(data_2025_inc, fill_value=0)) / 2
    pred_2026['rolling_mean_2y'] = pred_2026['location_name'].map(rolling_2y).fillna(0).values

    # incident_lag1_log
    pred_2026['incident_lag1_log'] = np.log1p(data_2025['incident_count'].values)

    # Seasonal features
    q_lag_cols = ['q1_count_lag1', 'q2_count_lag1', 'q3_count_lag1', 'q4_count_lag1']
    available_q = [c for c in q_lag_cols if c in pred_2026.columns]
    if len(available_q) >= 2:
        pred_2026['max_quarter_lag1'] = pred_2026[available_q].max(axis=1)
        pred_2026['seasonal_var_lag1'] = pred_2026[available_q].std(axis=1)
        pred_2026['q_ratio_max_lag1'] = pred_2026['max_quarter_lag1'] / (data_2025['incident_count'].values + 1)

    # NOTE: Do NOT recompute interaction/momentum/ratio features here.
    # The notebook only recomputes the features above; all others keep their
    # values from the initial loop (copied from data_2025).

    # Ensure all selected features exist
    X_2026 = pred_2026[selected_features].fillna(0)

    # Generate predictions — pure GB (notebook best_weight=0.0)
    pred_2026_log = gb_model.predict(X_2026)
    predictions_2026 = np.maximum(np.expm1(pred_2026_log), 0).round().astype(int)

    # Create output
    final_predictions = pd.DataFrame({
        'Location': data_2025['location_name'].values,
        'Predicted_Incidents_2026': predictions_2026,
    }).sort_values('Predicted_Incidents_2026', ascending=False)

    print(f"    Total locations: {len(final_predictions)}")
    print(f"    Total predicted incidents: {final_predictions['Predicted_Incidents_2026'].sum():,}")
    print(f"    Average per location: {final_predictions['Predicted_Incidents_2026'].mean():.1f}")
    print(f"\n    Top 10:")
    print(final_predictions.head(10).to_string(index=False))

    if output_path:
        final_predictions.to_csv(output_path, index=False)
        print(f"\n    Saved to: {output_path}")

    return final_predictions
