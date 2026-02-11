"""
Prediction Module
=================
Generates predictions for future periods using the trained
Gradient Boosting model.

Handles:
    - Building lag/trend/interaction features for the prediction year
    - Inverse log-transform to get real incident counts
    - Summary statistics and distribution analysis
    - CSV export with anonymized location identifiers

Author: Nader Mohamed
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import os
import warnings

warnings.filterwarnings('ignore')


class PredictionGenerator:
    """
    Generate predictions for a future year using historical features
    and the trained model.

    Usage:
        generator = PredictionGenerator(df_features, feature_cols, trainer)
        predictions = generator.generate_predictions(2026)
        generator.save_predictions("output/predictions_2026.csv")
    """

    def __init__(self, df_features: pd.DataFrame, feature_cols: List[str],
                 trainer):
        """
        Args:
            df_features: Historical feature DataFrame
            feature_cols: Selected feature column names
            trainer: Trained ModelTrainer instance (has .predict() method)
        """
        self.df_features = df_features
        self.feature_cols = feature_cols
        self.trainer = trainer
        self.predictions_df = None

    # ------------------------------------------------------------------
    # Feature Construction for Future Year
    # ------------------------------------------------------------------

    def create_future_features(self, prediction_year: int = 2026) -> pd.DataFrame:
        """
        Build the feature vector for each location for the prediction year.

        Uses the latest available data to shift forward:
            - Current year's count → lag1
            - Last year's lag1 → lag2
            - etc.
        """
        print(f"\n[*] Building features for {prediction_year}...")

        locations = self.df_features['location_name'].unique()
        latest_year = self.df_features['year'].max()

        pred_df = pd.DataFrame({
            'location_name': locations,
            'year': prediction_year,
        })

        for loc in locations:
            loc_data = self.df_features[
                self.df_features['location_name'] == loc
            ].sort_values('year')

            if len(loc_data) == 0:
                continue

            latest = loc_data[loc_data['year'] == latest_year]
            if len(latest) == 0:
                latest = loc_data.iloc[[-1]]
            latest = latest.iloc[0]

            mask = pred_df['location_name'] == loc

            # --- Lag features ---
            pred_df.loc[mask, 'incident_count_lag1'] = latest.get('incident_count', 0)
            pred_df.loc[mask, 'incident_count_lag2'] = latest.get('incident_count_lag1', 0)
            pred_df.loc[mask, 'incident_count_lag3'] = latest.get('incident_count_lag2', 0)

            pred_df.loc[mask, 'injury_count_lag1'] = latest.get('injury_count', 0)
            pred_df.loc[mask, 'damage_count_lag1'] = latest.get('damage_count', 0)
            pred_df.loc[mask, 'collision_count_lag1'] = latest.get('collision_count', 0)

            # --- Transformed lags ---
            lag1 = latest.get('incident_count', 0)
            lag2 = latest.get('incident_count_lag1', 0)
            pred_df.loc[mask, 'lag1_log'] = np.log1p(lag1)
            pred_df.loc[mask, 'lag1_sqrt'] = np.sqrt(max(0, lag1))
            pred_df.loc[mask, 'lag1_squared'] = lag1 ** 2
            pred_df.loc[mask, 'lag1_cubed'] = lag1 ** 3
            pred_df.loc[mask, 'lag2_log'] = np.log1p(lag2)

            # --- Rolling averages ---
            lag3 = latest.get('incident_count_lag2', 0)
            pred_df.loc[mask, 'incident_rolling_avg_2y'] = (lag1 + lag2) / 2
            pred_df.loc[mask, 'incident_rolling_avg_3y'] = (lag1 + lag2 + lag3) / 3

            # --- Trend ---
            trend = latest.get('incident_trend', 0)
            pred_df.loc[mask, 'incident_trend'] = trend
            pred_df.loc[mask, 'trend_acceleration'] = latest.get('trend_acceleration', 0)
            pred_df.loc[mask, 'yoy_pct_change'] = latest.get('yoy_pct_change', 0)

            # --- Historical stats ---
            for col in ['hist_mean_incidents', 'hist_std_incidents',
                        'hist_max_incidents', 'hist_min_incidents',
                        'hist_total_incidents',
                        'hist_avg_injury_rate', 'hist_avg_damage_rate',
                        'hist_avg_collision_rate', 'cv']:
                pred_df.loc[mask, col] = latest.get(col, 0)

            # --- Interaction features ---
            hist_mean = latest.get('hist_mean_incidents', 0)
            hist_max = latest.get('hist_max_incidents', 0)
            pred_df.loc[mask, 'weighted_trend'] = trend * np.log1p(lag1)
            pred_df.loc[mask, 'trend_magnitude'] = abs(trend) * lag1
            pred_df.loc[mask, 'trend_x_lag1'] = trend * lag1
            pred_df.loc[mask, 'lag1_x_mean'] = lag1 * hist_mean
            pred_df.loc[mask, 'lag1_to_mean_ratio'] = lag1 / (hist_mean + 1)
            pred_df.loc[mask, 'lag1_to_max_ratio'] = lag1 / (hist_max + 1)

            # --- Seasonal ---
            for col in ['peak_quarter_count', 'seasonal_variance',
                        'h1_proportion', 'q4_proportion']:
                pred_df.loc[mask, col] = latest.get(col, 0)

            # --- Momentum ---
            pred_df.loc[mask, 'lag_diff_12'] = lag1 - lag2
            pred_df.loc[mask, 'lag_diff_23'] = lag2 - lag3

            # --- Flags ---
            pred_df.loc[mask, 'high_volume_flag'] = int(lag1 > hist_mean * 1.5)
            pred_df.loc[mask, 'low_volume_flag'] = int(lag1 < hist_mean * 0.5)

        # Clean up
        pred_df = pred_df.replace([np.inf, -np.inf], 0).fillna(0)
        print(f"    Built features for {len(pred_df)} locations")

        return pred_df

    # ------------------------------------------------------------------
    # Generate Predictions
    # ------------------------------------------------------------------

    def generate_predictions(self, prediction_year: int = 2026) -> pd.DataFrame:
        """Generate and store predictions for the given year."""
        print(f"\n[*] Generating {prediction_year} predictions...")

        pred_df = self.create_future_features(prediction_year)

        # Ensure all required feature columns exist
        for col in self.feature_cols:
            if col not in pred_df.columns:
                pred_df[col] = 0

        X_pred = pred_df[self.feature_cols]

        # Predict (trainer handles log inverse-transform)
        predictions = self.trainer.predict(X_pred)

        self.predictions_df = pd.DataFrame({
            'Location': pred_df['location_name'],
            f'Predicted_Incidents_{prediction_year}': predictions,
        }).sort_values(
            f'Predicted_Incidents_{prediction_year}', ascending=False
        ).reset_index(drop=True)

        total = predictions.sum()
        print(f"    Predictions generated for {len(self.predictions_df)} locations")
        print(f"    Total predicted incidents: {total:,}")

        return self.predictions_df

    # ------------------------------------------------------------------
    # Export & Summary
    # ------------------------------------------------------------------

    def save_predictions(self, output_path: str):
        """Save predictions to CSV."""
        if self.predictions_df is None:
            raise ValueError("Call generate_predictions() first.")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        self.predictions_df.to_csv(output_path, index=False)
        print(f"[+] Predictions saved: {output_path}")

    def get_summary_stats(self) -> Dict[str, float]:
        """Summary statistics of predictions."""
        if self.predictions_df is None:
            raise ValueError("Call generate_predictions() first.")

        pred_col = [c for c in self.predictions_df.columns if 'Predicted' in c][0]
        vals = self.predictions_df[pred_col]

        return {
            'total': int(vals.sum()),
            'locations': len(vals),
            'mean': round(vals.mean(), 1),
            'median': round(vals.median(), 1),
            'std': round(vals.std(), 1),
            'min': int(vals.min()),
            'max': int(vals.max()),
        }

    def get_distribution_tiers(self) -> Dict[str, int]:
        """Count locations in each prediction tier."""
        if self.predictions_df is None:
            raise ValueError("Call generate_predictions() first.")

        pred_col = [c for c in self.predictions_df.columns if 'Predicted' in c][0]
        vals = self.predictions_df[pred_col]

        return {
            '200+ incidents': int((vals >= 200).sum()),
            '100-199 incidents': int(((vals >= 100) & (vals < 200)).sum()),
            '50-99 incidents': int(((vals >= 50) & (vals < 100)).sum()),
            '20-49 incidents': int(((vals >= 20) & (vals < 50)).sum()),
            '<20 incidents': int((vals < 20).sum()),
        }

    def print_summary(self, n_top: int = 10):
        """Print prediction summary (no location names for privacy)."""
        if self.predictions_df is None:
            raise ValueError("Call generate_predictions() first.")

        stats = self.get_summary_stats()
        tiers = self.get_distribution_tiers()

        print("\n" + "=" * 50)
        print("  PREDICTION SUMMARY")
        print("=" * 50)
        print(f"  Locations:   {stats['locations']}")
        print(f"  Total:       {stats['total']:,} predicted incidents")
        print(f"  Average:     {stats['mean']} per location")
        print(f"  Median:      {stats['median']}")
        print(f"  Range:       {stats['min']} - {stats['max']}")
        print()
        print("  Distribution:")
        for tier, count in tiers.items():
            print(f"    {tier:20s}  {count} locations")
        print("=" * 50)


# Convenience function
def generate_predictions(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    trainer,
    prediction_year: int = 2026,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """One-liner to generate and optionally save predictions."""
    gen = PredictionGenerator(df_features, feature_cols, trainer)
    predictions = gen.generate_predictions(prediction_year)
    gen.print_summary()

    if output_path:
        gen.save_predictions(output_path)

    return predictions
