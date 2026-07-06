"""
Feature Engineering — matches notebook cells 8, 9, 10, 11, 12 exactly.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:

    def __init__(self, df_clean: pd.DataFrame):
        self.df_clean = df_clean
        self.location_year = None
        self.df_features = None
        self._selected_features = None

    # ==========================================================
    # Step 1: Aggregate to location × year (notebook cell 8)
    # ==========================================================

    def aggregate_location_year(self) -> pd.DataFrame:
        print("[*] Aggregating by location × year...")

        df = self.df_clean

        location_year = df.groupby(['location_name', 'year']).agg(
            incident_count=('g_date', 'count'),
            injury_count=('injury_type_name', lambda x: (x != 'No Injury').sum()),
            damage_count=('vehicle_damage_name', lambda x: (x != 'No Damage').sum()),
            collision_count=('vehicle_collision_type_name', lambda x: (x != 'No Collision').sum()),
        ).reset_index()

        # Quarterly breakdown
        quarterly_pivot = df.groupby(['location_name', 'year', 'quarter']).size().unstack(fill_value=0)
        quarterly_pivot.columns = [f'q{q}_count' for q in quarterly_pivot.columns]
        quarterly_pivot = quarterly_pivot.reset_index()
        location_year = location_year.merge(quarterly_pivot, on=['location_name', 'year'], how='left')

        # Incident type breakdown
        type_pivot = df.groupby(['location_name', 'year', 'incident_type_name']).size().unstack(fill_value=0)
        type_pivot.columns = [f'type_{c.replace(" ", "_").lower()}' for c in type_pivot.columns]
        type_pivot = type_pivot.reset_index()
        location_year = location_year.merge(type_pivot, on=['location_name', 'year'], how='left')

        location_year = location_year.fillna(0)

        self.location_year = location_year
        print(f"    {len(location_year)} location-year rows")
        return location_year

    # ==========================================================
    # Step 2: Create features (notebook cell 9 — exact copy)
    # ==========================================================

    def create_features(self) -> pd.DataFrame:
        if self.location_year is None:
            raise ValueError("Call aggregate_location_year() first")

        print("[*] Creating features...")
        df = self.location_year

        locations = df['location_name'].unique()
        years = sorted(df['year'].unique())

        # Complete location-year grid
        grid = pd.DataFrame(
            [(loc, yr) for loc in locations for yr in years],
            columns=['location_name', 'year'],
        )
        df_full = grid.merge(df, on=['location_name', 'year'], how='left')
        numeric_cols = df_full.select_dtypes(include=[np.number]).columns
        df_full[numeric_cols] = df_full[numeric_cols].fillna(0)
        df_full = df_full.sort_values(['location_name', 'year']).reset_index(drop=True)

        # === LAG FEATURES (1-2 only, matching notebook) ===
        for lag in [1, 2]:
            df_full[f'incident_lag{lag}'] = df_full.groupby('location_name')['incident_count'].shift(lag)
            df_full[f'injury_lag{lag}'] = df_full.groupby('location_name')['injury_count'].shift(lag)
            df_full[f'damage_lag{lag}'] = df_full.groupby('location_name')['damage_count'].shift(lag)
            df_full[f'collision_lag{lag}'] = df_full.groupby('location_name')['collision_count'].shift(lag)

        # === RATE FEATURES ===
        df_full['damage_rate_lag1'] = df_full['damage_lag1'] / (df_full['incident_lag1'] + 1)
        df_full['injury_rate_lag1'] = df_full['injury_lag1'] / (df_full['incident_lag1'] + 1)
        df_full['collision_rate_lag1'] = df_full['collision_lag1'] / (df_full['incident_lag1'] + 1)

        # === GROWTH FEATURES ===
        df_full['growth_rate'] = (df_full['incident_lag1'] - df_full['incident_lag2']) / (df_full['incident_lag2'] + 1)
        df_full['trend'] = df_full.groupby('location_name')['incident_count'].diff().shift(1)

        # === ROLLING / HISTORICAL STATISTICS ===
        df_full['rolling_mean_2y'] = df_full.groupby('location_name')['incident_count'].transform(
            lambda x: x.rolling(2, min_periods=1).mean().shift(1)
        )
        df_full['hist_mean'] = df_full.groupby('location_name')['incident_count'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df_full['hist_max'] = df_full.groupby('location_name')['incident_count'].transform(
            lambda x: x.expanding().max().shift(1)
        )
        df_full['hist_std'] = df_full.groupby('location_name')['incident_count'].transform(
            lambda x: x.expanding().std().shift(1)
        )
        df_full['hist_min'] = df_full.groupby('location_name')['incident_count'].transform(
            lambda x: x.expanding().min().shift(1)
        )

        # === LOG FEATURES ===
        df_full['incident_log'] = np.log1p(df_full['incident_count'])
        df_full['incident_lag1_log'] = np.log1p(df_full['incident_lag1'])
        df_full['hist_mean_log'] = df_full.groupby('location_name')['incident_log'].transform(
            lambda x: x.expanding().mean().shift(1)
        )

        # === SEASONAL FEATURES (lagged quarterly counts) ===
        for q in ['q1_count', 'q2_count', 'q3_count', 'q4_count']:
            if q in df_full.columns:
                df_full[f'{q}_lag1'] = df_full.groupby('location_name')[q].shift(1)

        q_lag_cols = ['q1_count_lag1', 'q2_count_lag1', 'q3_count_lag1', 'q4_count_lag1']
        df_full['max_quarter_lag1'] = df_full[q_lag_cols].max(axis=1)
        df_full['seasonal_var_lag1'] = df_full[q_lag_cols].std(axis=1)
        df_full['q_ratio_max_lag1'] = df_full['max_quarter_lag1'] / (df_full['incident_lag1'] + 1)

        # === LAGGED INCIDENT TYPE FEATURES ===
        type_cols_in_df = [c for c in df_full.columns if c.startswith('type_')]
        for tc in type_cols_in_df:
            df_full[f'{tc}_lag1'] = df_full.groupby('location_name')[tc].shift(1)

        # === INTERACTION FEATURES ===
        df_full['trend_x_lag1'] = df_full['trend'] * df_full['incident_lag1']
        df_full['trend_x_lag1_log'] = df_full['trend'] * df_full['incident_lag1_log']
        df_full['trend_x_hist_mean'] = df_full['trend'] * df_full['hist_mean']
        df_full['trend_positive'] = (df_full['trend'] > 0).astype(int)
        df_full['trend_abs'] = df_full['trend'].abs()

        # === MOMENTUM FEATURES ===
        df_full['acceleration'] = df_full.groupby('location_name')['trend'].diff().shift(0)
        df_full['weighted_trend'] = (
            0.7 * df_full['trend']
            + 0.3 * df_full.groupby('location_name')['incident_count'].diff(2).shift(1).fillna(0)
        )

        # === RATIO FEATURES ===
        df_full['lag1_to_hist_mean'] = df_full['incident_lag1'] / (df_full['hist_mean'] + 1)
        df_full['lag1_to_hist_max'] = df_full['incident_lag1'] / (df_full['hist_max'] + 1)
        df_full['range_normalized'] = (
            (df_full['incident_lag1'] - df_full['hist_min'])
            / (df_full['hist_max'] - df_full['hist_min'] + 1)
        )

        # === SEASONAL SHAPE FEATURES ===
        df_full['q1_share_lag1'] = df_full['q1_count_lag1'] / (df_full['incident_lag1'] + 1)
        df_full['q2_share_lag1'] = df_full['q2_count_lag1'] / (df_full['incident_lag1'] + 1)
        df_full['q3_share_lag1'] = df_full['q3_count_lag1'] / (df_full['incident_lag1'] + 1)
        df_full['q4_share_lag1'] = df_full['q4_count_lag1'] / (df_full['incident_lag1'] + 1)
        df_full['h1_vs_h2_lag1'] = (
            (df_full['q1_count_lag1'] + df_full['q2_count_lag1'])
            / (df_full['q3_count_lag1'] + df_full['q4_count_lag1'] + 1)
        )

        # === SQRT FEATURES ===
        df_full['incident_lag1_sqrt'] = np.sqrt(df_full['incident_lag1'])
        df_full['hist_mean_sqrt'] = np.sqrt(df_full['hist_mean'])

        # === VOLATILITY / STABILITY ===
        df_full['cv_lag1'] = df_full['hist_std'] / (df_full['hist_mean'] + 1)
        df_full['stability_score'] = 1 / (1 + df_full['hist_std'])

        # === DAMAGE SEVERITY COMPOSITE ===
        df_full['severity_index_lag1'] = (
            df_full['damage_rate_lag1'] + df_full['injury_rate_lag1'] + df_full['collision_rate_lag1']
        ) / 3

        df_full = df_full.fillna(0)
        self.df_features = df_full
        print(f"    Feature dataset shape: {df_full.shape}")
        return df_full

    # ==========================================================
    # Step 3: High-volume flag (notebook cell 10)
    # ==========================================================

    def add_high_volume_flag(self) -> pd.DataFrame:
        if self.df_features is None:
            raise ValueError("Call create_features() first")

        location_totals = self.df_features.groupby('location_name')['incident_count'].sum()
        threshold = location_totals.mean() + 2 * location_totals.std()
        high_volume = location_totals[location_totals > threshold].index.tolist()

        print(f"[*] High-volume locations ({len(high_volume)}): {high_volume}")
        self.df_features['is_high_volume'] = self.df_features['location_name'].isin(high_volume).astype(int)
        return self.df_features

    # ==========================================================
    # Step 4: Feature selection (notebook cells 11-12)
    # ==========================================================

    def get_feature_columns(self) -> List[str]:
        """Get all valid feature columns (excluding leakage)."""
        if self.df_features is None:
            raise ValueError("Call create_features() first")

        leakage_cols = [
            'injury_count', 'damage_count', 'collision_count',
            'q1_count', 'q2_count', 'q3_count', 'q4_count',
        ]
        type_cols_raw = [
            c for c in self.df_features.columns
            if c.startswith('type_') and '_lag1' not in c
        ]

        exclude_cols = (
            ['location_name', 'year', 'incident_count', 'incident_log']
            + leakage_cols + type_cols_raw
        )

        feature_cols = [
            c for c in self.df_features.columns
            if c not in exclude_cols and self.df_features[c].dtype in ['int64', 'float64']
        ]
        return feature_cols

    def select_top_features(self, n: int = 20) -> List[str]:
        """RF + ExtraTrees averaged importance (notebook cell 12)."""
        if self.df_features is None:
            raise ValueError("Build features first")

        feature_cols = self.get_feature_columns()
        print(f"[*] Selecting top {n} features from {len(feature_cols)} candidates...")

        # Use all years with valid data (years >= min+1 so lags exist)
        years = sorted(self.df_features['year'].unique())
        train_years = years[:-1]  # all except last
        train_data = self.df_features[self.df_features['year'].isin(train_years)]

        X = train_data[feature_cols]
        y = np.log1p(train_data['incident_count'])

        rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
        et = ExtraTreesRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        et.fit(X, y)

        avg_imp = (
            pd.Series(rf.feature_importances_, index=feature_cols)
            + pd.Series(et.feature_importances_, index=feature_cols)
        ) / 2
        avg_imp = avg_imp.sort_values(ascending=False)

        selected = avg_imp.head(n).index.tolist()
        self._selected_features = selected

        print(f"    Selected {len(selected)} features:")
        for i, f in enumerate(selected):
            print(f"      {i+1:2d}. {f} ({avg_imp[f]:.4f})")

        return selected

    # ==========================================================
    # Full pipeline
    # ==========================================================

    def build_all(self, n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING")
        print("=" * 60 + "\n")

        self.aggregate_location_year()
        self.create_features()
        self.add_high_volume_flag()
        feature_cols = self.get_feature_columns()
        selected = self.select_top_features(n=n_features)

        return self.df_features, selected
