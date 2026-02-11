"""
Model Training Module
=====================
Trains a pure Gradient Boosting Regressor on log-transformed
incident counts with strict temporal train/test splitting.

Key design decisions:
    - Log transform on target: handles right-skewed distribution
    - Temporal split: model only ever sees past data (no random CV)
    - Heavy regularization: max_depth=3, lr=0.03, subsample=0.8
    - With only 232 training samples, a simple well-tuned model
      generalizes better than complex ensembles

Final performance: R²=0.9282, MAE=20.99, RMSE=30.56

Author: Nader Mohamed
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Train and manage the Gradient Boosting prediction model.

    Workflow:
        trainer = ModelTrainer()
        trainer.prepare_data(df_features, feature_cols)
        model = trainer.train_gradient_boosting()
        importance = trainer.get_feature_importance()
    """

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_raw = None     # before log transform
        self.y_test_raw = None      # before log transform
        self.model = None
        self.use_log_transform = True

    # ------------------------------------------------------------------
    # Data Preparation (temporal split)
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df_features: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'incident_count',
        test_year: Optional[int] = None,
        log_transform: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Strict temporal train/test split.

        Training set: all years before test_year (after lag warm-up).
        Test set: test_year only.
        Target is log-transformed to handle skewed distribution.
        """
        print("[*] Preparing temporal train/test split...")

        self.use_log_transform = log_transform

        available_years = sorted(df_features['year'].unique())
        min_year = min(available_years) + 1  # need >= 1 lag year

        df_model = df_features[df_features['year'] >= min_year].copy()

        if test_year is None:
            test_year = max(available_years)

        train_years = [y for y in available_years if min_year <= y < test_year]

        print(f"    Train years: {train_years}")
        print(f"    Test year:   {test_year}")

        train = df_model[df_model['year'].isin(train_years)]
        test = df_model[df_model['year'] == test_year]

        self.X_train = train[feature_cols].fillna(0)
        self.X_test = test[feature_cols].fillna(0)

        # Store raw target for evaluation
        self.y_train_raw = train[target_col].values
        self.y_test_raw = test[target_col].values

        # Log-transform target
        if log_transform:
            self.y_train = np.log1p(train[target_col])
            self.y_test = np.log1p(test[target_col])
            print("    Target: log-transformed (key technique)")
        else:
            self.y_train = train[target_col]
            self.y_test = test[target_col]

        print(f"    Train samples: {len(self.X_train)}")
        print(f"    Test samples:  {len(self.X_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------
    # Model Training
    # ------------------------------------------------------------------

    def train_gradient_boosting(
        self,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        min_samples_leaf: int = 4,
        random_state: int = 42,
    ) -> GradientBoostingRegressor:
        """
        Train a heavily regularized Gradient Boosting model.

        Default hyperparameters are the final tuned values.
        Regularization is critical with small training sets (~232 samples).
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first.")

        print("\n[*] Training Gradient Boosting Regressor...")
        print(f"    n_estimators={n_estimators}, max_depth={max_depth}, "
              f"lr={learning_rate}, subsample={subsample}")

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.model.fit(self.X_train, self.y_train)

        # Evaluate on log-scale
        train_r2 = self.model.score(self.X_train, self.y_train)
        test_r2 = self.model.score(self.X_test, self.y_test)
        print(f"    Train R² (log-scale): {train_r2:.4f}")
        print(f"    Test  R² (log-scale): {test_r2:.4f}")

        # Evaluate on original scale
        if self.use_log_transform:
            preds_raw = np.expm1(self.model.predict(self.X_test))
            preds_raw = np.maximum(0, preds_raw)
            r2_raw = r2_score(self.y_test_raw, preds_raw)
            print(f"    Test  R² (original):  {r2_raw:.4f}")

        return self.model

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions (inverse-transforms if log was used).
        Returns non-negative rounded integers.
        """
        if self.model is None:
            raise ValueError("Train a model first.")

        preds = self.model.predict(X)

        if self.use_log_transform:
            preds = np.expm1(preds)

        return np.maximum(0, np.round(preds)).astype(int)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on log-scale (no inverse transform)."""
        if self.model is None:
            raise ValueError("Train a model first.")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Feature importances from the trained GB model."""
        if self.model is None:
            raise ValueError("Train a model first.")

        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, save_dir: str = 'models'):
        """Save trained model to disk."""
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, 'gradient_boosting.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'use_log_transform': self.use_log_transform,
                'feature_cols': list(self.X_train.columns),
            }, f)
        print(f"[+] Model saved to {path}")

    def load_model(self, load_dir: str = 'models'):
        """Load trained model from disk."""
        path = os.path.join(load_dir, 'gradient_boosting.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.use_log_transform = data['use_log_transform']
        print(f"[+] Model loaded from {path}")


# Convenience function
def train_model(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'incident_count',
) -> Tuple['ModelTrainer', GradientBoostingRegressor]:
    """One-liner: prepare data, train GB, return trainer + model."""
    trainer = ModelTrainer()
    trainer.prepare_data(df_features, feature_cols, target_col)
    model = trainer.train_gradient_boosting()
    return trainer, model
