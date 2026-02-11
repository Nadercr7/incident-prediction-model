"""
Model Evaluation Module
=======================
Comprehensive evaluation metrics and visualization for the
incident prediction model.

Metrics:
    - MAE, RMSE, R², Median AE
    - sMAPE (symmetric — handles low-count locations fairly)
    - Baseline comparison (mean predictor)

Visualizations:
    - Actual vs Predicted scatter
    - Residuals distribution
    - Feature importance bar chart

Author: Nader Mohamed
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
)
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evaluate model predictions against actual values.

    Usage:
        evaluator = ModelEvaluator(y_true, y_pred)
        metrics = evaluator.calculate_all_metrics()
        evaluator.print_report("Gradient Boosting")
    """

    def __init__(self, y_true: np.ndarray = None, y_pred: np.ndarray = None):
        self.y_true = np.array(y_true) if y_true is not None else None
        self.y_pred = np.array(y_pred) if y_pred is not None else None
        self.metrics = {}
        plt.style.use('seaborn-v0_8-whitegrid')

    def set_data(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Set actual and predicted values."""
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.metrics = {}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_mae(self) -> float:
        return mean_absolute_error(self.y_true, self.y_pred)

    def calculate_rmse(self) -> float:
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def calculate_r2(self) -> float:
        return r2_score(self.y_true, self.y_pred)

    def calculate_median_ae(self) -> float:
        return median_absolute_error(self.y_true, self.y_pred)

    def calculate_smape(self) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        Preferred over MAPE for count data — handles zeros and
        doesn't disproportionately penalize small actual values.
        """
        denom = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        mask = denom > 0
        if mask.sum() == 0:
            return 0.0
        smape = np.mean(np.abs(self.y_true[mask] - self.y_pred[mask]) / denom[mask]) * 100
        return smape

    def calculate_baseline_mae(self) -> float:
        """MAE of a naive mean predictor (baseline)."""
        mean_pred = np.full_like(self.y_true, self.y_true.mean(), dtype=float)
        return mean_absolute_error(self.y_true, mean_pred)

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate and return all metrics."""
        self.metrics = {
            'R2': self.calculate_r2(),
            'MAE': self.calculate_mae(),
            'Median_AE': self.calculate_median_ae(),
            'RMSE': self.calculate_rmse(),
            'sMAPE': self.calculate_smape(),
            'Baseline_MAE': self.calculate_baseline_mae(),
        }
        improvement = (1 - self.metrics['MAE'] / self.metrics['Baseline_MAE']) * 100
        self.metrics['Improvement_vs_Baseline'] = improvement
        return self.metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, model_name: str = "Model"):
        """Print formatted evaluation report."""
        if not self.metrics:
            self.calculate_all_metrics()

        m = self.metrics
        print("\n" + "=" * 60)
        print(f"  {model_name.upper()} — EVALUATION REPORT")
        print("=" * 60)
        print(f"  R² Score:         {m['R2']:.4f}  ({m['R2']*100:.1f}% variance explained)")
        print(f"  MAE:              {m['MAE']:.2f} incidents")
        print(f"  Median AE:        {m['Median_AE']:.2f} incidents")
        print(f"  RMSE:             {m['RMSE']:.2f}")
        print(f"  sMAPE:            {m['sMAPE']:.2f}%")
        print(f"  Baseline MAE:     {m['Baseline_MAE']:.2f} (mean predictor)")
        print(f"  Improvement:      {m['Improvement_vs_Baseline']:.1f}% lower error vs baseline")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def plot_actual_vs_predicted(self, save_path: Optional[str] = None):
        """Scatter plot of actual vs predicted values."""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(self.y_true, self.y_pred, alpha=0.5, color='steelblue',
                   edgecolors='navy', linewidth=0.5)

        max_val = max(self.y_true.max(), self.y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Incident Count', fontsize=12)
        ax.set_ylabel('Predicted Incident Count', fontsize=12)
        ax.set_title('Actual vs Predicted Incidents', fontsize=14, fontweight='bold')
        ax.legend()

        r2 = self.metrics.get('R2', self.calculate_r2())
        ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.93), xycoords='axes fraction',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Plot saved: {save_path}")
        return fig

    def plot_residuals(self, save_path: Optional[str] = None):
        """Residuals histogram + residuals vs predicted."""
        residuals = self.y_true - self.y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(residuals, bins=30, color='coral', edgecolor='darkred', alpha=0.7)
        axes[0].axvline(x=0, color='black', linestyle='--', lw=2)
        axes[0].set_xlabel('Residual (Actual - Predicted)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Residuals Distribution', fontweight='bold')

        axes[1].scatter(self.y_pred, residuals, alpha=0.5, color='steelblue')
        axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Value')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals vs Predicted', fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Plot saved: {save_path}")
        return fig

    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                                 top_n: int = 15, save_path: Optional[str] = None):
        """Horizontal bar chart of top-N feature importances."""
        fig, ax = plt.subplots(figsize=(10, 8))

        top = feature_importance.head(top_n)
        ax.barh(top['feature'][::-1], top['importance'][::-1],
                color='seagreen', edgecolor='darkgreen')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[+] Plot saved: {save_path}")
        return fig

    def create_full_report(self, model_name: str = "Model",
                            feature_importance: Optional[pd.DataFrame] = None,
                            save_dir: Optional[str] = None) -> Dict:
        """Generate metrics + all plots."""
        self.calculate_all_metrics()
        self.print_report(model_name)

        figs = {}
        figs['actual_vs_predicted'] = self.plot_actual_vs_predicted(
            save_path=f"{save_dir}/actual_vs_predicted.png" if save_dir else None
        )
        figs['residuals'] = self.plot_residuals(
            save_path=f"{save_dir}/residuals.png" if save_dir else None
        )
        if feature_importance is not None:
            figs['feature_importance'] = self.plot_feature_importance(
                feature_importance,
                save_path=f"{save_dir}/feature_importance.png" if save_dir else None
            )

        return {'metrics': self.metrics, 'figures': figs}
