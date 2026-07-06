"""
Incident Prediction Pipeline — main entry point.
Replicates the notebook (Incident_Prediction_Final.ipynb) exactly.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import DATA_PATH, OUTPUT_DIR, MIN_YEAR, MAX_YEAR, PREDICTION_YEAR, N_TOP_FEATURES
from src.data_loader import IncidentDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.prediction import predict_2026


def main():
    parser = argparse.ArgumentParser(description='Incident Prediction Pipeline')
    parser.add_argument('--data-path', default=DATA_PATH, help='Path to IncidentData.csv')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--year', type=int, default=PREDICTION_YEAR, help='Prediction year')
    parser.add_argument('--min-year', type=int, default=MIN_YEAR, help='Min year filter')
    parser.add_argument('--max-year', type=int, default=MAX_YEAR, help='Max year filter')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  INCIDENT PREDICTION PIPELINE")
    print("=" * 60)
    print(f"  Data:       {args.data_path}")
    print(f"  Year range: {args.min_year}-{args.max_year}")
    print(f"  Predict:    {args.year}")
    print("=" * 60)

    # 1 — Load & Clean
    loader = IncidentDataLoader(args.data_path)
    df_clean = loader.process_all(min_year=args.min_year, max_year=args.max_year)

    # 2 — Feature Engineering
    fe = FeatureEngineer(df_clean)
    df_features, selected_features = fe.build_all(n_features=N_TOP_FEATURES)

    # 3 — Train (temporal split: all-but-last for train, last for test)
    years = sorted(df_features['year'].unique())
    train_years = years[:-1]
    test_year = years[-1]

    gb_model, train_data, test_data = train_model(
        df_features, selected_features, train_years, test_year
    )

    # 4 — Evaluate
    metrics = evaluate_model(test_data)

    # 5 — Predict 2026
    output_csv = os.path.join(args.output_dir, f'predictions_{args.year}.csv')
    final_pred = predict_2026(df_features, gb_model, selected_features, output_path=output_csv)

    # 6 — Validation subset (Q1 partial)
    validation_csv = os.path.join(args.output_dir, f'validation_{args.year}_partial.csv')
    final_pred.to_csv(validation_csv, index=False)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Predictions:  {output_csv}")
    print(f"  Validation:   {validation_csv}")
    print("=" * 60)


if __name__ == '__main__':
    main()
