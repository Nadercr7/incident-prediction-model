"""
Incident Prediction Model — Main Pipeline
==========================================
Entry point for the complete prediction workflow:

    1. Load & preprocess incident data
    2. Engineer 56 features across 4 categories
    3. Select top 20 features (RF + ExtraTrees averaged)
    4. Train Gradient Boosting with log-transformed target
    5. Evaluate on held-out year (temporal split)
    6. Generate predictions for the next year
    7. Export results

Usage:
    python main.py                              # defaults
    python main.py --data-path data/my.csv      # custom data
    python main.py --year 2027                   # different year
    python main.py --config                      # view settings

Author: Nader Mohamed
Date: February 2026
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'config'))

from data_loader import IncidentDataLoader
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from prediction import PredictionGenerator
from config import (
    DATA_FILE, OUTPUT_DIR, MODELS_DIR,
    PREDICTION_YEAR, USE_LOG_TRANSFORM,
    GRADIENT_BOOSTING_PARAMS, print_config,
)


def print_banner():
    print("\n" + "=" * 60)
    print("  INCIDENT PREDICTION MODEL")
    print("=" * 60)
    print(f"  Version 2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)


def run_pipeline(
    data_path: str = None,
    output_path: str = None,
    prediction_year: int = 2026,
    n_features: int = 20,
    save_model: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run the complete prediction pipeline.

    Returns dict with keys: data_shape, feature_count, metrics,
    top_features, prediction_stats.
    """
    results = {}

    data_path = data_path or str(DATA_FILE)
    output_path = output_path or str(OUTPUT_DIR / f'predictions_{prediction_year}.csv')

    # ----- 1. LOAD DATA -----
    print("\n" + "-" * 60)
    print("  STEP 1: LOAD & PREPROCESS DATA")
    print("-" * 60)

    loader = IncidentDataLoader(data_path)
    df = loader.process_all()

    results['data_shape'] = df.shape
    results['date_range'] = loader.get_date_range()

    if verbose:
        print(f"\n  Records: {df.shape[0]:,}")
        d0, d1 = results['date_range']
        print(f"  Range:   {d0.date()} to {d1.date()}")

    # ----- 2. FEATURE ENGINEERING -----
    print("\n" + "-" * 60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("-" * 60)

    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features()
    feature_cols = engineer.select_top_features(n=n_features)

    results['feature_count'] = len(feature_cols)
    results['total_locations'] = df_features['location_name'].nunique()

    if verbose:
        print(f"\n  Features selected: {len(feature_cols)} (from 56 engineered)")
        print(f"  Locations:         {results['total_locations']}")

    # ----- 3. TRAIN MODEL -----
    print("\n" + "-" * 60)
    print("  STEP 3: TRAIN MODEL")
    print("-" * 60)

    trainer = ModelTrainer()
    trainer.prepare_data(
        df_features, feature_cols,
        log_transform=USE_LOG_TRANSFORM,
    )
    trainer.train_gradient_boosting(**GRADIENT_BOOSTING_PARAMS)

    if save_model:
        trainer.save_model(str(MODELS_DIR))

    # ----- 4. EVALUATE -----
    print("\n" + "-" * 60)
    print("  STEP 4: EVALUATE")
    print("-" * 60)

    test_preds = trainer.predict(trainer.X_test)
    evaluator = ModelEvaluator(trainer.y_test_raw, test_preds)
    metrics = evaluator.calculate_all_metrics()
    evaluator.print_report("Gradient Boosting")

    results['metrics'] = metrics

    importance = trainer.get_feature_importance()
    results['top_features'] = importance.head(10).to_dict('records')

    if verbose:
        print("\n  Top 10 Features:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['feature']:30s}  {row['importance']:.4f}")

    # Save evaluation plots
    eval_dir = OUTPUT_DIR / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    evaluator.create_full_report(
        "Gradient Boosting",
        feature_importance=importance,
        save_dir=str(eval_dir),
    )

    # ----- 5. PREDICT -----
    print("\n" + "-" * 60)
    print(f"  STEP 5: GENERATE {prediction_year} PREDICTIONS")
    print("-" * 60)

    generator = PredictionGenerator(df_features, feature_cols, trainer)
    predictions = generator.generate_predictions(prediction_year)
    generator.save_predictions(output_path)
    generator.print_summary()

    results['prediction_stats'] = generator.get_summary_stats()
    results['distribution'] = generator.get_distribution_tiers()

    # ----- SUMMARY -----
    m = metrics
    s = results['prediction_stats']
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"""
  Data:        {results['data_shape'][0]:,} incidents processed
  Features:    {results['feature_count']} selected (from 56 engineered)
  Model:       R²={m['R2']:.4f}, MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}
  Predictions: {s['total']:,} total across {s['locations']} locations
  Output:      {output_path}
""")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Incident Prediction Model — forecast incident counts by location'
    )
    parser.add_argument('--data-path', '-d', type=str, default=None,
                        help='Path to input CSV data file')
    parser.add_argument('--output-path', '-o', type=str, default=None,
                        help='Path to save predictions CSV')
    parser.add_argument('--year', '-y', type=int, default=PREDICTION_YEAR,
                        help=f'Year to predict (default: {PREDICTION_YEAR})')
    parser.add_argument('--features', '-f', type=int, default=20,
                        help='Number of features to select (default: 20)')
    parser.add_argument('--no-save-model', action='store_true',
                        help='Do not save trained model to disk')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--config', action='store_true',
                        help='Print configuration and exit')

    args = parser.parse_args()

    if args.config:
        print_config()
        return

    print_banner()

    run_pipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        prediction_year=args.year,
        n_features=args.features,
        save_model=not args.no_save_model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
