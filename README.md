# Incident Count Prediction Model

**Time-series regression model predicting operational incident counts across 116 locations using Gradient Boosting — built on real-world data during a Data Science internship at Zetta Global.**

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9282 (93% variance explained) |
| **MAE** | 20.99 incidents |
| **Median AE** | 14.00 incidents |
| **RMSE** | 30.56 |
| **Baseline MAE** | 64.54 (mean predictor) |
| **Improvement** | 67% lower error vs baseline |

---

## Problem Statement

A transportation company needed to forecast incident volumes per location for the upcoming year to enable proactive resource allocation, staffing decisions, and risk mitigation — shifting from reactive response to data-driven prevention.

**Key challenges:**
- Highly skewed distribution (top location had 10x the median count)
- Only 3 years of usable historical data (232 training samples)
- Risk of data leakage from same-year aggregate features

---

## Approach

### Pipeline Overview

```
Raw Data → Clean & Preprocess → Feature Engineering → Feature Selection → Model Training → Validation → Predictions
```

### 1. Data Processing
- 20,000+ real incident records across 3 years
- 116 unique locations
- Temporal feature extraction, binary flags, missing value handling

### 2. Feature Engineering (56 features → 20 selected)

| Category | Description | Impact |
|----------|-------------|--------|
| **Interaction Features** | Trend × magnitude, weighted trends, ratio features | Highest impact — key breakthrough |
| **Lag Features** | Prior-year counts (raw, log, sqrt), damage/collision rates | Core predictors |
| **Seasonal Patterns** | Quarterly distributions, peak quarter, half-year proportions | Temporal signal |
| **Historical Statistics** | Rolling averages, min/max/mean, trend acceleration | Stability signal |

Feature selection via averaged Random Forest + ExtraTrees importance scores.

### 3. Model: Pure Gradient Boosting

After testing Random Forest, Ridge, SVR, ensembles, and stacking — a single, heavily regularized Gradient Boosting model outperformed all combinations:

```python
GradientBoostingRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    min_samples_leaf=4,
    random_state=42
)
```

**Why this works:** With only 232 training samples, a simple well-tuned model generalizes better than complex ensembles.

### 4. Validation Strategy

- **Strict temporal split** — model only ever sees the past (no random CV)
- **Log-transformed target** to handle right-skewed distribution
- **Data leakage detection** — caught and removed same-year features with 99%+ target correlation

---

## Top Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Weighted Trend | 18.4% |
| 2 | Trend Magnitude | 15.0% |
| 3 | Peak Quarter Count | 13.4% |
| 4 | Key Incident Type (lag) | 12.8% |
| 5 | Year-over-Year Trend | 8.2% |
| 6 | Seasonal Variance | 6.3% |
| 7 | High-Volume Flag | 5.1% |

---

## Prediction Summary (2026)

| Metric | Value |
|--------|-------|
| Total Predicted Incidents | 8,744 |
| Locations Covered | 116 |
| Average per Location | ~75 |
| Distribution: 200+ incidents | 6 sites |
| Distribution: 100–199 | 17 sites |
| Distribution: 50–99 | 38 sites |
| Distribution: 20–49 | 29 sites |
| Distribution: < 20 | 26 sites |

---

## Project Structure

```
incident-prediction-model/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.py              # All hyperparameters and paths
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── feature_engineering.py # 56 features across 4 categories
│   ├── model_training.py      # GB training with temporal split
│   ├── model_evaluation.py    # Metrics, plots, baseline comparison
│   └── prediction.py          # Future-year prediction generation
│
├── main.py                    # Full pipeline entry point
│
├── data/
│   └── sample_data.csv        # Synthetic sample (real data excluded)
│
├── output/                    # Generated predictions (gitignored)
└── models/                    # Saved models (gitignored)
```

---

## Setup & Usage

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/incident-prediction-model.git
cd incident-prediction-model
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline with default settings
python main.py

# Custom data path
python main.py --data-path path/to/your/data.csv

# Predict for a different year
python main.py --year 2027

# View configuration
python main.py --config
```

### Use Individual Modules

```python
from src.data_loader import IncidentDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.prediction import PredictionGenerator

# Load → Engineer → Train → Predict
loader = IncidentDataLoader("data/incidents.csv")
df = loader.process_all()

engineer = FeatureEngineer(df)
df_features = engineer.build_all_features()
features = engineer.get_feature_columns()

trainer = ModelTrainer()
trainer.prepare_data(df_features, features)
model = trainer.train_gradient_boosting()

generator = PredictionGenerator(df_features, features, model)
predictions = generator.generate_predictions(2026)
generator.save_predictions("output/predictions_2026.csv")
```

---

## Key Learnings

1. **Feature engineering > model complexity** — The jump from ~80% to 93% R² came from interaction features (weighted trends, trend × lag), not from more complex algorithms.

2. **Temporal validation is non-negotiable** — Random splits in time-series data inflate metrics. Strict past→future split ensures honest, deployable results.

3. **Data leakage detection saved the project** — Same-year aggregate features (damage count, collision count) had 99%+ correlation with the target. Without catching this, the model would appear R²=0.99 but fail completely in production.

4. **Small data ≠ bad model** — With only 232 training samples, careful regularization (shallow trees, low learning rate, subsampling) achieved strong generalization.

---

## Limitations

- **Limited history**: 3 years of data constrains lag feature depth
- **No external factors**: Weather, policy changes, economic conditions not included
- **Annual granularity**: Monthly/weekly predictions would require more data
- **Point predictions**: No confidence intervals (could add with quantile regression)
- **sMAPE = 56.4%**: Inflated by low-count locations where ±5 incidents = large percentage error

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **Model** — Gradient Boosting Regressor
- **Validation** — Temporal train/test split
- **Visualization** — Matplotlib, Seaborn

---

## Privacy Notice

This project was completed during a Data Science internship at **Zetta Global**. All location names, employee data, and sensitive operational details have been removed. Only methodology, aggregate statistics, and anonymized results are shared. The raw dataset (`IncidentData.csv`) is not included in this repository.

---

## Author

**Nader Mohamed**
Data Science Intern · Zetta Global · February 2026

---

## License

MIT License — see [LICENSE](LICENSE) for details.
