<div align="center">

# 📊 Incident Count Prediction Model

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Model-Gradient_Boosting-success?style=for-the-badge" alt="Model">
  <img src="https://img.shields.io/badge/R²-0.9282-blue?style=for-the-badge" alt="R²">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p><strong>Time-series regression model predicting operational incident counts across 116 locations<br>using a heavily regularized Gradient Boosting pipeline — built on 20,000+ real-world records.</strong></p>

<p><em>Developed during a Data Science Internship at <strong>Zetta Global</strong> · February 2026</em></p>

</div>

---

## 🎯 Results at a Glance

<table>
  <tr>
    <td align="center"><h3>0.9282</h3><sub>R² Score</sub></td>
    <td align="center"><h3>20.99</h3><sub>MAE</sub></td>
    <td align="center"><h3>14.00</h3><sub>Median AE</sub></td>
    <td align="center"><h3>30.56</h3><sub>RMSE</sub></td>
    <td align="center"><h3>67%</h3><sub>vs Baseline</sub></td>
  </tr>
</table>

> **93% of variance explained** — the model reduces prediction error by **67%** compared to a naive mean predictor (MAE: 20.99 vs baseline 64.54).

---

## 🧩 Problem Statement

A transportation company needed to **forecast incident volumes per location for the upcoming year** to enable proactive resource allocation, staffing decisions, and risk mitigation — shifting from **reactive response** to **data-driven prevention**.

<table>
  <tr>
    <td>⚠️ <strong>Skewed Distribution</strong><br><sub>Top location had 10× the median count</sub></td>
    <td>📉 <strong>Limited History</strong><br><sub>Only 3 years of usable data (232 training samples)</sub></td>
    <td>🔍 <strong>Leakage Risk</strong><br><sub>Same-year aggregates had 99%+ target correlation</sub></td>
  </tr>
</table>

---

## 🔬 Approach

### Pipeline

```
Raw Data ──▶ Clean & Preprocess ──▶ Feature Engineering ──▶ Feature Selection ──▶ Model Training ──▶ Validation ──▶ Predictions
  20K+          Temporal features       56 features            Top 20 (RF+ET)       Gradient Boosting   Temporal split   8,744 forecasts
 records        Binary flags            4 categories           averaged importance   log-transformed     strict past→future
```

### 1️⃣ Data Processing

| Aspect | Detail |
|:-------|:-------|
| **Records** | 20,000+ real-world incident records |
| **Timespan** | 3 years (2023–2025) |
| **Locations** | 116 unique sites |
| **Processing** | Temporal feature extraction, binary flags, missing value handling |

### 2️⃣ Feature Engineering — 56 features → 20 selected

<table>
  <tr>
    <th width="200">Category</th>
    <th>Description</th>
    <th width="180">Impact</th>
  </tr>
  <tr>
    <td>🔗 <strong>Interaction Features</strong></td>
    <td>Trend × magnitude, weighted trends, ratio features</td>
    <td><img src="https://img.shields.io/badge/★★★-Highest_Impact-success?style=flat-square" alt="highest"></td>
  </tr>
  <tr>
    <td>⏪ <strong>Lag Features</strong></td>
    <td>Prior-year counts (raw, log, sqrt), damage/collision rates</td>
    <td><img src="https://img.shields.io/badge/★★★-Core_Predictors-blue?style=flat-square" alt="core"></td>
  </tr>
  <tr>
    <td>🌡️ <strong>Seasonal Patterns</strong></td>
    <td>Quarterly distributions, peak quarter, half-year proportions</td>
    <td><img src="https://img.shields.io/badge/★★-Temporal_Signal-orange?style=flat-square" alt="temporal"></td>
  </tr>
  <tr>
    <td>📈 <strong>Historical Stats</strong></td>
    <td>Rolling averages, min/max/mean, trend acceleration, CV</td>
    <td><img src="https://img.shields.io/badge/★★-Stability_Signal-orange?style=flat-square" alt="stability"></td>
  </tr>
</table>

> Feature selection via **averaged Random Forest + ExtraTrees importance scores** — two-model averaging reduces selection bias.

### 3️⃣ Model: Pure Gradient Boosting

After testing Random Forest, Ridge, SVR, ensembles, and stacking — a **single, heavily regularized Gradient Boosting** model outperformed all combinations:

```python
GradientBoostingRegressor(
    n_estimators=150,      # conservative count
    max_depth=3,           # shallow trees prevent overfitting
    learning_rate=0.03,    # slow learning for better generalization
    subsample=0.8,         # stochastic gradient boosting
    min_samples_leaf=4,    # regularization
    random_state=42
)
```

> **Why this works:** With only **232 training samples**, a simple well-tuned model generalizes better than complex ensembles. The key was **log-transforming the target** to handle right-skewed count distributions.

### 4️⃣ Validation Strategy

| Strategy | Purpose |
|:---------|:--------|
| 🕐 **Strict temporal split** | Model only ever sees the past — no random CV |
| 📐 **Log-transformed target** | Handles right-skewed distribution of counts |
| 🛡️ **Data leakage detection** | Caught & removed same-year features with 99%+ target correlation |

---

## 📊 Top Feature Importances

```
  Weighted Trend        ████████████████████░░░░░░░  18.4%
  Trend Magnitude       ███████████████░░░░░░░░░░░░  15.0%
  Peak Quarter Count    █████████████░░░░░░░░░░░░░░  13.4%
  Key Incident (lag)    ████████████░░░░░░░░░░░░░░░  12.8%
  Year-over-Year Trend  ████████░░░░░░░░░░░░░░░░░░░   8.2%
  Seasonal Variance     ██████░░░░░░░░░░░░░░░░░░░░░   6.3%
  High-Volume Flag      █████░░░░░░░░░░░░░░░░░░░░░░   5.1%
```

> **Interaction features** (weighted trend, trend × magnitude) dominate — this was the key breakthrough that pushed R² from ~0.80 to 0.93.

---

## 🔮 Prediction Summary (2026)

<div align="center">

| | Metric | Value |
|:---:|:-------|------:|
| 📋 | **Total Predicted Incidents** | **8,744** |
| 🏢 | **Locations Covered** | **116** |
| 📏 | **Average per Location** | **~75** |

</div>

### Distribution Across Locations

```
  200+ incidents  ██░░░░░░░░░░░░░░░░░░   6 sites   (high-risk)
  100–199         ████░░░░░░░░░░░░░░░░  17 sites
  50–99           ████████░░░░░░░░░░░░  38 sites   (largest group)
  20–49           ██████░░░░░░░░░░░░░░  29 sites
  < 20            █████░░░░░░░░░░░░░░░  26 sites   (low-volume)
```

---

## 📁 Project Structure

```
incident-prediction-model/
│
├── 📄 README.md                    # You are here
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Dependencies
├── 📄 .gitignore                   # Privacy & cleanup rules
│
├── ⚙️ config/
│   └── config.py                   # Hyperparameters, feature lists, paths
│
├── 📦 src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV loading & preprocessing
│   ├── feature_engineering.py      # 56 features across 4 categories
│   ├── model_training.py           # GB training with log-transformed target
│   ├── model_evaluation.py         # R², MAE, RMSE, sMAPE, baseline comparison
│   └── prediction.py               # Future-year prediction generation
│
├── 🚀 main.py                     # Full pipeline entry point (CLI)
│
├── 📊 data/
│   └── sample_data.csv             # Synthetic sample (real data excluded)
│
├── 📈 output/                     # Generated predictions (gitignored)
└── 🤖 models/                     # Saved models (gitignored)
```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/Nadercr7/incident-prediction-model.git
cd incident-prediction-model
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py                                  # default settings
python main.py --data-path data/my_data.csv     # custom data
python main.py --year 2027                      # different prediction year
python main.py --config                         # view configuration
python main.py --quiet                          # reduced output
```

<details>
<summary><strong>📝 Use Individual Modules</strong> (click to expand)</summary>

<br>

```python
from src.data_loader import IncidentDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.prediction import PredictionGenerator

# 1. Load & preprocess
loader = IncidentDataLoader("data/incidents.csv")
df = loader.process_all()

# 2. Engineer features (56 total)
engineer = FeatureEngineer(df)
df_features = engineer.build_all_features()

# 3. Select top 20 features
features = engineer.select_top_features(n=20)

# 4. Train model
trainer = ModelTrainer()
trainer.prepare_data(df_features, features)
model = trainer.train_gradient_boosting()

# 5. Generate predictions
generator = PredictionGenerator(df_features, features, trainer)
predictions = generator.generate_predictions(2026)
generator.save_predictions("output/predictions_2026.csv")
```

</details>

---

## 💡 Key Learnings

<table>
  <tr>
    <td width="60" align="center">🔧</td>
    <td><strong>Feature engineering > model complexity</strong><br>The jump from ~80% to 93% R² came from <em>interaction features</em> (weighted trends, trend × lag), not from more complex algorithms.</td>
  </tr>
  <tr>
    <td align="center">🕐</td>
    <td><strong>Temporal validation is non-negotiable</strong><br>Random splits in time-series data inflate metrics. Strict past→future split ensures honest, deployable results.</td>
  </tr>
  <tr>
    <td align="center">🛡️</td>
    <td><strong>Data leakage detection saved the project</strong><br>Same-year aggregate features had 99%+ correlation with the target. Without catching this, the model would appear R²=0.99 but fail completely in production.</td>
  </tr>
  <tr>
    <td align="center">📉</td>
    <td><strong>Small data ≠ bad model</strong><br>With only 232 training samples, careful regularization (shallow trees, low learning rate, subsampling) achieved strong generalization.</td>
  </tr>
</table>

---

## ⚠️ Limitations

| Limitation | Detail |
|:-----------|:-------|
| **Limited history** | 3 years of data constrains lag feature depth |
| **No external factors** | Weather, policy changes, economic conditions not included |
| **Annual granularity** | Monthly/weekly predictions would require more data |
| **Point predictions** | No confidence intervals (could add with quantile regression) |
| **sMAPE = 56.4%** | Inflated by low-count locations where ±5 incidents = large % error |

---

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-444876?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn">
</p>

---

## 🔒 Privacy Notice

> This project was completed during a Data Science internship at **Zetta Global**. All location names, employee data, and sensitive operational details have been **removed**. Only methodology, aggregate statistics, and anonymized results are shared. The raw dataset is **not included** in this repository.

---

## 👤 Author

<table>
  <tr>
    <td>
      <strong>Nader Mohamed</strong><br>
      Data Science Intern · <strong>Zetta Global</strong> · February 2026<br><br>
      <a href="https://github.com/Nadercr7"><img src="https://img.shields.io/badge/GitHub-Nadercr7-181717?style=flat-square&logo=github" alt="GitHub"></a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>⭐ If you found this project useful, consider giving it a star!</sub>
</div>
