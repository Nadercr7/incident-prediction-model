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

## 🎯 Results

<table>
  <tr>
    <td align="center"><h3>0.9282</h3><sub>R² Score</sub></td>
    <td align="center"><h3>20.99</h3><sub>MAE</sub></td>
    <td align="center"><h3>14.00</h3><sub>Median AE</sub></td>
    <td align="center"><h3>30.56</h3><sub>RMSE</sub></td>
    <td align="center"><h3>67%</h3><sub>vs Baseline</sub></td>
  </tr>
</table>

> The model explains **93% of variance** and reduces error by **67%** compared to a naive mean predictor.

---

## 🧩 Overview

A transportation company needed to **forecast incident volumes per location** for the upcoming year — shifting from reactive response to data-driven prevention.

**Pipeline:**

```
20K+ Records ──▶ Feature Engineering (56 → 20) ──▶ Gradient Boosting ──▶ 8,744 Predictions
```

**Key technical decisions:**
- **Log-transformed target** to handle right-skewed count distributions
- **Strict temporal split** — model only ever sees past data (no random CV)
- **Feature selection** via averaged RF + ExtraTrees importance (reduces selection bias)
- **Data leakage detection** — caught same-year features with 99%+ target correlation

---

## 📊 What Made It Work

```
  Weighted Trend        ████████████████████░░░░░░░  18.4%
  Trend Magnitude       ███████████████░░░░░░░░░░░░  15.0%
  Peak Quarter Count    █████████████░░░░░░░░░░░░░░  13.4%
  Key Incident (lag)    ████████████░░░░░░░░░░░░░░░  12.8%
  Year-over-Year Trend  ████████░░░░░░░░░░░░░░░░░░░   8.2%
```

> **Interaction features** (weighted trend, trend × magnitude) drove R² from ~0.80 to 0.93 — feature engineering mattered more than model complexity.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Nadercr7/incident-prediction-model.git
cd incident-prediction-model
pip install -r requirements.txt
python main.py                              # run full pipeline
python main.py --data-path your_data.csv    # custom data
python main.py --year 2027                  # different year
```

---

## 📁 Structure

```
├── main.py                      # Pipeline entry point (CLI)
├── config/config.py             # Hyperparameters & feature lists
├── src/
│   ├── data_loader.py           # Loading & preprocessing
│   ├── feature_engineering.py   # 56 features, top-20 selection
│   ├── model_training.py        # GB with log-transformed target
│   ├── model_evaluation.py      # Metrics & visualizations
│   └── prediction.py            # Future-year forecasting
└── data/sample_data.csv         # Synthetic sample (real data excluded)
```

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

## 🔒 Privacy

> All location names, employee data, and sensitive details have been **removed**. Only methodology and aggregate statistics are shared. The raw dataset is **not included**.

---

## 👤 Author

**Nader Mohamed** · Data Science Intern · [Zetta Global](https://www.zettaglobal.com/) · February 2026

<a href="https://github.com/Nadercr7"><img src="https://img.shields.io/badge/GitHub-Nadercr7-181717?style=flat-square&logo=github" alt="GitHub"></a>

---

<div align="center">
  <sub>⭐ If you found this useful, consider giving it a star!</sub>
</div>
