# Healthcare Dynamic Pricing Model

A machine-learning system for **healthcare insurance premium prediction** and **cost optimisation**, featuring **Explainable AI (XAI)** via SHAP values.

---

## Overview

This project builds an end-to-end pipeline that:

1. **Generates** synthetic, realistic healthcare insurance data (13 patient features, 1 target premium).
2. **Pre-processes** the data – label encoding, feature engineering (risk score, interaction terms), and standard scaling.
3. **Trains** an XGBoost regression model to predict annual insurance premiums.
4. **Evaluates** the model with MAE, RMSE, R², MAPE, and 5-fold cross-validation.
5. **Explains** predictions using SHAP – global feature importance and per-patient local explanations.
6. **Optimises** each patient's insurance cost by searching over actionable variables (exercise frequency, BMI, plan type) using differential evolution.
7. **Visualises** results with six publication-quality plots saved to `outputs/`.

---

## Project Structure

```
Healthcare-Dynamic-Pricing-Model/
├── data/
│   └── generate_data.py        # Synthetic data generation
├── models/
│   ├── pricing_model.py        # XGBoost pricing model
│   └── optimization_model.py   # Cost optimisation via differential evolution
├── xai/
│   └── explainer.py            # SHAP-based XAI explanations
├── visualization/
│   └── plots.py                # All visualisation functions
├── utils/
│   └── preprocessing.py        # Encoding, feature engineering, scaling
├── tests/
│   └── test_model.py           # 26 unit tests (pytest)
├── outputs/                    # Generated plots (created at runtime)
├── main.py                     # Full pipeline entry point
└── requirements.txt
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python main.py
```

Sample output:

```
============================================================
  Healthcare Dynamic Pricing Model
  Machine Learning + XAI Pipeline
============================================================

[1/6] Generating synthetic healthcare data …
      Dataset shape: (5000, 14)
      Premium range: $1,000 – $92,445
      Mean premium:  $23,814

[2/6] Preprocessing & feature engineering …
      Features: 16  |  Train: 4000  |  Test: 1000

[3/6] Training XGBoost pricing model …

      Test-set metrics:
        MAE  : $1,138.01
        RMSE : $1,491.94
        R²   : 0.9857
        MAPE : 4.15%

[4/6] Running 5-fold cross-validation …
      CV R² = 0.9845 ± 0.0018

[5/6] Computing SHAP explanations …

      Top-5 features by mean |SHAP|:
        smoker                          4,231.18
        age_bmi_interaction             2,187.43
        plan_type                       1,954.06
        age                             1,830.22
        pre_existing_conditions         1,204.55

[6/6] Running cost optimisation for 10 patients …
      Mean potential annual saving: $15,272.43
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Features

| Feature | Description |
|---|---|
| `age` | Patient age (18–75) |
| `bmi` | Body Mass Index (15–50) |
| `num_dependents` | Number of dependents (0–5) |
| `smoker` | Smoking status (0/1) |
| `region` | Geographic region |
| `pre_existing_conditions` | Number of pre-existing conditions (0–4) |
| `previous_claims` | Claims in the past 3 years (0–6) |
| `annual_income` | Annual income (USD) |
| `employment_type` | employed / self_employed / unemployed |
| `exercise_frequency` | Weekly exercise days (0–7) |
| `chronic_conditions` | Chronic condition flag (0/1) |
| `mental_health_history` | Mental health history flag (0/1) |
| `plan_type` | bronze / silver / gold / platinum |

Engineered features added automatically: `age_bmi_interaction`, `risk_score`, `income_to_age_ratio`.

---

## Output Plots

All plots are saved to `outputs/`:

| File | Description |
|---|---|
| `feature_importance.png` | XGBoost built-in feature importances |
| `shap_importance.png` | Mean absolute SHAP values |
| `shap_summary.png` | SHAP beeswarm summary |
| `predictions_vs_actual.png` | Predicted vs. actual premiums scatter |
| `residuals.png` | Residual distribution and residuals vs. predicted |
| `optimisation_savings.png` | Distribution of premium savings per patient |

---

## Model Performance

| Metric | Value |
|---|---|
| R² | 0.9857 |
| MAE | $1,138 |
| RMSE | $1,492 |
| MAPE | 4.15% |
| CV R² (5-fold) | 0.9845 ± 0.0018 |

---

## Technologies

- **XGBoost** – gradient boosted trees for tabular regression
- **SHAP** – SHapley Additive exPlanations for model interpretability
- **scikit-learn** – preprocessing and evaluation utilities
- **SciPy** – differential evolution for cost optimisation
- **Matplotlib / Seaborn** – visualisation
