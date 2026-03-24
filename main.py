"""
Main pipeline script for the Healthcare Dynamic Pricing Model.

Orchestrates:
  1. Synthetic data generation
  2. Preprocessing & feature engineering
  3. XGBoost model training and evaluation
  4. Cross-validation
  5. SHAP-based XAI explanations
  6. Cost optimisation
  7. Visualisation output
"""

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.generate_data import generate_healthcare_data
from models.optimization_model import CostOptimizer
from models.pricing_model import PricingModel
from utils.preprocessing import prepare_dataset
from visualization.plots import (
    plot_feature_importance,
    plot_optimisation_savings,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_shap_bar,
    plot_shap_summary,
)
from xai.explainer import PricingExplainer


def run_pipeline(n_samples: int = 5000, random_state: int = 42) -> dict:
    """Execute the full Healthcare Dynamic Pricing Model pipeline.

    Args:
        n_samples: Number of synthetic samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with evaluation metrics, SHAP importance, and
        optimisation summary.
    """
    print("=" * 60)
    print("  Healthcare Dynamic Pricing Model")
    print("  Machine Learning + XAI Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data generation
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic healthcare data …")
    df = generate_healthcare_data(n_samples=n_samples, random_state=random_state)
    print(f"      Dataset shape: {df.shape}")
    print(f"      Premium range: ${df['annual_premium'].min():,.0f} – "
          f"${df['annual_premium'].max():,.0f}")
    print(f"      Mean premium:  ${df['annual_premium'].mean():,.0f}")

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    print("\n[2/6] Preprocessing & feature engineering …")
    data = prepare_dataset(df, random_state=random_state)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    scaler = data["scaler"]
    X_test_df = data["X_test_df"]
    print(f"      Features: {len(feature_names)}  |  "
          f"Train: {len(y_train)}  |  Test: {len(y_test)}")

    # ------------------------------------------------------------------
    # 3. Model training & evaluation
    # ------------------------------------------------------------------
    print("\n[3/6] Training XGBoost pricing model …")
    model = PricingModel(random_state=random_state)
    model.fit(X_train, y_train, feature_names=feature_names)

    metrics = model.evaluate(X_test, y_test)
    print(f"\n      Test-set metrics:")
    print(f"        MAE  : ${metrics['MAE']:,.2f}")
    print(f"        RMSE : ${metrics['RMSE']:,.2f}")
    print(f"        R²   : {metrics['R2']:.4f}")
    print(f"        MAPE : {metrics['MAPE']:.2f}%")

    # ------------------------------------------------------------------
    # 4. Cross-validation
    # ------------------------------------------------------------------
    print("\n[4/6] Running 5-fold cross-validation …")
    cv_results = model.cross_validate(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
    )
    print(f"      CV R² = {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")

    # ------------------------------------------------------------------
    # 5. XAI – SHAP explanations
    # ------------------------------------------------------------------
    print("\n[5/6] Computing SHAP explanations …")
    explainer = PricingExplainer(model, feature_names)
    shap_sample = X_test[:200]
    shap_values = explainer.compute_shap_values(shap_sample)
    shap_importance = explainer.global_feature_importance(shap_sample)

    print("\n      Top-5 features by mean |SHAP|:")
    for _, row in shap_importance.head(5).iterrows():
        print(f"        {row['feature']:30s}  {row['mean_abs_shap']:,.2f}")

    local_exp = explainer.local_explanation(X_test[0])
    print(f"\n      Local explanation (patient #1):")
    for _, row in local_exp.head(5).iterrows():
        print(f"        {row['feature']:30s}  SHAP={row['shap_value']:+,.2f}")

    # ------------------------------------------------------------------
    # 6. Cost optimisation
    # ------------------------------------------------------------------
    print("\n[6/6] Running cost optimisation for 10 patients …")
    optimizer = CostOptimizer(model, scaler, feature_names)
    opt_df = optimizer.batch_optimise(X_test_df, n_samples=10, seed=random_state)
    mean_saving = opt_df["savings"].mean()
    print(f"      Mean potential annual saving: ${mean_saving:,.2f}")
    print(f"\n      Optimisation summary (first 5 patients):")
    print(opt_df[["original_premium", "optimised_premium", "savings",
                   "optimal_plan"]].head().to_string(index=False))

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    print("\n[+] Generating visualisation plots …")
    y_pred = model.predict(X_test)

    paths = []
    paths.append(plot_feature_importance(model.get_feature_importance()))
    paths.append(plot_shap_bar(shap_importance))
    paths.append(plot_shap_summary(shap_values, shap_sample, feature_names))
    paths.append(plot_predictions_vs_actual(y_test, y_pred))
    paths.append(plot_residuals(y_test, y_pred))
    paths.append(plot_optimisation_savings(opt_df))

    for p in paths:
        print(f"      Saved: {p}")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)

    return {
        "metrics": metrics,
        "cv_results": cv_results,
        "shap_importance": shap_importance,
        "optimisation_summary": opt_df,
    }


if __name__ == "__main__":
    run_pipeline()
