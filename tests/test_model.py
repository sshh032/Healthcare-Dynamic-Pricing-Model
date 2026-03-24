"""
Unit tests for the Healthcare Dynamic Pricing Model.

Tests cover:
  - Data generation
  - Preprocessing utilities
  - PricingModel training, prediction, and evaluation
  - PricingExplainer SHAP values
  - CostOptimizer optimisation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import generate_healthcare_data, split_features_target
from models.optimization_model import CostOptimizer
from models.pricing_model import PricingModel
from utils.preprocessing import (
    apply_encoders,
    encode_categoricals,
    engineer_features,
    prepare_dataset,
    scale_features,
)
from xai.explainer import PricingExplainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    return generate_healthcare_data(n_samples=300, random_state=0)


@pytest.fixture(scope="module")
def prepared_data(small_df):
    return prepare_dataset(small_df, random_state=0)


@pytest.fixture(scope="module")
def fitted_model(prepared_data):
    model = PricingModel(n_estimators=50, random_state=0)
    model.fit(
        prepared_data["X_train"],
        prepared_data["y_train"],
        feature_names=prepared_data["feature_names"],
    )
    return model


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_shape(self, small_df):
        assert small_df.shape == (300, 14)

    def test_no_nulls(self, small_df):
        assert small_df.isnull().sum().sum() == 0

    def test_premium_positive(self, small_df):
        assert (small_df["annual_premium"] > 0).all()

    def test_columns_present(self, small_df):
        expected = {
            "age", "bmi", "num_dependents", "smoker", "region",
            "pre_existing_conditions", "previous_claims", "annual_income",
            "employment_type", "exercise_frequency", "chronic_conditions",
            "mental_health_history", "plan_type", "annual_premium",
        }
        assert expected == set(small_df.columns)

    def test_split_features_target(self, small_df):
        X, y = split_features_target(small_df)
        assert "annual_premium" not in X.columns
        assert len(y) == len(small_df)

    def test_reproducibility(self):
        df1 = generate_healthcare_data(n_samples=50, random_state=42)
        df2 = generate_healthcare_data(n_samples=50, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_healthcare_data(n_samples=50, random_state=1)
        df2 = generate_healthcare_data(n_samples=50, random_state=2)
        assert not df1.equals(df2)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_encode_categoricals(self, small_df):
        X, _ = split_features_target(small_df)
        enc, encoders = encode_categoricals(X)
        for col in ["region", "employment_type", "plan_type"]:
            assert enc[col].dtype in [np.int32, np.int64, int, np.intp]
        assert set(encoders.keys()) == {"region", "employment_type", "plan_type"}

    def test_apply_encoders(self, small_df):
        X, _ = split_features_target(small_df)
        enc, encoders = encode_categoricals(X)
        reapplied = apply_encoders(X, encoders)
        pd.testing.assert_frame_equal(enc, reapplied)

    def test_engineer_features_adds_columns(self, small_df):
        X, _ = split_features_target(small_df)
        enc, _ = encode_categoricals(X)
        feat = engineer_features(enc)
        assert "age_bmi_interaction" in feat.columns
        assert "risk_score" in feat.columns
        assert "income_to_age_ratio" in feat.columns

    def test_scale_features_shape(self, prepared_data):
        assert prepared_data["X_train"].shape[1] == len(prepared_data["feature_names"])
        assert prepared_data["X_test"].shape[1] == len(prepared_data["feature_names"])

    def test_prepare_dataset_keys(self, prepared_data):
        expected_keys = {
            "X_train", "X_test", "y_train", "y_test",
            "X_train_df", "X_test_df", "encoders", "scaler", "feature_names",
        }
        assert expected_keys == set(prepared_data.keys())


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPricingModel:
    def test_predict_shape(self, fitted_model, prepared_data):
        preds = fitted_model.predict(prepared_data["X_test"])
        assert preds.shape == (len(prepared_data["y_test"]),)

    def test_predict_positive(self, fitted_model, prepared_data):
        preds = fitted_model.predict(prepared_data["X_test"])
        assert (preds > 0).all()

    def test_evaluate_keys(self, fitted_model, prepared_data):
        metrics = fitted_model.evaluate(prepared_data["X_test"], prepared_data["y_test"])
        assert set(metrics.keys()) == {"MAE", "RMSE", "R2", "MAPE"}

    def test_r2_reasonable(self, fitted_model, prepared_data):
        metrics = fitted_model.evaluate(prepared_data["X_test"], prepared_data["y_test"])
        assert metrics["R2"] > 0.7, f"R² too low: {metrics['R2']}"

    def test_feature_importance_length(self, fitted_model, prepared_data):
        imp = fitted_model.get_feature_importance()
        assert len(imp) == len(prepared_data["feature_names"])

    def test_unfitted_predict_raises(self):
        m = PricingModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((1, 5)))

    def test_unfitted_importance_raises(self):
        m = PricingModel()
        with pytest.raises(RuntimeError):
            m.get_feature_importance()


# ---------------------------------------------------------------------------
# XAI tests
# ---------------------------------------------------------------------------

class TestPricingExplainer:
    def test_shap_values_shape(self, fitted_model, prepared_data):
        explainer = PricingExplainer(fitted_model, prepared_data["feature_names"])
        sv = explainer.compute_shap_values(prepared_data["X_test"][:50])
        assert sv.shape == (50, len(prepared_data["feature_names"]))

    def test_global_importance_columns(self, fitted_model, prepared_data):
        explainer = PricingExplainer(fitted_model, prepared_data["feature_names"])
        df = explainer.global_feature_importance(prepared_data["X_test"][:50])
        assert list(df.columns) == ["feature", "mean_abs_shap"]
        assert len(df) == len(prepared_data["feature_names"])

    def test_global_importance_sorted(self, fitted_model, prepared_data):
        explainer = PricingExplainer(fitted_model, prepared_data["feature_names"])
        df = explainer.global_feature_importance(prepared_data["X_test"][:50])
        assert df["mean_abs_shap"].is_monotonic_decreasing

    def test_local_explanation(self, fitted_model, prepared_data):
        explainer = PricingExplainer(fitted_model, prepared_data["feature_names"])
        row = prepared_data["X_test"][0]
        df = explainer.local_explanation(row)
        assert list(df.columns) == ["feature", "feature_value", "shap_value"]
        assert len(df) == len(prepared_data["feature_names"])


# ---------------------------------------------------------------------------
# Optimisation tests
# ---------------------------------------------------------------------------

class TestCostOptimizer:
    def test_optimise_single_keys(self, fitted_model, prepared_data):
        optimizer = CostOptimizer(
            fitted_model, prepared_data["scaler"], prepared_data["feature_names"]
        )
        row = prepared_data["X_test_df"].iloc[0]
        result = optimizer.optimise_single(row, seed=0)
        expected_keys = {
            "original_premium", "optimised_premium", "savings",
            "optimal_exercise_frequency", "optimal_bmi", "optimal_plan",
        }
        assert expected_keys == set(result.keys())

    def test_batch_optimise_shape(self, fitted_model, prepared_data):
        optimizer = CostOptimizer(
            fitted_model, prepared_data["scaler"], prepared_data["feature_names"]
        )
        df = optimizer.batch_optimise(prepared_data["X_test_df"], n_samples=3, seed=0)
        assert df.shape[0] == 3

    def test_optimised_premium_non_negative(self, fitted_model, prepared_data):
        optimizer = CostOptimizer(
            fitted_model, prepared_data["scaler"], prepared_data["feature_names"]
        )
        row = prepared_data["X_test_df"].iloc[0]
        result = optimizer.optimise_single(row, seed=0)
        assert result["optimised_premium"] >= 0
