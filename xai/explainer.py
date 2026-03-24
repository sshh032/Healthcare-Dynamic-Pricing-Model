"""
Explainable AI (XAI) module using SHAP values.

Provides SHAP-based explanations for the pricing model predictions,
including global feature importance and per-patient local explanations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class PricingExplainer:
    """SHAP-based explainer for the healthcare pricing model.

    Wraps a shap.TreeExplainer to compute global and local explanations
    for an XGBoost-backed PricingModel.
    """

    def __init__(self, model, feature_names: list[str]) -> None:
        """Initialise the explainer.

        Args:
            model: Fitted PricingModel instance (must have a `.model` XGBRegressor).
            feature_names: Ordered list of feature names.
        """
        self.explainer = shap.TreeExplainer(model.model)
        self.feature_names = feature_names
        self._shap_values: np.ndarray | None = None

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a feature matrix.

        Args:
            X: Feature matrix (scaled or unscaled – must match what the model expects).

        Returns:
            SHAP values array with shape (n_samples, n_features).
        """
        shap_values = self.explainer.shap_values(X)
        self._shap_values = shap_values
        return shap_values

    def global_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Compute mean absolute SHAP values as a global feature importance table.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    def local_explanation(self, X_row: np.ndarray) -> pd.DataFrame:
        """Compute a SHAP explanation for a single prediction.

        Args:
            X_row: A single sample (1-D array or 2-D array with 1 row).

        Returns:
            DataFrame with columns ['feature', 'feature_value', 'shap_value']
            sorted by absolute SHAP value descending.
        """
        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)
        sv = self.explainer.shap_values(X_row)[0]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "feature_value": X_row[0],
            "shap_value": sv,
        }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)
        return df

    def get_shap_values(self) -> np.ndarray | None:
        """Return the last computed SHAP values array.

        Returns:
            SHAP values if compute_shap_values has been called, else None.
        """
        return self._shap_values
