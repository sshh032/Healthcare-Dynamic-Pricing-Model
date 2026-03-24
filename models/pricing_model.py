"""
Pricing prediction model using XGBoost with cross-validation and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


class PricingModel:
    """XGBoost-based healthcare insurance premium prediction model.

    Attributes:
        model: The underlying XGBRegressor.
        feature_names: Names of input features.
        is_fitted: Whether the model has been trained.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """Initialise the pricing model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: XGBoost learning rate (eta).
            subsample: Row sub-sampling ratio per tree.
            colsample_bytree: Column sub-sampling ratio per tree.
            reg_alpha: L1 regularisation term.
            reg_lambda: L2 regularisation term.
            random_state: Random seed for reproducibility.
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "PricingModel":
        """Train the model.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            feature_names: Optional list of feature names for interpretability.

        Returns:
            self (for method chaining).
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        if feature_names is not None:
            self.feature_names = list(feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate premium predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted premiums.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Compute evaluation metrics on a held-out test set.

        Args:
            X_test: Test feature matrix.
            y_test: True target values.

        Returns:
            Dictionary with MAE, RMSE, R², and MAPE metrics.
        """
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100)
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Run k-fold cross-validation and report mean/std of R².

        Args:
            X: Full feature matrix.
            y: Full target vector.
            n_splits: Number of CV folds.
            random_state: Random seed for fold splitting.

        Returns:
            Dictionary with mean and std of R² across folds.
        """
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        return {"cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted in descending order.

        Returns:
            Pandas Series of feature importances indexed by feature name.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling get_feature_importance().")
        importances = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [f"f{i}" for i in range(len(importances))]
        return pd.Series(importances, index=names).sort_values(ascending=False)
