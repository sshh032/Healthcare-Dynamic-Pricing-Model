"""
Cost Optimisation module.

Provides tools to find the optimal insurance plan configuration that minimises
the predicted premium while satisfying clinical and regulatory constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from models.pricing_model import PricingModel


# Bounds and allowed values for optimisable features
FEATURE_BOUNDS = {
    "exercise_frequency": (0, 7),
    "bmi": (18.5, 40.0),
}

PLAN_OPTIONS = ["bronze", "silver", "gold", "platinum"]
PLAN_COSTS = {"bronze": 0.70, "silver": 0.90, "gold": 1.10, "platinum": 1.35}


class CostOptimizer:
    """Optimise insurance premiums for given patient profiles.

    Uses differential evolution to minimise the predicted premium by
    searching over actionable lifestyle/plan variables while keeping
    all non-actionable features fixed.
    """

    def __init__(self, model: PricingModel, scaler, feature_names: list[str]) -> None:
        """Initialise the optimiser.

        Args:
            model: Fitted PricingModel.
            scaler: Fitted StandardScaler used during preprocessing.
            feature_names: Ordered list of feature names the model expects.
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def _build_sample(
        self, base_row: np.ndarray, exercise: float, bmi: float, plan_encoded: float
    ) -> pd.DataFrame:
        """Build a full feature vector from base + optimisable variables.

        Returns a single-row DataFrame so that the scaler can match feature names.
        """
        row = base_row.copy()
        if "exercise_frequency" in self.feature_names:
            idx = self.feature_names.index("exercise_frequency")
            row[idx] = exercise
        if "bmi" in self.feature_names:
            idx = self.feature_names.index("bmi")
            row[idx] = bmi
        if "plan_type" in self.feature_names:
            idx = self.feature_names.index("plan_type")
            row[idx] = plan_encoded
        if "age_bmi_interaction" in self.feature_names:
            age_idx = self.feature_names.index("age")
            bmi_interaction_idx = self.feature_names.index("age_bmi_interaction")
            row[bmi_interaction_idx] = row[age_idx] * bmi
        return pd.DataFrame([row], columns=self.feature_names)

    def optimise_single(
        self,
        patient_df_row: pd.Series,
        seed: int = 0,
    ) -> dict:
        """Find the premium-minimising plan configuration for one patient.

        Args:
            patient_df_row: A single row from the preprocessed (but unscaled)
                feature DataFrame.
            seed: Random seed for differential evolution.

        Returns:
            Dictionary with keys:
                original_premium, optimised_premium, savings,
                optimal_exercise, optimal_bmi, optimal_plan.
        """
        base = patient_df_row.values.astype(float)

        def objective(params: np.ndarray) -> float:
            exercise, bmi, plan_idx = params
            plan_idx = int(round(plan_idx)) % len(PLAN_OPTIONS)
            row_df = self._build_sample(base, exercise, bmi, float(plan_idx))
            scaled = self.scaler.transform(row_df)
            return float(self.model.predict(scaled)[0])

        bounds = [
            FEATURE_BOUNDS["exercise_frequency"],
            FEATURE_BOUNDS["bmi"],
            (0, len(PLAN_OPTIONS) - 1),
        ]
        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=200,
            tol=1e-4,
            polish=True,
            workers=1,
        )

        original_scaled = self.scaler.transform(
            pd.DataFrame([base], columns=self.feature_names)
        )
        original_premium = float(self.model.predict(original_scaled)[0])

        opt_exercise, opt_bmi, opt_plan_idx = result.x
        opt_plan_idx = int(round(opt_plan_idx)) % len(PLAN_OPTIONS)

        return {
            "original_premium": original_premium,
            "optimised_premium": result.fun,
            "savings": original_premium - result.fun,
            "optimal_exercise_frequency": round(opt_exercise, 1),
            "optimal_bmi": round(opt_bmi, 2),
            "optimal_plan": PLAN_OPTIONS[opt_plan_idx],
        }

    def batch_optimise(
        self,
        X_df: pd.DataFrame,
        n_samples: int = 10,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Run optimisation for a sample of patients.

        Args:
            X_df: Unscaled feature DataFrame (preprocessed but not scaled).
            n_samples: Number of patients to optimise.
            seed: Random seed.

        Returns:
            DataFrame summarising optimisation results per patient.
        """
        rows = []
        sample = X_df.head(n_samples)
        for i, (_, row) in enumerate(sample.iterrows()):
            result = self.optimise_single(row, seed=seed + i)
            rows.append(result)
        return pd.DataFrame(rows)
