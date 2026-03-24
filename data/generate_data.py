"""
Synthetic Healthcare Insurance Data Generator.

Generates realistic synthetic data for training and evaluating the
Healthcare Dynamic Pricing Model.
"""

import numpy as np
import pandas as pd


def generate_healthcare_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic healthcare insurance dataset.

    Features:
        age: Patient age (18-75)
        bmi: Body Mass Index (15-50)
        num_dependents: Number of dependents (0-5)
        smoker: Whether the patient smokes (0/1)
        region: Geographic region (northeast/northwest/southeast/southwest)
        pre_existing_conditions: Number of pre-existing conditions (0-4)
        previous_claims: Number of claims in the past 3 years (0-6)
        annual_income: Annual income in USD (20k-200k)
        employment_type: Type of employment (employed/self_employed/unemployed)
        exercise_frequency: Weekly exercise frequency (0-7 days)
        chronic_conditions: Binary flag for chronic conditions (0/1)
        mental_health_history: Binary flag for mental health history (0/1)
        plan_type: Insurance plan type (bronze/silver/gold/platinum)

    Target:
        annual_premium: Annual insurance premium in USD

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic healthcare insurance data.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 76, size=n_samples).astype(float)
    bmi = rng.uniform(15, 50, size=n_samples)
    num_dependents = rng.integers(0, 6, size=n_samples).astype(float)
    smoker = rng.integers(0, 2, size=n_samples).astype(float)
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], size=n_samples)
    pre_existing = rng.integers(0, 5, size=n_samples).astype(float)
    prev_claims = rng.integers(0, 7, size=n_samples).astype(float)
    annual_income = rng.uniform(20_000, 200_000, size=n_samples)
    employment_type = rng.choice(["employed", "self_employed", "unemployed"], size=n_samples)
    exercise_freq = rng.integers(0, 8, size=n_samples).astype(float)
    chronic = rng.integers(0, 2, size=n_samples).astype(float)
    mental_health = rng.integers(0, 2, size=n_samples).astype(float)
    plan_type = rng.choice(["bronze", "silver", "gold", "platinum"], size=n_samples)

    region_factor = np.where(
        region == "northeast", 1.15,
        np.where(region == "northwest", 1.05,
                 np.where(region == "southeast", 1.0, 0.95))
    )
    employment_factor = np.where(
        employment_type == "employed", 1.0,
        np.where(employment_type == "self_employed", 1.1, 1.25)
    )
    plan_factor = np.where(
        plan_type == "bronze", 0.7,
        np.where(plan_type == "silver", 0.9,
                 np.where(plan_type == "gold", 1.1, 1.35))
    )
    bmi_risk = np.where(bmi < 18.5, 1.1,
                np.where(bmi < 25.0, 1.0,
                         np.where(bmi < 30.0, 1.15, 1.35)))

    base_premium = (
        2500
        + 180 * age
        + 60 * bmi
        + 400 * num_dependents
        + 8500 * smoker
        + 900 * pre_existing
        + 600 * prev_claims
        - 0.008 * annual_income
        + 1500 * chronic
        + 700 * mental_health
        - 200 * exercise_freq
    )
    base_premium = (
        base_premium
        * region_factor
        * employment_factor
        * plan_factor
        * bmi_risk
    )
    noise = rng.normal(0, 800, size=n_samples)
    annual_premium = np.maximum(base_premium + noise, 1000.0)

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "num_dependents": num_dependents,
        "smoker": smoker,
        "region": region,
        "pre_existing_conditions": pre_existing,
        "previous_claims": prev_claims,
        "annual_income": annual_income,
        "employment_type": employment_type,
        "exercise_frequency": exercise_freq,
        "chronic_conditions": chronic,
        "mental_health_history": mental_health,
        "plan_type": plan_type,
        "annual_premium": annual_premium,
    })

    return df


def split_features_target(df: pd.DataFrame, target_col: str = "annual_premium"):
    """Split DataFrame into features and target.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
