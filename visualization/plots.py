"""
Visualisation utilities for the Healthcare Dynamic Pricing Model.

Generates and saves publication-quality plots for:
  - Feature importance (XGBoost built-in and SHAP)
  - SHAP summary plot
  - Prediction vs. actual scatter
  - Residual distribution
  - Optimisation savings histogram
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for scripts/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def plot_feature_importance(
    importance_series: pd.Series,
    top_n: int = 15,
    title: str = "XGBoost Feature Importance",
    filename: str = "feature_importance.png",
) -> Path:
    """Bar chart of model feature importances.

    Args:
        importance_series: Pandas Series indexed by feature name.
        top_n: Number of top features to display.
        title: Plot title.
        filename: Output filename (saved to outputs/).

    Returns:
        Path to the saved figure.
    """
    top = importance_series.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top[::-1].plot(kind="barh", ax=ax, color=sns.color_palette("muted")[0])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    filename: str = "shap_summary.png",
) -> Path:
    """SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        X: Feature matrix corresponding to shap_values.
        feature_names: Ordered list of feature names.
        filename: Output filename.

    Returns:
        Path to the saved figure.
    """
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_shap_bar(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    filename: str = "shap_importance.png",
) -> Path:
    """Horizontal bar chart of mean absolute SHAP values.

    Args:
        importance_df: DataFrame with columns ['feature', 'mean_abs_shap'].
        top_n: Number of top features to display.
        filename: Output filename.

    Returns:
        Path to the saved figure.
    """
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
            color=sns.color_palette("muted")[2])
    ax.set_title("Mean |SHAP| Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "predictions_vs_actual.png",
) -> Path:
    """Scatter plot of predicted vs. actual premiums.

    Args:
        y_true: True premium values.
        y_pred: Predicted premium values.
        filename: Output filename.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=sns.color_palette("muted")[0])
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title("Predicted vs Actual Premiums", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Premium (USD)")
    ax.set_ylabel("Predicted Premium (USD)")
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "residuals.png",
) -> Path:
    """Residual distribution histogram.

    Args:
        y_true: True premium values.
        y_pred: Predicted premium values.
        filename: Output filename.

    Returns:
        Path to the saved figure.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=50, color=sns.color_palette("muted")[1], edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residual Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Residual (USD)")
    axes[0].set_ylabel("Count")

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10,
                    color=sns.color_palette("muted")[3])
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Premium (USD)")
    axes[1].set_ylabel("Residual (USD)")

    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_optimisation_savings(
    opt_df: pd.DataFrame,
    filename: str = "optimisation_savings.png",
) -> Path:
    """Histogram of per-patient premium savings after optimisation.

    Args:
        opt_df: DataFrame returned by CostOptimizer.batch_optimise().
        filename: Output filename.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(opt_df["savings"], bins=20, color=sns.color_palette("muted")[4],
            edgecolor="white")
    ax.axvline(opt_df["savings"].mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean saving: ${opt_df['savings'].mean():,.0f}")
    ax.set_title("Premium Savings via Optimisation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Savings (USD)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
