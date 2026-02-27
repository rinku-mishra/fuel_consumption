"""
Plotting utilities for Simple vs Multiple Linear Regression.

All functions return a ``matplotlib.figure.Figure`` so callers can display
them (``plt.show()``, ``st.pyplot()``) or save them (``fig.savefig(...)``).

Each function accepts an optional ``ax`` parameter; when *None*, a new
figure + axes pair is created automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fuel_model import RegressionResult


# ──────────────────────────────────────────────────────────────────────────────
# 2-D: single feature vs target
# ──────────────────────────────────────────────────────────────────────────────

def plot_simple_vs_feature(
    X_train: np.ndarray,
    y_train: np.ndarray,
    result: "RegressionResult",
    feature_name: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter of one (scaled) feature vs CO2 with the fitted regression line.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features (uses column 0 only).
    y_train : np.ndarray
        Training target values.
    result : RegressionResult
        Simple regression result (must have been trained on column 0).
    feature_name : str
        Label for the x-axis.
    ax : plt.Axes, optional
        Axes to draw on; a new figure is created if *None*.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    x = X_train[:, 0]

    # Regression line in scaled feature space
    coef_s = result.model.coef_.flatten()[0]
    intercept_s = result.model.intercept_[0]
    y_line = coef_s * x + intercept_s

    ax.scatter(x, y_train, color="#4361ee", alpha=0.55, s=30, edgecolors="white",
               linewidth=0.3, label="Training data")
    ax.plot(np.sort(x), y_line[np.argsort(x)], color="#e63946", lw=2, label="Fit")
    ax.set_xlabel(f"{feature_name} (standardised)")
    ax.set_ylabel("CO₂ Emissions (g/km)")
    ax.set_title(f"Simple Regression: {feature_name} vs CO₂")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25)

    if fig is not None:
        fig.tight_layout()
    return fig or plt.gcf()


# ──────────────────────────────────────────────────────────────────────────────
# 3-D: two features + regression plane
# ──────────────────────────────────────────────────────────────────────────────

def plot_multiple_3d(
    X_test: np.ndarray,
    y_test: np.ndarray,
    result: "RegressionResult",
    feature_names: list[str],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """3-D scatter of two features and CO2 with the fitted regression plane.

    Parameters
    ----------
    X_test : np.ndarray
        Scaled test features (columns 0 and 1).
    y_test : np.ndarray
        Test target values.
    result : RegressionResult
        Multiple regression result.
    feature_names : list[str]
        Labels for the two feature axes.
    ax : plt.Axes, optional
        A 3-D axes object; a new figure is created if *None*.

    Returns
    -------
    plt.Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    X1, X2 = X_test[:, 0], X_test[:, 1]
    coef = result.model.coef_.flatten()
    intercept = result.model.intercept_[0]

    # Build a mesh grid for the regression surface
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(X1.min(), X1.max(), 50),
        np.linspace(X2.min(), X2.max(), 50),
    )
    y_surface = intercept + coef[0] * x1_grid + coef[1] * x2_grid

    # Colour-code points above / below the plane
    y_pred = result.model.predict(X_test)
    above = y_test.flatten() >= y_pred.flatten()
    below = ~above

    ax.scatter(X1[above], X2[above], y_test[above],
               label="Above plane", s=45, alpha=0.7, edgecolors="k", linewidth=0.3)
    ax.scatter(X1[below], X2[below], y_test[below],
               label="Below plane", s=45, alpha=0.3, edgecolors="k", linewidth=0.3)
    ax.plot_surface(x1_grid, x2_grid, y_surface, color="gray", alpha=0.2)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel("CO₂ Emissions")
    ax.set_title("Multiple Linear Regression — Regression Plane")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Residuals
# ──────────────────────────────────────────────────────────────────────────────

def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Residuals (actual − predicted) vs predicted values.

    A well-fitted model should show residuals scattered randomly around 0.

    Parameters
    ----------
    y_test : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Model predictions.
    model_name : str
        Label used in the plot title.
    ax : plt.Axes, optional
        Axes to draw on; a new figure is created if *None*.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    residuals = y_test.flatten() - y_pred.flatten()
    ax.scatter(y_pred, residuals, alpha=0.55, s=30, color="#4361ee",
               edgecolors="white", linewidth=0.3)
    ax.axhline(0, color="#e63946", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted CO₂")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals — {model_name}")
    ax.grid(True, alpha=0.25)

    if fig is not None:
        fig.tight_layout()
    return fig or plt.gcf()


# ──────────────────────────────────────────────────────────────────────────────
# Scatter matrix (optional exploratory view)
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter_matrix(
    df: pd.DataFrame,
    target_col: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Pairwise scatter matrix of all columns in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (numeric columns only).
    target_col : str
        Name of the target column (used only for labelling).
    output_path : Path, optional
        If given, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    axes = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10))
    for row in axes:
        for ax in row:
            ax.xaxis.label.set_rotation(90)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha("right")
    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return plt.gcf()
