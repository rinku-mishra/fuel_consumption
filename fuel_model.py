"""
Regression modelling module for Fuel Consumption → CO2 Emissions.

Provides helpers to:

1. **Load** the raw dataset from a local CSV file.
2. **Prepare** the data — drop irrelevant columns, resolve features & target.
3. **Train** simple (1-feature) and multiple (2-feature) linear regression.
4. **Compare** models via R², MAE, and RMSE on a held-out test set.

All public functions are stateless; pipeline orchestration is in
:func:`run_pipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    DROP_CATEGORICAL,
    DROP_NUMERIC_REDUNDANT,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract per-feature std from a fitted StandardScaler
# ──────────────────────────────────────────────────────────────────────────────

def _scaler_std(scaler: StandardScaler) -> np.ndarray:
    """Return per-feature standard deviation from a fitted ``StandardScaler``.

    Handles both modern (``scale_``) and older sklearn versions (``var_``).
    """
    if hasattr(scaler, "scale_") and scaler.scale_ is not None:
        return np.asarray(scaler.scale_)
    return np.sqrt(np.asarray(scaler.var_))


# ──────────────────────────────────────────────────────────────────────────────
# Data class: regression result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """Immutable container holding a trained model and its evaluation metrics.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"Simple (1 feature)"``).
    model : LinearRegression
        The fitted scikit-learn model.
    scaler : StandardScaler | None
        The scaler used on input features (needed to convert coefficients
        back to the original scale).
    feature_names : list[str]
        Column names used as features for this model.
    r2_train, r2_test : float
        Coefficient of determination on train / test sets.
    mae_test, rmse_test : float
        Mean Absolute Error and Root Mean Squared Error on the test set.
    coef_original_scale : np.ndarray
        Model coefficients transformed back to the original (un-standardised)
        feature space for interpretability.
    intercept_original_scale : float
        Corresponding intercept in original scale.
    y_pred_test : np.ndarray
        Predictions on the test set (kept for residual plots).
    """

    name: str
    model: LinearRegression
    scaler: StandardScaler | None
    feature_names: list[str]
    r2_train: float
    r2_test: float
    mae_test: float
    rmse_test: float
    coef_original_scale: np.ndarray
    intercept_original_scale: float
    y_pred_test: np.ndarray = field(repr=False)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preparation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase & strip column names so downstream code is case-insensitive."""
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the fuel-consumption CSV from a local file path.

    Parameters
    ----------
    path : Path
        Path to the local CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with normalised column names.
    """
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    return _normalize_column_names(df)


def prepare_data(
    df: pd.DataFrame,
    drop_categorical: list[str] | None = None,
    drop_numeric: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, list[str], str]:
    """Drop irrelevant columns and return (cleaned_df, feature_cols, target_col).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (column names should already be normalised).
    drop_categorical : list[str], optional
        Columns to drop (defaults to :data:`config.DROP_CATEGORICAL`).
    drop_numeric : list[str], optional
        Redundant numeric columns to drop (defaults to
        :data:`config.DROP_NUMERIC_REDUNDANT`).
    target_column : str
        Name of the target column (case-insensitive match).

    Returns
    -------
    tuple[pd.DataFrame, list[str], str]
        ``(df_clean, feature_column_names, resolved_target_name)``
    """
    drop_categorical = drop_categorical or DROP_CATEGORICAL
    drop_numeric = drop_numeric or DROP_NUMERIC_REDUNDANT

    to_drop = [c for c in drop_categorical + drop_numeric if c in df.columns]
    df_clean = df.drop(columns=to_drop, errors="ignore")

    # Resolve target by case-insensitive match
    target: str | None = None
    for col in df_clean.columns:
        if col.upper() == target_column.upper():
            target = col
            break
    if target is None:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df_clean.columns)}"
        )

    feature_cols = [c for c in df_clean.columns if c != target]
    df_clean = df_clean[[*feature_cols, target]].astype(float)
    logger.info(
        "Prepared %d rows × %d features  (target=%s)",
        len(df_clean), len(feature_cols), target,
    )
    return df_clean, feature_cols, target


# ──────────────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────────────

def train_simple_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
) -> RegressionResult:
    """Fit a simple linear regression using **only the first feature**.

    The model is trained on standardised features; coefficients are
    converted back to the original scale for interpretability.
    """
    # Use only the first feature column
    X_train_1 = X_train[:, 0:1]
    X_test_1 = X_test[:, 0:1]

    model = LinearRegression()
    model.fit(X_train_1, y_train)
    y_pred = model.predict(X_test_1)

    # Convert coefficients to original (un-standardised) scale
    std_0 = _scaler_std(scaler)[0]
    coef_scaled = float(np.asarray(model.coef_).flatten()[0])
    intercept_scaled = float(np.asarray(model.intercept_).ravel()[0])
    coef_orig = coef_scaled / std_0
    intercept_orig = intercept_scaled - (float(scaler.mean_[0]) * coef_scaled) / std_0

    return RegressionResult(
        name="Simple (1 feature)",
        model=model,
        scaler=scaler,
        feature_names=feature_names[:1],
        r2_train=r2_score(y_train, model.predict(X_train_1)),
        r2_test=r2_score(y_test, y_pred),
        mae_test=float(mean_absolute_error(y_test, y_pred)),
        rmse_test=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        coef_original_scale=np.array([coef_orig]),
        intercept_original_scale=intercept_orig,
        y_pred_test=y_pred,
    )


def train_multiple_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    feature_names: list[str],
) -> RegressionResult:
    """Fit a multiple linear regression using **all provided features**.

    Coefficients are converted back to the original scale.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert coefficients to original (un-standardised) scale
    std_devs = _scaler_std(scaler)
    coef_scaled = np.asarray(model.coef_).flatten()
    intercept_scaled = float(np.asarray(model.intercept_).ravel()[0])
    coef_orig = coef_scaled / std_devs
    intercept_orig = intercept_scaled - float(np.sum(scaler.mean_ * coef_scaled / std_devs))

    return RegressionResult(
        name="Multiple (2 features)",
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        r2_train=r2_score(y_train, model.predict(X_train)),
        r2_test=r2_score(y_test, y_pred),
        mae_test=float(mean_absolute_error(y_test, y_pred)),
        rmse_test=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        coef_original_scale=coef_orig,
        intercept_original_scale=intercept_orig,
        y_pred_test=y_pred,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    path: Path = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,       # df_clean
    pd.DataFrame,       # df_metrics
    RegressionResult,   # simple_result
    RegressionResult,   # multiple_result
    np.ndarray,         # X_train  (scaled)
    np.ndarray,         # X_test   (scaled)
    np.ndarray,         # y_train
    np.ndarray,         # y_test
    StandardScaler,     # scaler
    list[str],          # feature_cols
]:
    """End-to-end pipeline: load → clean → split → scale → train → compare.

    Parameters
    ----------
    path : Path
        Local CSV file path for the raw dataset.
    test_size : float
        Fraction of data for the test set.
    random_state : int
        Seed for reproducible train/test splits.

    Returns
    -------
    tuple
        ``(df_clean, df_metrics, simple_result, multiple_result,
        X_train, X_test, y_train, y_test, scaler, feature_cols)``
    """
    # 1. Load & prepare
    df = load_data(path=path)
    df_clean, feature_cols, target = prepare_data(df)

    # 2. Split & scale
    X = df_clean[feature_cols].to_numpy()
    y = df_clean[target].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state,
    )
    logger.info(
        "Train/test split: %d / %d samples (test_size=%.2f)",
        len(y_train), len(y_test), test_size,
    )

    # 3. Train both models
    simple_result = train_simple_regression(
        X_train, X_test, y_train, y_test, scaler, feature_cols,
    )
    multiple_result = train_multiple_regression(
        X_train, X_test, y_train, y_test, scaler, feature_cols,
    )
    logger.info(
        "Simple R²=%.4f | Multiple R²=%.4f (test)",
        simple_result.r2_test, multiple_result.r2_test,
    )

    # 4. Build comparison table
    df_metrics = pd.DataFrame([
        {
            "Model": simple_result.name,
            "R² (train)": round(simple_result.r2_train, 4),
            "R² (test)": round(simple_result.r2_test, 4),
            "MAE (test)": round(simple_result.mae_test, 4),
            "RMSE (test)": round(simple_result.rmse_test, 4),
        },
        {
            "Model": multiple_result.name,
            "R² (train)": round(multiple_result.r2_train, 4),
            "R² (test)": round(multiple_result.r2_test, 4),
            "MAE (test)": round(multiple_result.mae_test, 4),
            "RMSE (test)": round(multiple_result.rmse_test, 4),
        },
    ])

    return (
        df_clean, df_metrics, simple_result, multiple_result,
        X_train, X_test, y_train, y_test, scaler, feature_cols,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Comparison helper
# ──────────────────────────────────────────────────────────────────────────────

def get_comparison_summary(
    simple: RegressionResult,
    multiple: RegressionResult,
) -> dict[str, Any]:
    """Return a dict summarising how multiple regression compares to simple.

    Keys
    ----
    r2_improvement : float
        ``multiple.r2_test − simple.r2_test`` (positive = multiple is better).
    mae_reduction : float
        ``simple.mae_test − multiple.mae_test`` (positive = multiple is better).
    rmse_reduction : float
        ``simple.rmse_test − multiple.rmse_test``.
    better_model : str
        ``"Multiple"`` or ``"Simple"`` based on test R².
    """
    return {
        "r2_improvement": multiple.r2_test - simple.r2_test,
        "mae_reduction": simple.mae_test - multiple.mae_test,
        "rmse_reduction": simple.rmse_test - multiple.rmse_test,
        "better_model": "Multiple" if multiple.r2_test > simple.r2_test else "Simple",
    }
