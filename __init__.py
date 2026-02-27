"""
fuel_consumption — Simple vs Multiple Linear Regression on CO₂ emissions.

This package provides:

- **config** — tuneable constants (data path, column lists, split ratio).
- **fuel_model** — data loading, preparation, model training, and comparison.
- **plots** — matplotlib visualisations (2-D regression, 3-D plane, residuals).
- **dashboard** — Streamlit interactive UI.
- **cli** — command-line entry point for headless / CI usage.
"""

from .config import DATA_PATH, TEST_SIZE
from .fuel_model import (
    RegressionResult,
    get_comparison_summary,
    load_data,
    prepare_data,
    run_pipeline,
    train_multiple_regression,
    train_simple_regression,
)

__all__ = [
    "DATA_PATH",
    "TEST_SIZE",
    "RegressionResult",
    "get_comparison_summary",
    "load_data",
    "prepare_data",
    "run_pipeline",
    "train_multiple_regression",
    "train_simple_regression",
]
