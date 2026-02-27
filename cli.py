"""
Command-line interface for Fuel Consumption → CO2 Emissions regression.

Trains simple (1 feature) vs multiple (2 feature) linear regression models,
prints a metrics comparison table, and optionally saves diagnostic figures.

Usage examples
--------------
    # Print metrics only
    python cli.py

    # Save figures to a folder (headless)
    python cli.py --output-dir output

    # Show interactive matplotlib windows
    python cli.py --show --test-size 0.25

    # Run as package module
    python -m fuel_consumption.cli --output-dir output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve imports whether run as a script or as a package module
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR.parent))

from fuel_model import get_comparison_summary, run_pipeline  # noqa: E402
from plots import (  # noqa: E402
    plot_multiple_3d,
    plot_residuals,
    plot_simple_vs_feature,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Build and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train simple vs multiple linear regression on the Fuel "
            "Consumption dataset and compare metrics."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save comparison figures (default: no save).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib figure windows interactively.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test set (default: 0.2).",
    )
    return parser.parse_args()


def _print_metrics(df_metrics, summary: dict, feature_cols: list[str],
                   simple_result, multiple_result) -> None:
    """Print formatted metrics table and comparison summary to stdout."""
    print("\n" + "=" * 60)
    print("  Simple vs Multiple Regression — Metrics Comparison")
    print("=" * 60)
    print(df_metrics.to_string(index=False))
    print()

    # Comparison summary
    print("Comparison summary:")
    print(f"  R² improvement (multiple vs simple) : {summary['r2_improvement']:+.4f}")
    print(f"  MAE reduction  (multiple)           : {summary['mae_reduction']:+.4f}")
    print(f"  RMSE reduction (multiple)           : {summary['rmse_reduction']:+.4f}")
    print(f"  Better model   (by test R²)         : {summary['better_model']}")
    print()

    # Coefficients in original (un-standardized) scale
    print("Simple regression coefficients (original scale):")
    print(f"  {feature_cols[0]}: {simple_result.coef_original_scale[0]:.4f}")
    print(f"  Intercept     : {simple_result.intercept_original_scale:.4f}")
    print()

    print("Multiple regression coefficients (original scale):")
    for name, coef in zip(feature_cols, multiple_result.coef_original_scale):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept     : {multiple_result.intercept_original_scale:.4f}")
    print()


def _save_figures(args: argparse.Namespace, X_train, X_test, y_train, y_test,
                  simple_result, multiple_result, feature_cols) -> None:
    """Generate and optionally save / display diagnostic figures."""
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")  # headless backend when not showing windows
    import matplotlib.pyplot as plt

    out_dir = args.output_dir or Path(".")
    if args.output_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "simple_regression_feature1.png": plot_simple_vs_feature(
            X_train, y_train, simple_result, feature_cols[0],
        ),
        "multiple_regression_3d.png": plot_multiple_3d(
            X_test, y_test, multiple_result, feature_cols,
        ),
        "residuals_simple.png": plot_residuals(
            y_test, simple_result.y_pred_test, simple_result.name,
        ),
        "residuals_multiple.png": plot_residuals(
            y_test, multiple_result.y_pred_test, multiple_result.name,
        ),
    }

    for filename, fig in figures.items():
        if args.output_dir:
            fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", out_dir / filename)
        if args.show:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the full training pipeline, print results, and save figures."""
    args = _parse_args()

    # Run training pipeline
    (
        _df_clean,
        df_metrics,
        simple_result,
        multiple_result,
        X_train,
        X_test,
        y_train,
        y_test,
        _scaler,
        feature_cols,
    ) = run_pipeline(test_size=args.test_size)

    # Print metrics & coefficients
    summary = get_comparison_summary(simple_result, multiple_result)
    _print_metrics(df_metrics, summary, feature_cols, simple_result, multiple_result)

    # Figures (only if requested)
    if args.output_dir is not None or args.show:
        _save_figures(
            args, X_train, X_test, y_train, y_test,
            simple_result, multiple_result, feature_cols,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
