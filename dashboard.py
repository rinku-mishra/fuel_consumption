"""
Streamlit dashboard — Fuel Consumption → CO₂ Emissions.

Interactive comparison of Simple (1 feature) vs Multiple (2 feature) linear
regression on the Fuel Consumption dataset.

Launch
------
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the package directory is on sys.path so sibling imports work
# regardless of how Streamlit launches the script.
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fuel_model import get_comparison_summary, run_pipeline
from plots import plot_multiple_3d, plot_residuals, plot_simple_vs_feature

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fuel Consumption → CO₂",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — lightweight, scoped to metrics & layout
# ──────────────────────────────────────────────────────────────────────────────
_CUSTOM_CSS = """
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a4d2e, #2d6a4f);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] { color: #b7e4c7; }
    div[data-testid="stMetricLabel"] { color: #d8f3dc; }

    /* Section headers */
    .section-header {
        border-bottom: 2px solid #2d6a4f;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem;
    }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached pipeline — avoids re-training on every widget interaction
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_pipeline(test_size: float):
    """Run the full training pipeline; results are cached by *test_size*.

    Streamlit's ``@st.cache_data`` serialises the return value so the heavy
    computation (data download + model training) only happens once per unique
    ``test_size``.
    """
    (
        df_clean,
        df_metrics,
        simple_result,
        multiple_result,
        X_train,
        X_test,
        y_train,
        y_test,
        _scaler,
        feature_cols,
    ) = run_pipeline(test_size=test_size)

    return {
        "df_clean": df_clean,
        "df_metrics": df_metrics,
        "simple_result": simple_result,
        "multiple_result": multiple_result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tab renderers — each tab is its own function for readability
# ──────────────────────────────────────────────────────────────────────────────

def _render_metrics_tab(df_metrics, simple_result, multiple_result) -> None:
    """Tab 1 — side-by-side metric cards + interpretation text."""
    st.markdown("### Model Comparison — Key Metrics")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    summary = get_comparison_summary(simple_result, multiple_result)

    st.markdown("#### Interpretation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "R² improvement",
            f"{summary['r2_improvement']:+.4f}",
            delta="better" if summary["r2_improvement"] > 0 else "worse",
        )
    with c2:
        st.metric(
            "MAE reduction",
            f"{summary['mae_reduction']:+.4f}",
            delta="lower error" if summary["mae_reduction"] > 0 else "higher error",
        )
    with c3:
        st.metric("Best model (test R²)", summary["better_model"])

    st.info(
        "**Simple regression** uses one predictor (e.g. engine size). "
        "**Multiple regression** adds a second feature (e.g. combined fuel consumption in MPG). "
        "A higher test R² and lower MAE / RMSE indicate better generalisation.",
        icon="💡",
    )


def _render_coefficients_tab(feature_cols, simple_result, multiple_result,
                             y_test) -> None:
    """Tab 2 — coefficients in original scale + residual comparison."""
    st.markdown("### Coefficients (original scale)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Simple (1 feature)**")
        st.code(
            f"Coefficient ({feature_cols[0]}): "
            f"{simple_result.coef_original_scale[0]:.4f}\n"
            f"Intercept: {simple_result.intercept_original_scale:.4f}",
            language="text",
        )
    with col2:
        st.markdown("**Multiple (2 features)**")
        lines = [
            f"Coefficient ({name}): {c:.4f}"
            for name, c in zip(feature_cols, multiple_result.coef_original_scale)
        ]
        lines.append(f"Intercept: {multiple_result.intercept_original_scale:.4f}")
        st.code("\n".join(lines), language="text")

    # Residual plots side-by-side
    st.markdown("### Residuals Comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_residuals(y_test, simple_result.y_pred_test, simple_result.name, ax=ax1)
    plot_residuals(y_test, multiple_result.y_pred_test, multiple_result.name, ax=ax2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_plots_tab(X_train, X_test, y_train, y_test,
                      simple_result, multiple_result, feature_cols) -> None:
    """Tab 3 — 2-D regression line and 3-D regression plane."""
    st.markdown("### Simple Regression — Feature vs CO₂")
    fig1 = plot_simple_vs_feature(X_train, y_train, simple_result, feature_cols[0])
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("### Multiple Regression — 3-D Plane")
    fig2 = plot_multiple_3d(X_test, y_test, multiple_result, feature_cols)
    st.pyplot(fig2)
    plt.close(fig2)


def _render_data_tab(df_clean) -> None:
    """Tab 4 — cleaned dataset preview and summary statistics."""
    st.markdown("### Cleaned Dataset (features + target)")
    st.dataframe(df_clean.head(200), use_container_width=True, hide_index=True)
    st.caption(f"Showing first 200 of **{len(df_clean):,}** rows.")

    st.markdown("### Summary Statistics")
    st.dataframe(df_clean.describe(), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.title("⛽ Fuel Consumption → CO₂ Emissions")
    st.caption(
        "Compare **Simple** (1 feature) vs **Multiple** (2 feature) "
        "linear regression on the Fuel Consumption dataset."
    )

    # ── Sidebar controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        test_size = st.slider(
            "Test-set fraction",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            help="Proportion of data held out for evaluation.",
        )
        run_btn = st.button("Load data & train models", type="primary",
                            use_container_width=True)

    # ── Pipeline execution ───────────────────────────────────────────────
    if run_btn:
        st.session_state["trained"] = True
        st.session_state["test_size"] = test_size

    if not st.session_state.get("trained", False):
        st.info(
            "👈 Click **Load data & train models** in the sidebar to begin.",
            icon="🚀",
        )
        return

    # Use the cached pipeline to avoid re-training on tab switches
    active_test_size = st.session_state.get("test_size", 0.2)
    with st.spinner("Running pipeline…"):
        try:
            data = _cached_pipeline(active_test_size)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    st.success("Models trained successfully. Explore the tabs below.", icon="✅")

    # ── Unpack results ───────────────────────────────────────────────────
    df_clean = data["df_clean"]
    df_metrics = data["df_metrics"]
    simple_result = data["simple_result"]
    multiple_result = data["multiple_result"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_cols = data["feature_cols"]

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrics",
        "📈 Coefficients & Residuals",
        "🔬 Visualisations",
        "📋 Data",
    ])

    with tab1:
        _render_metrics_tab(df_metrics, simple_result, multiple_result)
    with tab2:
        _render_coefficients_tab(feature_cols, simple_result, multiple_result,
                                 y_test)
    with tab3:
        _render_plots_tab(X_train, X_test, y_train, y_test,
                          simple_result, multiple_result, feature_cols)
    with tab4:
        _render_data_tab(df_clean)


if __name__ == "__main__":
    main()
