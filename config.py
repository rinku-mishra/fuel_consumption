"""
Configuration constants for the Fuel Consumption → CO2 prediction pipeline.

All tuneable parameters (data source, column lists, train/test split) live
here so they can be changed in one place without touching model or UI code.

Environment variables are **not** used intentionally — this is a data-science
demo, not a web service.  If you later need secrets (e.g. a database URL),
add ``python-dotenv`` and load them here.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
DATA_PATH: Path = Path(__file__).parent / "e10efaa3-a8cc-4072-845a-13e03d996c30.csv"

# ---------------------------------------------------------------------------
# Columns to drop during data preparation
# ---------------------------------------------------------------------------
# Categorical / string columns that linear regression cannot use directly.
DROP_CATEGORICAL: list[str] = [
    "MODELYEAR",
    "MAKE",
    "MODEL",
    "VEHICLECLASS",
    "TRANSMISSION",
    "FUELTYPE",
]

# Numeric columns that are redundant or highly correlated with the features
# we keep (ENGINESIZE, FUELCONSUMPTION_COMB_MPG).
DROP_NUMERIC_REDUNDANT: list[str] = [
    "CYLINDERS",
    "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2       # 20 % held out for evaluation
RANDOM_STATE: int = 42       # reproducible splits

# ---------------------------------------------------------------------------
# Target & features
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "CO2EMISSIONS"

# Feature columns used for multiple regression.
# ``None`` → automatically inferred from the columns remaining after drops.
FEATURE_COLUMNS: list[str] | None = None
