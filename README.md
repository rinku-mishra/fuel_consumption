# ⛽ Fuel Consumption → CO₂ Emissions

> Compare **Simple** (1 feature) vs **Multiple** (2 feature) linear regression
> on the Fuel Consumption CO₂ dataset.

The pipeline reads the local dataset, drops categorical / redundant columns,
standardises features, trains both models, and reports R², MAE, and RMSE on a
held-out test set.  An interactive **Streamlit dashboard** lets you explore
metrics, coefficients, residual plots, and a 3-D regression plane.

---

## 📁 Project Structure

```
fuel_consumption/
├── __init__.py        # Package exports
├── config.py          # Tuneable constants (data path, columns, split ratio)
├── fuel_model.py      # Data loading, preparation, training, comparison
├── plots.py           # Matplotlib visualisations (2-D, 3-D, residuals)
├── dashboard.py       # Streamlit interactive UI
├── cli.py             # Command-line entry point
├── requirements.txt   # Pinned Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

---

## 🚀 Quick Start

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/<your-username>/fuel_consumption.git
cd fuel_consumption
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the CLI (headless)

```bash
# Print metrics table to stdout
python cli.py

# Save diagnostic figures
python cli.py --output-dir output

# Show interactive matplotlib windows + custom test size
python cli.py --show --test-size 0.25
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open the URL printed in the terminal (default: <http://localhost:8501>).

1. Adjust the **test-set fraction** in the sidebar.
2. Click **Load data & train models**.
3. Explore the four tabs: *Metrics*, *Coefficients & Residuals*, *Visualisations*, *Data*.

---

## ⚙️ Configuration

All tuneable parameters live in **`config.py`**:

| Constant | Default | Description |
|---|---|---|
| `DATA_PATH` | Local CSV file | Path to the local dataset file |
| `DROP_CATEGORICAL` | 6 columns | Categorical columns dropped before training |
| `DROP_NUMERIC_REDUNDANT` | 4 columns | Highly correlated numeric columns dropped |
| `TEST_SIZE` | `0.2` | Fraction of data held out for testing |
| `RANDOM_STATE` | `42` | Seed for reproducible splits |
| `TARGET_COLUMN` | `CO2EMISSIONS` | Target variable name |
| `FEATURE_COLUMNS` | `None` (auto) | Override feature list; `None` = infer automatically |

---

## 📊 Models

| Model | Features | Notes |
|---|---|---|
| **Simple Linear Regression** | 1st remaining feature (e.g. `ENGINESIZE`) | Easy to interpret; may underfit |
| **Multiple Linear Regression** | All remaining features (e.g. `ENGINESIZE` + `FUELCONSUMPTION_COMB_MPG`) | Usually better R² and lower error |

Both models are trained on **standardised** features (`StandardScaler`).
Coefficients are converted back to the **original scale** for interpretability.

---

## 🧑‍💻 Developer Guide

### Running as a package module

```bash
python -m fuel_consumption.cli --output-dir output
```

### Importing in your own code

```python
from fuel_consumption import run_pipeline, get_comparison_summary

df_clean, df_metrics, simple, multiple, *_ = run_pipeline(test_size=0.3)
summary = get_comparison_summary(simple, multiple)
print(summary)
```

### Adding a new plot

1. Create a function in `plots.py` that returns a `matplotlib.figure.Figure`.
2. Accept an optional `ax` parameter for embedding in subplots.
3. Import and call it from `dashboard.py` or `cli.py`.

### Code style

- Python ≥ 3.10 (uses `X | Y` union syntax).
- Type hints on all public functions.
- Docstrings follow NumPy style.
- No runtime `try/except ImportError` hacks — imports are deterministic.

---

## 🚢 Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select your repo, branch `main`, and file `dashboard.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

### Docker (self-hosted)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t fuel-co2 .
docker run -p 8501:8501 fuel-co2
```

### Heroku / Railway / Render

Create a `Procfile`:

```
web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
