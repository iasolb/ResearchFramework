# Research Framework

A lightweight pandas-based framework for research workflows. Manages datasets and working subsets, tracks dependent/independent/control variables, provides clean interfaces for transforming columns, and includes a full Monte Carlo simulation module for uncertainty analysis.

## Installation

Core dependencies:

```bash
pip install pandas numpy geopandas
```

For the simulation module:

```bash
pip install scipy matplotlib
```

For running the example workflows:

```bash
pip install statsmodels scikit-learn
```

For running the tests:

```bash
pip install pytest
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
ResearchFramework/
├── ResearchHandler.py     # Core data handling class + ModelSpec
├── transforms.py          # Reusable single- and multi-column transforms
├── simulation.py          # Monte Carlo simulation module
├── requirements.txt       # Dependencies (core + optional)
├── LICENSE                # MIT
├── .gitignore
├── README.md
├── tests/
│   └── test_handler.py    # pytest suite with synthetic data
└── examples/
    ├── data/
    │   └── startup_portfolio.csv      # Sample portfolio dataset
    ├── ols_mincer.py                  # OLS Mincer wage equation
    ├── random_forest_churn.py         # Random forest churn prediction
    ├── heckman_selection.py           # Heckman two-step selection model
    └── monte_carlo_portfolio.py       # Monte Carlo portfolio valuation
```

## Quick Start

```python
import numpy as np
from ResearchHandler import ResearchHandler
from transforms import mean_center, log_transform, z_score

def clean(df):
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna(subset=["income", "age", "education"])
    df["female"] = (df["gender"] == "F").astype(int)
    return df

# Initialize from a CSV with a cleaning function
rh = ResearchHandler("survey_data.csv", clean)

# Transform with named functions from transforms.py
rh.normalize_and_attach("income", log_transform, "log_income")
rh.normalize_and_attach("age", mean_center, "age_centered")

# Create a working subset
rh.create_subset(lambda df: (df["age"] >= 18) & (df["employed"] == 1))

# Set up variables from the subset
rh.set_dependent("log_income", full=False)
rh.add_independents("age_centered", "education", full=False)
rh.add_controls("female", full=False)

# Retrieve design matrix and outcome vector
X = rh.get_X()
y = rh.get_y()

# Or get a frozen snapshot with full metadata
spec = rh.get_spec()
```

## ResearchHandler API

### `ResearchHandler(source, handler=None, *, shapefile=False)`

Constructor. Accepts a CSV filepath, shapefile path, DataFrame, or GeoDataFrame. The optional `handler` function transforms the data after loading.

```python
# From a CSV with a cleaning function
def clean(df):
    df.columns = df.columns.str.lower()
    df["married"] = (df["marital_status"] == "married").astype(int)
    df = df.drop_duplicates()
    return df.dropna()

rh = ResearchHandler("data.csv", clean)

# From a CSV without cleaning
rh = ResearchHandler("data.csv")

# From a shapefile
rh = ResearchHandler("regions.shp", shapefile=True)

# From an existing DataFrame or GeoDataFrame
rh = ResearchHandler(existing_df)
rh = ResearchHandler(existing_df, clean)
```

The `handler` function receives a `pd.DataFrame` (or `gpd.GeoDataFrame` for shapefiles) and must return one. If the source type is unsupported, a `TypeError` is raised. The `shapefile` parameter is keyword-only.

### `create_subset(condition)`

Creates a working subset of the full dataset based on a boolean condition.

```python
rh.create_subset(lambda df: df["age"] > 30)
rh.create_subset(lambda df: (df["income"] > 20000) & (df["employed"] == 1))
rh.create_subset(lambda df: df["country"].isin(["US", "UK", "CA"]))
```

### `reset_subset()`

Clears the working subset back to `None`.

### `set_dependent(col, full=True)`

Sets the dependent (outcome) variable. Locks the source mode (see Design Notes).

```python
rh.set_dependent("log_income")
rh.set_dependent("log_income", full=False)
```

### `add_independents(*cols, full=True)`

Adds one or more independent (predictor) variables.

```python
rh.add_independents("education", "experience", "tenure")
rh.add_independents("education", "experience", full=False)
```

### `add_controls(*cols, full=True)`

Adds one or more control variables.

```python
rh.add_controls("female", "married", "region_code")
rh.add_controls("female", "married", full=False)
```

### `get_X()` / `get_y()`

Returns the design matrix as a `pd.DataFrame` or the dependent variable as a `pd.Series`.

```python
X = rh.get_X()
y = rh.get_y()
```

### `get_spec()`

Returns a frozen `ModelSpec` snapshot of the current variable specification. Contains copies of the design matrix, dependent variable, column name metadata, and the source DataFrame. Nothing mutates after creation.

```python
spec = rh.get_spec()

spec.X              # DataFrame — same as get_X()
spec.y              # Series — same as get_y()
spec.independents   # ("education", "experience")
spec.controls       # ("female",)
spec.dependent      # "log_income"
spec.columns        # ("education", "experience", "female")
spec.all_columns    # ("log_income", "education", "experience", "female")
spec.source_label   # "full" or "subset"
spec.n              # number of observations
spec.data           # copy of the source DataFrame (for distribution fitting)
```

`ModelSpec` is the bridge between `ResearchHandler` and the simulation module — pass it to `Simulation.from_spec()` to build a data-driven Monte Carlo simulation.

### `attach(col_name, series, to_full=True, quiet=False)`

Attaches a precomputed Series to the full dataset or subset.

```python
from transforms import square

rh.attach("age_sq", square(rh.data["age"]))
rh.attach("age_sq", square(rh.subset["age"]), to_full=False)
```

### `normalize_and_attach(source_col, normalizing_function, new_colname, full=True)`

Applies a single-column transformation and attaches the result.

```python
from transforms import log_transform, z_score, mean_center, min_max_scale

rh.normalize_and_attach("income", log_transform, "log_income")
rh.normalize_and_attach("gpa", z_score, "gpa_z")
rh.normalize_and_attach("age", mean_center, "age_centered")
rh.normalize_and_attach("score", min_max_scale, "score_scaled")
rh.normalize_and_attach("wage", log_transform, "log_wage", full=False)
```

### `calculate_and_attach(source_cols, func, new_colname, full=True)`

Applies a multi-column transformation and attaches the result. The function receives a DataFrame subset of the specified columns.

```python
from transforms import interaction, row_mean, row_sum, safe_ratio

rh.calculate_and_attach(["education", "experience"], interaction, "edu_x_exp")
rh.calculate_and_attach(["math", "reading", "science"], row_mean, "avg_score")
rh.calculate_and_attach(["q1", "q2", "q3", "q4"], row_sum, "annual_total")
rh.calculate_and_attach(
    ["revenue", "visits"],
    safe_ratio("revenue", "visits"),
    "rev_per_visit",
    full=False
)
```

### `clear_caches()`

Clears the dependent, independents, controls, and source mode lock so you can set up a new specification without reinitializing.

```python
rh.clear_caches()
```

## Transforms Reference

`transforms.py` provides reusable functions so you don't have to write lambdas inline every time.

### Single-column transforms (Series → Series)

For use with `normalize_and_attach`:

| Function | Description | Example |
|----------|-------------|---------|
| `mean_center` | `x - mean(x)` | `rh.normalize_and_attach("age", mean_center, "age_c")` |
| `z_score` | `(x - mean) / std` | `rh.normalize_and_attach("gpa", z_score, "gpa_z")` |
| `min_max_scale` | Scale to [0, 1] | `rh.normalize_and_attach("score", min_max_scale, "score_01")` |
| `log_transform` | `ln(x)` | `rh.normalize_and_attach("income", log_transform, "log_inc")` |
| `log1p_transform` | `ln(1 + x)`, safe for zeros | `rh.normalize_and_attach("tickets", log1p_transform, "log_tix")` |
| `square` | `x²` | `rh.normalize_and_attach("exp", square, "exp_sq")` |
| `rank_transform` | Replace with rank | `rh.normalize_and_attach("score", rank_transform, "score_rank")` |

### Factory transforms (return a callable)

| Function | Description | Example |
|----------|-------------|---------|
| `winsorize(lower, upper)` | Clip at quantiles | `rh.normalize_and_attach("income", winsorize(0.01, 0.99), "inc_wins")` |
| `demean_by_group(group_col)` | Subtract group means | `rh.normalize_and_attach("income", demean_by_group(rh.data["industry"]), "inc_dm")` |

### Multi-column transforms (DataFrame → Series)

For use with `calculate_and_attach`:

| Function | Description | Example |
|----------|-------------|---------|
| `interaction` | Product of first two columns | `rh.calculate_and_attach(["edu", "exp"], interaction, "edu_x_exp")` |
| `row_mean` | Row-wise average | `rh.calculate_and_attach(["m", "r", "s"], row_mean, "avg")` |
| `row_sum` | Row-wise sum | `rh.calculate_and_attach(["q1", "q2"], row_sum, "total")` |
| `safe_ratio(num, denom)` | Division, 0 → NaN | `rh.calculate_and_attach(["rev", "vis"], safe_ratio("rev", "vis"), "rpv")` |

---

## Simulation Module

`simulation.py` provides a Monte Carlo simulation framework for running models under uncertainty. It integrates with `ResearchHandler` via `ModelSpec`, or can be used standalone with manually specified distributions.

### Simulation Quick Start

```python
from simulation import Simulation, DistributionSpec

sim = Simulation(
    variables=[
        DistributionSpec("revenue", "normal", {"mean": 1_000_000, "std": 200_000}),
        DistributionSpec("cost",    "uniform", {"low": 400_000, "high": 700_000}),
    ],
    model=lambda row: row["revenue"] - row["cost"],
    n_iterations=10_000,
    seed=42,
)

result = sim.run()
print(f"Mean:   ${result.mean:,.0f}")
print(f"95% CI: [${result.ci_lower:,.0f}, ${result.ci_upper:,.0f}]")
```

### Using with ResearchHandler

The primary integration path is `get_spec()` + `Simulation.from_spec()`. This fits distributions from your observed data, infers the correlation structure, and returns a ready-to-run simulation — no manual `InputManager` wiring needed.

```python
from ResearchHandler import ResearchHandler
from simulation import Simulation

rh = ResearchHandler("labor_data.csv", clean)
rh.normalize_and_attach("income", log_transform, "log_income")

rh.set_dependent("log_income")
rh.add_independents("education", "experience")
rh.add_controls("female")

spec = rh.get_spec()

# Fit distributions from the observed data and simulate
sim = Simulation.from_spec(
    spec,
    model=lambda row: 8.0 + 0.08 * row["education"] + 0.02 * row["experience"],
    n_iterations=10_000,
    seed=42,
)
result = sim.run()
```

Override the distribution family per column when the default doesn't fit:

```python
sim = Simulation.from_spec(
    spec,
    model=my_model,
    dist_type="normal",                                     # default for all columns
    overrides={"income": {"dist_type": "lognormal"},        # income is right-skewed
               "satisfaction": {"dist_type": "empirical"}},  # resample directly
)
```

Draw from the joint distribution without a model (useful for bootstrapping):

```python
sim = Simulation.from_spec(spec, include_dependent=True, dist_type="empirical")
result = sim.engine.run()
result.draws  # DataFrame with correlated draws of all variables including dependent
```

Note: `include_dependent=True` requires `model=None` — the two modes are mutually exclusive.

For lower-level control you can still use `InputManager` directly:

```python
from simulation import InputManager, DistributionSpec, ModelFunction, MonteCarloEngine

mgr = InputManager()
mgr.fit_from_data(rh.data, ["growth_rate", "churn_rate"], dist_type="normal")
mgr.infer_correlation_from_data(rh.data)
mgr.add_variable(DistributionSpec("subsidy", "uniform", {"low": 0, "high": 50_000}))

model = ModelFunction(lambda row: row["growth_rate"] * 100_000 + row["subsidy"])
engine = MonteCarloEngine(mgr, model, n_iterations=10_000, seed=42)
result = engine.run()
result.summarize()
```

### Simulation API Reference

#### `DistributionSpec(name, dist_type, params, empirical_data=None)`

Defines an uncertain variable and its probability distribution. Validation happens on construction — missing params or unknown distribution types raise immediately.

| `dist_type` | Required `params` |
|-------------|-------------------|
| `"normal"` | `{"mean": ..., "std": ...}` |
| `"uniform"` | `{"low": ..., "high": ...}` |
| `"lognormal"` | `{"mean": ..., "sigma": ...}` |
| `"beta"` | `{"a": ..., "b": ...}` |
| `"triangular"` | `{"left": ..., "mode": ..., "right": ...}` |
| `"exponential"` | `{"scale": ...}` |
| `"empirical"` | `empirical_data=np.array([...])` |

```python
DistributionSpec("revenue",  "normal",   {"mean": 1e6, "std": 2e5})
DistributionSpec("cost",     "uniform",  {"low": 4e5, "high": 7e5})
DistributionSpec("duration", "empirical", empirical_data=observed_array)
```

#### `InputManager`

Collects uncertain variables, fits distributions from data, manages correlation, and draws samples.

```python
mgr = InputManager()

# Register manually
mgr.add_variable(DistributionSpec("x", "normal", {"mean": 0, "std": 1}))
mgr.add_variables([...])
mgr.remove_variable("x")

# Fit from observed data
mgr.fit_from_data(df, ["col_a", "col_b"], dist_type="normal")
mgr.fit_from_data(df, ["col_c"], dist_type="empirical")

# Correlation
mgr.set_correlation_matrix(np.array([[1.0, 0.6], [0.6, 1.0]]))
mgr.infer_correlation_from_data(df)

# Draw samples — returns DataFrame of shape (n, n_variables)
draws = mgr.draw(10_000, seed=42)
```

When a correlation matrix is set, draws use a Gaussian copula (Cholesky decomposition + inverse-CDF transform) to produce correlated samples with the correct marginal distributions. Without a correlation matrix, draws are independent.

#### `ModelFunction(func, vectorized=False)`

Wraps the user-supplied model function.

```python
# Row-wise: receives a pd.Series per iteration
model = ModelFunction(lambda row: row["revenue"] - row["cost"])

# Vectorized: receives the full DataFrame, returns an array (faster)
model = ModelFunction(
    lambda df: (df["revenue"] - df["cost"]).values,
    vectorized=True,
)

# Multi-output: return a dict per row
def multi(row):
    profit = row["revenue"] - row["cost"]
    return {"profit": profit, "margin": profit / row["revenue"]}
model = ModelFunction(multi)
```

#### `MonteCarloEngine(inputs, model, n_iterations=10_000, seed=None)`

Runs the simulation loop.

```python
engine = MonteCarloEngine(mgr, model, n_iterations=10_000, seed=42)
result = engine.run()
result = engine.run(store_draws=False)  # save memory on large runs
```

#### `SimulationResult`

Container for outcomes and summary statistics.

```python
result.outcomes         # np.ndarray of model outputs
result.draws            # DataFrame of input draws (if stored)

result.summarize()      # compute and cache all stats, returns dict
result.mean             # cached after summarize()
result.median
result.std
result.ci_lower         # 95% CI by default
result.ci_upper
result.percentiles      # {1: ..., 5: ..., 10: ..., 25: ..., 50: ..., 75: ..., 90: ..., 95: ..., 99: ...}

result.to_dataframe()   # draws + outcomes in one exportable DataFrame
```

#### `Simulation(variables, model, *, n_iterations=10_000, seed=None, ...)`

Top-level facade that wires everything together. Use this with manual `DistributionSpec` lists, or use `Simulation.from_spec()` with a `ModelSpec` from `ResearchHandler`.

```python
sim = Simulation(
    variables=[
        DistributionSpec("growth",   "normal",     {"mean": 0.4, "std": 0.15}),
        DistributionSpec("churn",    "beta",        {"a": 2, "b": 30}),
        DistributionSpec("multiple", "triangular",  {"left": 3, "mode": 8, "right": 20}),
    ],
    model=portfolio_value,
    n_iterations=10_000,
    seed=42,
    correlation_matrix=corr_matrix,  # optional
)

result = sim.run()
```

The facade exposes sub-components directly:

```python
sim.engine                  # MonteCarloEngine
sim.input_manager           # InputManager
sim.sensitivity             # SensitivityAnalyzer
sim.convergence             # ConvergenceDiagnostics (class reference)
sim.plot                    # SimulationPlotter
```

#### `Simulation.from_spec(spec, model, *, dist_type, overrides, include_dependent, ...)`

Builds a `Simulation` from a `ModelSpec`. Fits distributions from the spec's observed data, infers the correlation matrix, and returns a ready-to-run simulation.

Two modes:

- **Standard** (default): `model` provided, `include_dependent=False`. Fits distributions on independents + controls. The model function produces outcomes from simulated inputs.
- **Joint distribution**: `model=None`, `include_dependent=True`. Fits distributions on all variables including dependent. Returns correlated draws with no model applied.

Passing both `model` and `include_dependent=True` raises `ValueError`.

```python
# Standard mode
sim = Simulation.from_spec(spec, model=my_func, seed=42)

# With per-column overrides
sim = Simulation.from_spec(
    spec, model=my_func,
    overrides={"income": {"dist_type": "lognormal"}},
)

# Joint distribution mode
sim = Simulation.from_spec(spec, include_dependent=True, dist_type="empirical")
```

### Sensitivity Analysis

Accessed via `sim.sensitivity` or by constructing `SensitivityAnalyzer(engine)` directly.

```python
# Tornado: which variable drives the most swing?
tornado = sim.sensitivity.tornado()
# Returns DataFrame: variable, low_value, high_value, low_outcome, high_outcome, swing
# Sorted by swing descending

# One-at-a-time: sweep a single variable across its range
oat = sim.sensitivity.one_at_a_time("revenue", n_steps=20)
# Returns DataFrame: variable_value, outcome

# Sobol indices: variance-based global sensitivity
sobol = sim.sensitivity.sobol_indices(n_samples=5_000, seed=99)
# Returns DataFrame: variable, S1, S1_conf — sorted by S1 descending
```

### Scenario Comparison

Define named scenarios with distribution parameter overrides, then compare outcomes against baseline:

```python
from simulation import Scenario

scenarios = [
    Scenario("bull_market", overrides={
        "multiple": {"left": 8, "mode": 15, "right": 30},
    }),
    Scenario("bear_market", overrides={
        "multiple": {"left": 2, "mode": 4, "right": 8},
        "growth": {"mean": 0.20},
    }),
]

# Full results
results = sim.compare_scenarios(scenarios)
# Returns {"baseline": SimulationResult, "bull_market": ..., "bear_market": ...}

# Summary table
summary = sim.compare_scenarios_summary(scenarios)
# Returns DataFrame: scenario, mean, median, std, ci_lower, ci_upper, min, max
```

Only the parameters that differ need to be specified — everything else stays at the base case.

### Convergence Diagnostics

Check whether the simulation ran enough iterations:

```python
# Quick report
report = sim.check_convergence(result)
# {"is_converged": True, "relative_se": 0.0051, "current_n": 10000, "suggested_n": 39261}

# Detailed: running statistics
running = ConvergenceDiagnostics.running_statistics(result.outcomes)
# DataFrame: iteration, cumulative_mean, cumulative_std

# Did it converge?
ConvergenceDiagnostics.is_converged(result.outcomes, window=1000, tolerance=0.01)

# How many iterations do I need for 0.5% precision?
ConvergenceDiagnostics.suggest_n(result.outcomes, target_tolerance=0.005)

# Snapshots at increasing N
snapshots = sim.engine.run_convergence()
# [SimulationResult(n=100), ..., SimulationResult(n=10000)] — all pre-summarized
```

### Plotting

All plot methods return matplotlib `Figure` objects. Accessed via `sim.plot` or `SimulationPlotter` directly.

```python
fig = sim.plot.histogram(result)                        # distribution with CI shading
fig = sim.plot.cumulative_density(result)               # empirical CDF with percentile markers
fig = sim.plot.convergence_plot(result.outcomes)         # running mean ± SE
fig = sim.plot.tornado_chart(tornado_data)               # sensitivity swings
fig = sim.plot.scenario_comparison(scenario_results)     # overlaid KDE curves

fig.savefig("output.png", dpi=150)
```

### Supported Distributions

New distributions can be added by inserting an entry into `_DISTRIBUTION_REGISTRY` at the top of `simulation.py`. Each entry defines how to draw samples, validate parameters, fit from data, and transform through the inverse-CDF for correlated draws. No other code changes are required.

---

## Example Workflows

All examples in `examples/` generate their own synthetic data so you can clone and run immediately:

```bash
python examples/ols_mincer.py
python examples/random_forest_churn.py
python examples/heckman_selection.py
python examples/monte_carlo_portfolio.py
```

### OLS Regression with statsmodels

A standard Mincer wage equation with log wages, centered experience, and a squared term.

```python
import numpy as np
import statsmodels.api as sm
from ResearchHandler import ResearchHandler
from transforms import log_transform, mean_center, square

def clean(df):
    df.columns = df.columns.str.lower()
    df["female"] = (df["gender"] == "F").astype(int)
    return df.dropna(subset=["wage", "education", "experience", "age", "gender"])

rh = ResearchHandler("labor_data.csv", clean)

rh.normalize_and_attach("wage", log_transform, "log_wage")
rh.normalize_and_attach("experience", mean_center, "exp_centered")
rh.attach("exp_centered_sq", square(rh.data["exp_centered"]))

rh.set_dependent("log_wage")
rh.add_independents("education", "exp_centered", "exp_centered_sq")
rh.add_controls("female")

X = sm.add_constant(rh.get_X())
y = rh.get_y()

model = sm.OLS(y, X).fit()
print(model.summary())
```

### Random Forest with scikit-learn

Predicting customer churn with engineered features and standardized inputs.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ResearchHandler import ResearchHandler
from transforms import z_score, log1p_transform, safe_ratio

rh = ResearchHandler("customer_data.csv", clean)

rh.calculate_and_attach(["revenue", "visits"], safe_ratio("revenue", "visits"), "rev_per_visit")
rh.normalize_and_attach("tenure", z_score, "tenure_z")
rh.normalize_and_attach("support_tickets", log1p_transform, "log_tickets")

rh.set_dependent("churned")
rh.add_independents("rev_per_visit", "tenure_z", "log_tickets")
rh.add_controls("gender_code", "region_code")

X = rh.get_X().fillna(0)
y = rh.get_y()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
```

### Heckman Selection Model (Two-Step)

Correct for selection bias in observed wages using the inverse Mills ratio.

```python
import statsmodels.api as sm
from scipy.stats import norm
from ResearchHandler import ResearchHandler
from transforms import mean_center, log_transform

rh = ResearchHandler("labor_survey.csv", clean)
rh.normalize_and_attach("age", mean_center, "age_centered")

# Step 1: Probit on full sample
rh.set_dependent("employed")
rh.add_independents("age_centered", "education")
rh.add_controls("married", "children")

probit = sm.Probit(rh.get_y(), sm.add_constant(rh.get_X())).fit(disp=0)
rh.attach("imr", norm.pdf(probit.fittedvalues) / norm.cdf(probit.fittedvalues))

# Step 2: OLS on employed subset with IMR correction
rh.clear_caches()
rh.create_subset(lambda df: df["employed"] == 1)
rh.normalize_and_attach("wage", log_transform, "log_wage", full=False)

rh.set_dependent("log_wage", full=False)
rh.add_independents("age_centered", "education", full=False)
rh.add_controls("imr", full=False)

ols = sm.OLS(rh.get_y(), sm.add_constant(rh.get_X())).fit()
print(ols.summary())
```

### Monte Carlo Portfolio Simulation

Simulate a VC portfolio's 3-year value under uncertainty about growth, churn, market multiples, and discount rates.

```python
from simulation import Simulation, DistributionSpec, Scenario

def portfolio_value(row):
    base_arr = 33.0 * 12
    net_growth = row["growth_rate"] - row["churn_rate"]
    projected_arr = base_arr * (1 + net_growth) ** 3
    terminal = projected_arr * row["revenue_multiple"]
    return terminal / (1 + row["discount_rate"]) ** 3

sim = Simulation(
    variables=[
        DistributionSpec("growth_rate",       "normal",     {"mean": 0.40, "std": 0.15}),
        DistributionSpec("churn_rate",        "beta",        {"a": 2, "b": 30}),
        DistributionSpec("revenue_multiple",  "triangular",  {"left": 3, "mode": 8, "right": 20}),
        DistributionSpec("discount_rate",     "normal",     {"mean": 0.12, "std": 0.03}),
    ],
    model=portfolio_value,
    n_iterations=10_000,
    seed=42,
)

result = sim.run()
tornado = sim.sensitivity.tornado()

results = sim.compare_scenarios([
    Scenario("bull", overrides={"revenue_multiple": {"left": 8, "mode": 15, "right": 30}}),
    Scenario("bear", overrides={"revenue_multiple": {"left": 2, "mode": 4, "right": 8}}),
])

sim.plot.histogram(result).savefig("distribution.png")
sim.plot.tornado_chart(tornado).savefig("tornado.png")
sim.plot.scenario_comparison(results).savefig("scenarios.png")
```

## Running Tests

From the repo root:

```bash
pytest tests/test_handler.py -v
```

The test suite covers the full `ResearchHandler` class and every function in `transforms.py`, using synthetic data with no external dependencies.

```bash
pytest tests/test_handler.py::TestSubset -v
pytest tests/test_handler.py::TestTransforms::test_z_score -v
```

## Design Notes

**Source mode locking.** The `set_dependent`, `add_independents`, and `add_controls` methods all accept a `full` parameter. The first call locks the source mode to either `"full"` or `"subset"`. Subsequent calls that use a different mode raise `ValueError` immediately rather than silently mixing columns from different DataFrames. `clear_caches()` resets the lock.

**Guard pattern.** Every method that accesses data checks `is not None` (not bare truthiness, which raises `ValueError` on DataFrames), handles both `full=True` and `full=False` branches explicitly, and bails early with a printed message when the needed dataset isn't available.

**ModelSpec as bridge.** `ResearchHandler` produces a frozen `ModelSpec` dataclass via `get_spec()`. The simulation module consumes it via `Simulation.from_spec()`. The dependency flows one direction: `simulation.py` can accept a `ModelSpec`, but does not import from `ResearchHandler.py`. `ResearchHandler.py` knows nothing about simulations.

**Distribution registry.** `_DISTRIBUTION_REGISTRY` maps string names to draw functions, scipy distributions, and parameter translation maps. Adding a new distribution is a single dictionary insertion — no other code changes needed. Correlated draws use a Gaussian copula (Cholesky decomposition of the correlation matrix applied to standard normal draws, then transformed through each variable's inverse-CDF).

**Sensitivity analysis.** Includes one-at-a-time sweeps, tornado charts, and variance-based Sobol indices via the Saltelli sampling scheme.