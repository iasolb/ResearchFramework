# ResearchHandler

A lightweight pandas-based data handling framework for research workflows. Manages full datasets and working subsets, tracks dependent/independent/control variables, and provides clean interfaces for transforming and attaching computed columns.

## Installation

No special installation required beyond standard dependencies:

```bash
pip install pandas numpy
```

Place `research_handler.py` in your project directory and import:

```python
from research_handler import ResearchHandler
```

## Quick Start

```python
import numpy as np
from research_handler import ResearchHandler

# Define a cleaning function that takes a raw DataFrame and returns a cleaned one
def clean(df):
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna(subset=["income", "age", "education"])
    df["female"] = (df["gender"] == "F").astype(int)
    return df

# Initialize — input file must be CSV
rh = ResearchHandler("survey_data.csv", clean)

# Log-transform income, center age around its mean
rh.normalize_and_attach("income", np.log, "log_income")
rh.normalize_and_attach("age", lambda s: s - s.mean(), "age_centered")

# Create a working subset of employed adults
rh.create_subset(lambda df: (df["age"] >= 18) & (df["employed"] == 1))

# Set up variables from the subset
rh.set_dependent("log_income", full=False)
rh.add_independents("age_centered", "education", full=False)
rh.add_controls("female", full=False)

# Retrieve design matrix and outcome vector
X = rh.get_X()
y = rh.get_y()
```

## API Reference

### `ResearchHandler(filepath, handling_function)`

Constructor. Reads a CSV and passes the raw DataFrame through your cleaning function.

```python
def clean(df):
    df.columns = df.columns.str.lower()
    df["married"] = (df["marital_status"] == "married").astype(int)
    df = df.drop_duplicates()
    return df.dropna()

rh = ResearchHandler("data.csv", clean)
```

The cleaning function receives the raw `pd.DataFrame` from `read_csv` and must return a cleaned `pd.DataFrame`. If reading or cleaning fails, `rh.data` will be `None` and all downstream methods will print a warning and return early.

### `create_subset(condition)`

Creates a working subset of the full dataset based on a boolean condition.

```python
# Simple filter
rh.create_subset(lambda df: df["age"] > 30)

# Multi-condition filter
rh.create_subset(lambda df: (df["income"] > 20000) & (df["employed"] == 1))

# Filter by category membership
rh.create_subset(lambda df: df["country"].isin(["US", "UK", "CA"]))
```

### `reset_subset()`

Clears the working subset back to `None`.

```python
rh.reset_subset()
```

### `set_dependent(col, full=True)`

Sets the dependent (outcome) variable. Pull from the full dataset by default, or from the subset with `full=False`.

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

### `get_X()`

Returns the design matrix as a `pd.DataFrame` by concatenating all independents and controls.

```python
X = rh.get_X()
```

### `get_y()`

Returns the dependent variable as a `pd.Series`.

```python
y = rh.get_y()
```

### `attach(col_name, series, to_full=True, quiet=False)`

Attaches a precomputed Series to the full dataset or subset.

```python
# Attach a squared term
rh.attach("age_sq", rh.data["age"] ** 2)

# Attach to subset instead
rh.attach("age_sq", rh.subset["age"] ** 2, to_full=False)
```

### `normalize_and_attach(source_col, normalizing_function, new_colname, full=True)`

Applies a single-column transformation and attaches the result.

```python
# Log transform
rh.normalize_and_attach("income", np.log, "log_income")

# Z-score standardization
rh.normalize_and_attach("gpa", lambda s: (s - s.mean()) / s.std(), "gpa_z")

# Mean-centering
rh.normalize_and_attach("age", lambda s: s - s.mean(), "age_centered")

# Min-max scaling
rh.normalize_and_attach("score", lambda s: (s - s.min()) / (s.max() - s.min()), "score_scaled")

# Apply to subset
rh.normalize_and_attach("wage", np.log, "log_wage", full=False)
```

### `apply_and_attach(source_cols, func, new_colname, full=True)`

Applies a multi-column transformation and attaches the result. The function receives a DataFrame subset of the specified columns.

```python
# Interaction term
rh.apply_and_attach(
    ["education", "experience"],
    lambda df: df["education"] * df["experience"],
    "edu_x_exp"
)

# Revenue calculation
rh.apply_and_attach(
    ["price", "quantity"],
    lambda df: df["price"] * df["quantity"],
    "revenue"
)

# Row-wise average across test scores
rh.apply_and_attach(
    ["math", "reading", "science"],
    lambda df: df.mean(axis=1),
    "avg_score"
)

# Ratio with safe division
rh.apply_and_attach(
    ["revenue", "visits"],
    lambda df: df["revenue"] / df["visits"].replace(0, np.nan),
    "rev_per_visit",
    full=False
)
```

### `clear_caches()`

Clears the dependent, independents, and controls so you can set up a new specification without reinitializing.

```python
rh.clear_caches()
```

## Example Workflows

### OLS Regression with statsmodels

A standard Mincer wage equation with log wages, centered experience, and a squared term.

```python
import numpy as np
import statsmodels.api as sm
from research_handler import ResearchHandler

def clean(df):
    df.columns = df.columns.str.lower()
    df["female"] = (df["gender"] == "F").astype(int)
    return df.dropna(subset=["wage", "education", "experience", "age", "gender"])

rh = ResearchHandler("labor_data.csv", clean)

# Transform variables
rh.normalize_and_attach("wage", np.log, "log_wage")
rh.normalize_and_attach("experience", lambda s: s - s.mean(), "exp_centered")
rh.attach("exp_centered_sq", rh.data["exp_centered"] ** 2)

# Specification 1: Full sample
rh.set_dependent("log_wage")
rh.add_independents("education", "exp_centered", "exp_centered_sq")
rh.add_controls("female")

X = sm.add_constant(rh.get_X())
y = rh.get_y()

model1 = sm.OLS(y, X).fit()
print(model1.summary())

# Specification 2: Women only
rh.clear_caches()
rh.create_subset(lambda df: df["female"] == 1)

rh.set_dependent("log_wage", full=False)
rh.add_independents("education", "exp_centered", "exp_centered_sq", full=False)

X2 = sm.add_constant(rh.get_X())
y2 = rh.get_y()

model2 = sm.OLS(y2, X2).fit()
print(model2.summary())
```

### Random Forest with scikit-learn

Predicting customer churn with engineered features and standardized inputs.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from research_handler import ResearchHandler

def clean(df):
    df.columns = df.columns.str.lower()
    df = df.dropna()
    df["gender_code"] = df["gender"].map({"M": 0, "F": 1})
    df["region_code"] = df["region"].astype("category").cat.codes
    return df

rh = ResearchHandler("customer_data.csv", clean)

# Feature engineering
rh.apply_and_attach(
    ["revenue", "visits"],
    lambda df: df["revenue"] / df["visits"].replace(0, np.nan),
    "rev_per_visit"
)
rh.normalize_and_attach("tenure", lambda s: (s - s.mean()) / s.std(), "tenure_z")
rh.normalize_and_attach("support_tickets", np.log1p, "log_tickets")

rh.set_dependent("churned")
rh.add_independents("rev_per_visit", "tenure_z", "log_tickets")
rh.add_controls("gender_code", "region_code")

X = rh.get_X()
y = rh.get_y()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print(classification_report(y_test, rf.predict(X_test)))
```

### Heckman Selection Model (Two-Step)

Correct for selection bias in observed wages using the inverse Mills ratio.

```python
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from research_handler import ResearchHandler

def clean(df):
    df.columns = df.columns.str.lower()
    df["married"] = (df["marital_status"] == "married").astype(int)
    return df.dropna(subset=["employed", "age", "education", "wage", "children"])

rh = ResearchHandler("labor_survey.csv", clean)

# Center age for the full sample before splitting
rh.normalize_and_attach("age", lambda s: s - s.mean(), "age_centered")

# Step 1: Selection equation (probit) on full sample
rh.set_dependent("employed")
rh.add_independents("age_centered", "education")
rh.add_controls("married", "children")

X_select = sm.add_constant(rh.get_X())
y_select = rh.get_y()

probit = sm.Probit(y_select, X_select).fit(disp=0)

# Compute and attach inverse Mills ratio
imr = norm.pdf(probit.fittedvalues) / norm.cdf(probit.fittedvalues)
rh.attach("imr", imr)

# Step 2: Outcome equation on workers only, with IMR as control
rh.clear_caches()
rh.create_subset(lambda df: df["employed"] == 1)

rh.normalize_and_attach("wage", np.log, "log_wage", full=False)

rh.set_dependent("log_wage", full=False)
rh.add_independents("age_centered", "education", full=False)
rh.add_controls("imr", full=False)

X_outcome = sm.add_constant(rh.get_X())
y_outcome = rh.get_y()

ols = sm.OLS(y_outcome, X_outcome).fit()
print(ols.summary())
```

### Logistic Regression with Interaction Terms

Modeling treatment effects with centered covariates and interaction terms.

```python
import numpy as np
import statsmodels.api as sm
from research_handler import ResearchHandler

def clean(df):
    df.columns = df.columns.str.lower()
    return df.dropna(subset=["outcome", "treatment", "age", "dosage"])

rh = ResearchHandler("trial_data.csv", clean)

# Center continuous variables to reduce multicollinearity in interactions
rh.normalize_and_attach("age", lambda s: s - s.mean(), "age_c")
rh.normalize_and_attach("dosage", lambda s: s - s.mean(), "dosage_c")

# Interaction: treatment x centered dosage
rh.apply_and_attach(
    ["treatment", "dosage_c"],
    lambda df: df["treatment"] * df["dosage_c"],
    "treat_x_dose"
)

rh.set_dependent("outcome")
rh.add_independents("treatment", "dosage_c", "treat_x_dose")
rh.add_controls("age_c")

X = sm.add_constant(rh.get_X())
y = rh.get_y()

logit = sm.Logit(y, X).fit(disp=0)
print(logit.summary())
```

## Design Notes

Every method that accesses data follows the same guard pattern: check `is not None` (not bare truthiness, which raises `ValueError` on DataFrames), handle both `full=True` and `full=False` branches explicitly, and bail early with a printed message when the needed dataset isn't available.

The `independents` and `controls` caches store references to Series pulled from either the full dataset or the subset. Call `clear_caches()` before setting up a new model specification to avoid mixing columns from different sources.