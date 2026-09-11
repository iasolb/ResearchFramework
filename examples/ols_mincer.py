"""
OLS Mincer Wage Equation
========================
Demonstrates: log transform, mean-centering, squared terms,
              running multiple specifications, subsetting.

Run: python examples/ols_mincer.py
Requires: pip install statsmodels
"""

import os
import tempfile
import numpy as np
import pandas as pd
import statsmodels.api as sm

from otter import Pond, log_transform, mean_center, square


# ---------------------------------------------------------------------------
# Generate synthetic labor data
# ---------------------------------------------------------------------------


def generate_data(n=2000, seed=42):
    np.random.seed(seed)
    female = np.random.choice([0, 1], n)
    education = np.random.normal(13, 2, n).clip(8, 22).round()
    experience = np.random.normal(15, 8, n).clip(0, 45).round()
    age = education + 6 + experience + np.random.normal(0, 2, n)
    age = age.clip(18, 70).round()

    # DGP: log(wage) = 1.5 + 0.08*edu + 0.03*exp - 0.0005*exp^2 - 0.15*female + noise
    log_wage = (
        1.5
        + 0.08 * education
        + 0.03 * experience
        - 0.0005 * experience**2
        - 0.15 * female
        + np.random.normal(0, 0.4, n)
    )
    wage = np.exp(log_wage)

    df = pd.DataFrame(
        {
            "wage": wage,
            "education": education,
            "experience": experience,
            "age": age,
            "gender": np.where(female, "F", "M"),
        }
    )
    path = os.path.join(tempfile.gettempdir(), "labor_data.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------------------------


def clean(df):
    df.columns = df.columns.str.lower()
    df["female"] = (df["gender"] == "F").astype(int)
    return df.dropna()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    csv_path = generate_data()
    pond = Pond(csv_path, clean)

    # Transform variables
    pond.normalize_and_attach("wage", log_transform, "log_wage")
    pond.normalize_and_attach("experience", mean_center, "exp_centered")
    # `if pond.data:` raises: a DataFrame has no truth value. A failed load
    # gives an EMPTY frame rather than None, so emptiness is the real check.
    if not pond.data.empty:
        pond.attach("exp_centered_sq", square(pond.data["exp_centered"]))

    # Specification 1: Full sample
    print("\n" + "=" * 60)
    print("SPECIFICATION 1: Full Sample Mincer Equation")
    print("=" * 60)

    pond.set_dependent("log_wage")
    pond.add_independents("education", "exp_centered", "exp_centered_sq")
    pond.add_controls("female")

    X = sm.add_constant(pond.get_X())
    y = pond.get_y()

    model1 = sm.OLS(y, X).fit()
    print(model1.summary())

    # Specification 2: Women only
    print("\n" + "=" * 60)
    print("SPECIFICATION 2: Women Only")
    print("=" * 60)

    pond.clear_caches()
    pond.create_pool(lambda df: df["female"] == 1)

    pond.set_dependent("log_wage", full=False)
    pond.add_independents("education", "exp_centered", "exp_centered_sq", full=False)

    X2 = sm.add_constant(pond.get_X())
    y2 = pond.get_y()

    model2 = sm.OLS(y2, X2).fit()
    print(model2.summary())


if __name__ == "__main__":
    main()
