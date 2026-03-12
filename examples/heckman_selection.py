"""
Heckman Selection Model (Two-Step)
===================================
Demonstrates: full workflow with probit -> IMR -> OLS on subset,
              using attach, clear_caches, create_subset, and
              normalize_and_attach across both full and subset data.

Run: python examples/heckman_selection.py
Requires: pip install statsmodels scipy
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ResearchHandler import ResearchHandler
from transforms import mean_center, log_transform


# ---------------------------------------------------------------------------
# Generate synthetic labor survey data
# ---------------------------------------------------------------------------


def generate_data(n=3000, seed=42):
    np.random.seed(seed)

    age = np.random.normal(40, 12, n).clip(18, 70).round()
    education = np.random.normal(13, 3, n).clip(6, 22).round()
    children = np.random.poisson(1.5, n).clip(0, 6)
    married = np.random.choice([0, 1], n, p=[0.4, 0.6])

    # Selection equation: employed = f(age, education, married, children)
    z_employ = -2 + 0.02 * age + 0.15 * education + 0.3 * married - 0.4 * children
    prob_employed = norm.cdf(z_employ + np.random.normal(0, 0.3, n))
    employed = np.random.binomial(1, prob_employed)

    # Wage equation (only observed for employed): log(wage) = f(age, education) + error
    log_wage = 1.0 + 0.06 * education + 0.01 * age + np.random.normal(0, 0.5, n)
    wage = np.exp(log_wage)
    wage = np.where(employed == 1, wage, np.nan)

    marital_status = np.where(married, "married", "single")

    df = pd.DataFrame(
        {
            "age": age,
            "education": education,
            "children": children,
            "marital_status": marital_status,
            "employed": employed,
            "wage": wage,
        }
    )
    path = os.path.join(tempfile.gettempdir(), "labor_survey.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------------------------


def clean(df):
    df.columns = df.columns.str.lower()
    df["married"] = (df["marital_status"] == "married").astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    csv_path = generate_data()
    rh = ResearchHandler(csv_path, clean)

    # Center age on the full sample
    rh.normalize_and_attach("age", mean_center, "age_centered")

    # -----------------------------------------------------------------------
    # Step 1: Selection equation (probit) on full sample
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: PROBIT SELECTION EQUATION (Full Sample)")
    print("=" * 60)

    rh.set_dependent("employed")
    rh.add_independents("age_centered", "education")
    rh.add_controls("married", "children")

    X_select = sm.add_constant(rh.get_X())
    y_select = rh.get_y()

    probit = sm.Probit(y_select, X_select).fit(disp=0)
    print(probit.summary())

    # Compute and attach inverse Mills ratio
    imr = norm.pdf(probit.fittedvalues) / norm.cdf(probit.fittedvalues)
    rh.attach("imr", imr)

    # -----------------------------------------------------------------------
    # Step 2: Outcome equation on employed workers, with IMR correction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: OLS WAGE EQUATION (Employed Only, with IMR)")
    print("=" * 60)

    rh.clear_caches()
    rh.create_subset(lambda df: df["employed"] == 1)

    # Log wages on the subset
    rh.normalize_and_attach("wage", log_transform, "log_wage", full=False)

    rh.set_dependent("log_wage", full=False)
    rh.add_independents("age_centered", "education", full=False)
    rh.add_controls("imr", full=False)

    X_outcome = sm.add_constant(rh.get_X())
    y_outcome = rh.get_y()

    ols = sm.OLS(y_outcome, X_outcome).fit()
    print(ols.summary())

    # Check significance of IMR (indicates selection bias)
    imr_pval = ols.pvalues.get("imr", None)
    if imr_pval is not None:
        if imr_pval < 0.05:
            print(f"\nIMR is significant (p={imr_pval:.4f}) — selection bias detected.")
        else:
            print(
                f"\nIMR is not significant (p={imr_pval:.4f}) — no evidence of selection bias."
            )


if __name__ == "__main__":
    main()
