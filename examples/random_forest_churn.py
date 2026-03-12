"""
Random Forest Churn Prediction
===============================
Demonstrates: feature engineering with apply_and_attach,
              z-scoring, log1p transform, safe ratio.

Run: python examples/random_forest_churn.py
Requires: pip install scikit-learn
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ResearchHandler import ResearchHandler
from transforms import z_score, log1p_transform, safe_ratio


# ---------------------------------------------------------------------------
# Generate synthetic customer data
# ---------------------------------------------------------------------------


def generate_data(n=3000, seed=42):
    np.random.seed(seed)

    gender = np.random.choice(["M", "F"], n)
    region = np.random.choice(["east", "west", "south", "north"], n)
    tenure = np.random.exponential(24, n).clip(1, 120).round()
    visits = np.random.poisson(10, n)
    revenue = visits * np.random.uniform(5, 50, n) + np.random.normal(0, 20, n)
    revenue = revenue.clip(0)
    support_tickets = np.random.poisson(2, n)

    # DGP: churn more likely with low tenure, low revenue/visit, many tickets
    rev_per_visit = np.where(visits > 0, revenue / visits, 0)
    logit = -1.5 - 0.03 * tenure - 0.02 * rev_per_visit + 0.3 * support_tickets
    prob = 1 / (1 + np.exp(-logit))
    churned = np.random.binomial(1, prob)

    df = pd.DataFrame(
        {
            "gender": gender,
            "region": region,
            "tenure": tenure,
            "visits": visits,
            "revenue": revenue,
            "support_tickets": support_tickets,
            "churned": churned,
        }
    )
    path = os.path.join(tempfile.gettempdir(), "customer_data.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------------------------


def clean(df):
    df.columns = df.columns.str.lower()
    df = df.dropna()
    df["gender_code"] = df["gender"].map({"M": 0, "F": 1})
    df["region_code"] = df["region"].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    csv_path = generate_data()
    rh = ResearchHandler(csv_path, clean)

    # Feature engineering
    rh.apply_and_attach(
        ["revenue", "visits"], safe_ratio("revenue", "visits"), "rev_per_visit"
    )
    rh.normalize_and_attach("tenure", z_score, "tenure_z")
    rh.normalize_and_attach("support_tickets", log1p_transform, "log_tickets")

    # Set up model variables
    rh.set_dependent("churned")
    rh.add_independents("rev_per_visit", "tenure_z", "log_tickets")
    rh.add_controls("gender_code", "region_code")

    X = rh.get_X().fillna(0)  # handle NaN from safe_ratio division
    y = rh.get_y()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit and evaluate
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("RANDOM FOREST CHURN PREDICTION")
    print("=" * 60)
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\n{classification_report(y_test, rf.predict(X_test))}")

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("Feature Importances:")
    for name, imp in importances.items():
        print(f"  {name:20s} {imp:.4f}")


if __name__ == "__main__":
    main()
