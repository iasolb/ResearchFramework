"""
Common transforms for use with ResearchHandler.

Single-column transforms (for use with normalize_and_attach):
    rh.normalize_and_attach("age", mean_center, "age_c")
    rh.normalize_and_attach("income", log_transform, "log_income")

Multi-column transforms (for use with apply_and_attach):
    rh.apply_and_attach(["education", "experience"], interaction, "edu_x_exp")
    rh.apply_and_attach(["math", "reading", "science"], row_mean, "avg_score")
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-column transforms (Series -> Series)
# ---------------------------------------------------------------------------


def mean_center(s: pd.Series) -> pd.Series:
    """Subtract the mean: x - mean(x)"""
    return s - s.mean()


def z_score(s: pd.Series) -> pd.Series:
    """Standardize to mean=0, std=1: (x - mean) / std"""
    return (s - s.mean()) / s.std()


def min_max_scale(s: pd.Series) -> pd.Series:
    """Scale to [0, 1]: (x - min) / (max - min)"""
    return (s - s.min()) / (s.max() - s.min())


def log_transform(s: pd.Series) -> pd.Series:
    """Natural log. Use for strictly positive values."""
    return pd.Series(np.log(s))


def log1p_transform(s: pd.Series) -> pd.Series:
    """log(1 + x). Safe for zeros (e.g. count data)."""
    return pd.Series(np.log1p(s))


def square(s: pd.Series) -> pd.Series:
    """x^2. Useful for quadratic terms."""
    return s**2


def rank_transform(s: pd.Series) -> pd.Series:
    """Replace values with their rank (average method for ties)."""
    return s.rank(method="average")


def winsorize(lower: float = 0.01, upper: float = 0.99):
    """
    Returns a transform that clips values at the given quantiles.

    Usage:
        rh.normalize_and_attach("income", winsorize(0.01, 0.99), "income_wins")
    """

    def _winsorize(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lo, hi)

    _winsorize.__name__ = f"winsorize({lower},{upper})"
    return _winsorize


def demean_by_group(group_col: pd.Series):
    """
    Returns a transform that subtracts group-level means (fixed-effects style).

    Usage:
        rh.normalize_and_attach("income", demean_by_group(rh.data["industry"]), "income_demeaned")
    """

    def _demean(s: pd.Series) -> pd.Series:
        return s - s.groupby(group_col).transform("mean")

    _demean.__name__ = f"demean_by_{group_col.name}"
    return _demean


# ---------------------------------------------------------------------------
# Multi-column transforms (DataFrame -> Series)
# ---------------------------------------------------------------------------


def interaction(df: pd.DataFrame) -> pd.Series:
    """
    Product of the first two columns. For use with apply_and_attach.

    Usage:
        rh.apply_and_attach(["education", "experience"], interaction, "edu_x_exp")
    """
    cols = df.columns
    return df[cols[0]] * df[cols[1]]


def row_mean(df: pd.DataFrame) -> pd.Series:
    """
    Row-wise mean across all provided columns.

    Usage:
        rh.apply_and_attach(["math", "reading", "science"], row_mean, "avg_score")
    """
    return df.mean(axis=1)


def row_sum(df: pd.DataFrame) -> pd.Series:
    """
    Row-wise sum across all provided columns.

    Usage:
        rh.apply_and_attach(["q1", "q2", "q3", "q4"], row_sum, "total")
    """
    return df.sum(axis=1)


def safe_ratio(numerator: str, denominator: str):
    """
    Returns a transform that computes a ratio, replacing division by zero with NaN.

    Usage:
        rh.apply_and_attach(["revenue", "visits"], safe_ratio("revenue", "visits"), "rev_per_visit")
    """

    def _ratio(df: pd.DataFrame) -> pd.Series:
        return df[numerator] / df[denominator].replace(0, np.nan)

    _ratio.__name__ = f"ratio({numerator}/{denominator})"
    return _ratio
