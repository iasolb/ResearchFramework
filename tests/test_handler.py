"""
Tests for ResearchHandler and transforms.

Run with: pytest tests/test_handler.py -v
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

# Adjust import path — tests run from repo root
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ResearchHandler import ResearchHandler
from transforms import (
    mean_center,
    z_score,
    min_max_scale,
    log_transform,
    log1p_transform,
    square,
    rank_transform,
    winsorize,
    demean_by_group,
    interaction,
    row_mean,
    row_sum,
    safe_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path):
    """Write a small synthetic CSV and return its path."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 65, n),
            "education": np.random.randint(8, 20, n),
            "experience": np.random.randint(0, 40, n),
            "income": np.random.lognormal(10, 1, n),
            "female": np.random.choice([0, 1], n),
            "employed": np.random.choice([0, 1], n, p=[0.2, 0.8]),
            "region": np.random.choice(["east", "west", "south", "north"], n),
        }
    )
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


def identity(df):
    """Cleaning function that returns data unchanged."""
    return df


@pytest.fixture
def rh(sample_csv):
    """Return a ResearchHandler loaded with sample data."""
    return ResearchHandler(sample_csv, identity)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_loads_data(self, rh):
        assert rh.data is not None
        assert len(rh.data) == 100

    def test_columns_present(self, rh):
        expected = {
            "age",
            "education",
            "experience",
            "income",
            "female",
            "employed",
            "region",
        }
        assert expected.issubset(set(rh.data.columns))

    def test_caches_initialized(self, rh):
        assert rh.subset is None
        assert rh.dependent is None
        assert rh.independents == []
        assert rh.controls == []

    def test_bad_filepath(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        assert handler.data is None

    def test_bad_cleaning_function(self, sample_csv):
        def bad_clean(df):
            raise ValueError("intentional error")

        handler = ResearchHandler(sample_csv, bad_clean)
        assert handler.data is None


# ---------------------------------------------------------------------------
# Subset tests
# ---------------------------------------------------------------------------


class TestSubset:
    def test_create_subset(self, rh):
        rh.create_subset(lambda df: df["age"] > 30)
        assert rh.subset is not None
        assert all(rh.subset["age"] > 30)

    def test_subset_is_copy(self, rh):
        rh.create_subset(lambda df: df["age"] > 30)
        rh.subset["new_col"] = 1
        assert "new_col" not in rh.data.columns

    def test_reset_subset(self, rh):
        rh.create_subset(lambda df: df["age"] > 30)
        rh.reset_subset()
        assert rh.subset is None

    def test_create_subset_no_data(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        handler.create_subset(lambda df: df["age"] > 30)  # should not raise
        assert handler.subset is None


# ---------------------------------------------------------------------------
# Variable setting tests
# ---------------------------------------------------------------------------


class TestVariables:
    def test_set_dependent_full(self, rh):
        rh.set_dependent("income")
        assert rh.dependent is not None
        assert rh.dependent.name == "income"

    def test_set_dependent_subset(self, rh):
        rh.create_subset(lambda df: df["employed"] == 1)
        rh.set_dependent("income", full=False)
        assert len(rh.dependent) == len(rh.subset)

    def test_add_independents(self, rh):
        rh.add_independents("age", "education")
        assert len(rh.independents) == 2
        assert rh.independents[0].name == "age"
        assert rh.independents[1].name == "education"

    def test_add_controls(self, rh):
        rh.add_controls("female", "employed")
        assert len(rh.controls) == 2

    def test_get_X(self, rh):
        rh.add_independents("age", "education")
        rh.add_controls("female")
        X = rh.get_X()
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == ["age", "education", "female"]

    def test_get_X_no_independents(self, rh):
        assert rh.get_X() is None

    def test_get_y(self, rh):
        rh.set_dependent("income")
        y = rh.get_y()
        assert isinstance(y, pd.Series)

    def test_get_y_not_set(self, rh):
        assert rh.get_y() is None

    def test_clear_caches(self, rh):
        rh.set_dependent("income")
        rh.add_independents("age")
        rh.add_controls("female")
        rh.clear_caches()
        assert rh.dependent is None
        assert rh.independents == []
        assert rh.controls == []


# ---------------------------------------------------------------------------
# Attach tests
# ---------------------------------------------------------------------------


class TestAttach:
    def test_attach_to_full(self, rh):
        rh.attach("income_sq", rh.data["income"] ** 2)
        assert "income_sq" in rh.data.columns

    def test_attach_to_subset(self, rh):
        rh.create_subset(lambda df: df["age"] > 30)
        rh.attach("flag", pd.Series(1, index=rh.subset.index), to_full=False)
        assert "flag" in rh.subset.columns

    def test_attach_quiet(self, rh, capsys):
        rh.attach("test_col", rh.data["age"], quiet=True)
        captured = capsys.readouterr()
        assert "Attached" not in captured.out

    def test_normalize_and_attach_log(self, rh):
        rh.normalize_and_attach("income", np.log, "log_income")
        assert "log_income" in rh.data.columns
        np.testing.assert_array_almost_equal(
            rh.data["log_income"].values, np.log(rh.data["income"].values)
        )

    def test_normalize_and_attach_zscore(self, rh):
        rh.normalize_and_attach("age", z_score, "age_z")
        assert abs(rh.data["age_z"].mean()) < 1e-10
        assert abs(rh.data["age_z"].std() - 1.0) < 0.05

    def test_apply_and_attach_interaction(self, rh):
        rh.apply_and_attach(
            ["age", "education"], lambda df: df["age"] * df["education"], "age_x_edu"
        )
        assert "age_x_edu" in rh.data.columns
        expected = rh.data["age"] * rh.data["education"]
        pd.testing.assert_series_equal(
            rh.data["age_x_edu"], expected, check_names=False
        )

    def test_apply_and_attach_row_mean(self, rh):
        rh.apply_and_attach(["age", "education", "experience"], row_mean, "avg")
        expected = rh.data[["age", "education", "experience"]].mean(axis=1)
        pd.testing.assert_series_equal(rh.data["avg"], expected, check_names=False)


# ---------------------------------------------------------------------------
# Guard clause tests
# ---------------------------------------------------------------------------


class TestGuards:
    def test_set_dependent_no_data(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        handler.set_dependent("income")  # should print warning, not raise
        assert handler.dependent is None

    def test_set_dependent_subset_when_none(self, rh):
        rh.set_dependent("income", full=False)  # subset is None
        assert rh.dependent is None

    def test_add_independents_no_data(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        handler.add_independents("age")
        assert handler.independents == []

    def test_attach_no_data(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        handler.attach("test", pd.Series([1, 2, 3]))  # should not raise

    def test_normalize_no_data(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        handler.normalize_and_attach("age", np.log, "log_age")  # should not raise

    def test_apply_no_data(self):
        handler = ResearchHandler("nonexistent.csv", identity)
        handler.apply_and_attach(["a", "b"], lambda df: df.sum(axis=1), "c")


# ---------------------------------------------------------------------------
# Transforms module tests
# ---------------------------------------------------------------------------


class TestTransforms:
    @pytest.fixture
    def s(self):
        return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

    def test_mean_center(self, s):
        result = mean_center(s)
        assert abs(result.mean()) < 1e-10

    def test_z_score(self, s):
        result = z_score(s)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 0.05

    def test_min_max_scale(self, s):
        result = min_max_scale(s)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_log_transform(self, s):
        result = log_transform(s)
        np.testing.assert_array_almost_equal(result.values, np.log(s.values))

    def test_log1p_transform(self):
        s = pd.Series([0, 1, 2, 3])
        result = log1p_transform(s)
        np.testing.assert_array_almost_equal(result.values, np.log1p(s.values))

    def test_square(self, s):
        result = square(s)
        np.testing.assert_array_almost_equal(result.values, s.values**2)

    def test_rank_transform(self, s):
        result = rank_transform(s)
        assert list(result) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_winsorize(self):
        s = pd.Series(range(100))
        result = winsorize(0.05, 0.95)(s)
        assert result.min() >= s.quantile(0.05)
        assert result.max() <= s.quantile(0.95)

    def test_winsorize_name(self):
        fn = winsorize(0.01, 0.99)
        assert "winsorize" in fn.__name__

    def test_demean_by_group(self):
        s = pd.Series([10, 20, 30, 40])
        groups = pd.Series(["A", "A", "B", "B"])
        result = demean_by_group(groups)(s)
        # A group: mean=15, so 10-15=-5, 20-15=5
        # B group: mean=35, so 30-35=-5, 40-35=5
        np.testing.assert_array_almost_equal(result.values, [-5, 5, -5, 5])

    def test_interaction(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = interaction(df)
        np.testing.assert_array_equal(result.values, [4, 10, 18])

    def test_row_mean(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = row_mean(df)
        np.testing.assert_array_almost_equal(result.values, [3.0, 4.0])

    def test_row_sum(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = row_sum(df)
        np.testing.assert_array_equal(result.values, [4, 6])

    def test_safe_ratio(self):
        df = pd.DataFrame({"rev": [100, 200, 300], "visits": [10, 0, 30]})
        result = safe_ratio("rev", "visits")(df)
        assert result.iloc[0] == 10.0
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 10.0
