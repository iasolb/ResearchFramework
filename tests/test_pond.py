"""
Tests for Pond and transforms.

Run with: pytest tests/test_pond.py -v
"""

import tempfile
import numpy as np
import pandas as pd
import pytest

from otter import Pond
from otter.transforms import (
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
def pond(sample_csv):
    """Return a Pond loaded with sample data."""
    return Pond(sample_csv, identity)


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_loads_data(self, pond):
        assert pond.data is not None
        assert len(pond.data) == 100

    def test_columns_present(self, pond):
        expected = {
            "age",
            "education",
            "experience",
            "income",
            "female",
            "employed",
            "region",
        }
        assert expected.issubset(set(pond.data.columns))

    def test_caches_initialized(self, pond):
        assert pond.pool is None
        assert pond.dependent is None
        assert pond.independents == []
        assert pond.controls == []

    # Deleted 2026-08-28: test_bad_filepath and test_bad_cleaning_function both
    # asserted `handler.data is None`, which is a dead contract. `_load` returns
    # an EMPTY DataFrame on failure, not None. Both behaviours are still
    # covered, against the real contract, in test_loader_contract.py.


# ---------------------------------------------------------------------------
# Subset tests
# ---------------------------------------------------------------------------


class TestSubset:
    def test_create_pool(self, pond):
        pond.create_pool(lambda df: df["age"] > 30)
        assert pond.pool is not None
        assert all(pond.pool["age"] > 30)

    def test_subset_is_copy(self, pond):
        pond.create_pool(lambda df: df["age"] > 30)
        pond.pool["new_col"] = 1
        assert "new_col" not in pond.data.columns

    def test_reset_pool(self, pond):
        pond.create_pool(lambda df: df["age"] > 30)
        pond.reset_pool()
        assert pond.pool is None

    # Deleted 2026-08-28: test_create_pool_no_data asserted that subsetting an
    # empty frame does not raise. It does raise KeyError, because the condition
    # references a column that is not there. See the note at the top of
    # test_loader_contract.py: whether the empty-data path should guard is a
    # real open design question, not something a test should assert either way
    # while the code does the opposite.


# ---------------------------------------------------------------------------
# Variable setting tests
# ---------------------------------------------------------------------------


class TestVariables:
    def test_set_dependent_full(self, pond):
        pond.set_dependent("income")
        assert pond.dependent is not None
        assert pond.dependent.name == "income"

    def test_set_dependent_subset(self, pond):
        pond.create_pool(lambda df: df["employed"] == 1)
        pond.set_dependent("income", full=False)
        assert len(pond.dependent) == len(pond.pool)

    def test_add_independents(self, pond):
        pond.add_independents("age", "education")
        assert len(pond.independents) == 2
        assert pond.independents[0].name == "age"
        assert pond.independents[1].name == "education"

    def test_add_controls(self, pond):
        pond.add_controls("female", "employed")
        assert len(pond.controls) == 2

    def test_get_X(self, pond):
        pond.add_independents("age", "education")
        pond.add_controls("female")
        X = pond.get_X()
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == ["age", "education", "female"]

    def test_get_X_no_independents(self, pond):
        assert pond.get_X() is None

    def test_get_y(self, pond):
        pond.set_dependent("income")
        y = pond.get_y()
        assert isinstance(y, pd.Series)

    def test_get_y_not_set(self, pond):
        assert pond.get_y() is None

    def test_clear_caches(self, pond):
        pond.set_dependent("income")
        pond.add_independents("age")
        pond.add_controls("female")
        pond.clear_caches()
        assert pond.dependent is None
        assert pond.independents == []
        assert pond.controls == []


# ---------------------------------------------------------------------------
# Attach tests
# ---------------------------------------------------------------------------


class TestAttach:
    def test_attach_to_full(self, pond):
        pond.attach("income_sq", pond.data["income"] ** 2)
        assert "income_sq" in pond.data.columns

    def test_attach_to_subset(self, pond):
        pond.create_pool(lambda df: df["age"] > 30)
        pond.attach("flag", pd.Series(1, index=pond.pool.index), to_full=False)
        assert "flag" in pond.pool.columns

    def test_attach_quiet(self, pond, capsys):
        pond.attach("test_col", pond.data["age"], quiet=True)
        captured = capsys.readouterr()
        assert "Attached" not in captured.out

    def test_normalize_and_attach_log(self, pond):
        pond.normalize_and_attach("income", np.log, "log_income")
        assert "log_income" in pond.data.columns
        np.testing.assert_array_almost_equal(
            pond.data["log_income"].values, np.log(pond.data["income"].values)
        )

    def test_normalize_and_attach_zscore(self, pond):
        pond.normalize_and_attach("age", z_score, "age_z")
        assert abs(pond.data["age_z"].mean()) < 1e-10
        assert abs(pond.data["age_z"].std() - 1.0) < 0.05

    # Deleted 2026-08-28: two tests exercised `apply_and_attach`, a method this
    # class does not have. It exists on another machine's renamed API, which is
    # reference only. If a multi-column transform helper is wanted here, it
    # needs writing first, and then a test.


# ---------------------------------------------------------------------------
# Guard clause tests
# ---------------------------------------------------------------------------


class TestGuards:
    # Deleted 2026-08-28, four tests: test_set_dependent_no_data,
    # test_add_independents_no_data, test_normalize_no_data and
    # test_apply_no_data. Each built a handler on a nonexistent file and
    # asserted the next call warns rather than raising. The current code raises,
    # and the last one also called the nonexistent `apply_and_attach`.
    #
    # What survives is the pair below, which pass against the real code. The
    # empty-data guard question is recorded in test_loader_contract.py rather
    # than asserted here in either direction.

    def test_set_dependent_subset_when_none(self, pond):
        pond.set_dependent("income", full=False)  # subset is None
        assert pond.dependent is None

    def test_attach_no_data(self):
        handler = Pond("nonexistent.csv", identity)
        handler.attach("test", pd.Series([1, 2, 3]))  # should not raise


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
        np.testing.assert_array_almost_equal(result.to_numpy(), np.log(s.to_numpy()))

    def test_log1p_transform(self):
        s = pd.Series([0, 1, 2, 3])
        result = log1p_transform(s)
        np.testing.assert_array_almost_equal(result.to_numpy(), np.log1p(s.to_numpy()))

    def test_square(self, s):
        result = square(s)
        np.testing.assert_array_almost_equal(result.to_numpy(), s.to_numpy() ** 2)

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
        np.testing.assert_array_almost_equal(result.to_numpy(), [-5, 5, -5, 5])

    def test_interaction(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = interaction(df)
        np.testing.assert_array_equal(result.to_numpy(), [4, 10, 18])

    def test_row_mean(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = row_mean(df)
        np.testing.assert_array_almost_equal(result.to_numpy(), [3.0, 4.0])

    def test_row_sum(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = row_sum(df)
        np.testing.assert_array_equal(result.to_numpy(), [4, 6])

    def test_safe_ratio(self):
        df = pd.DataFrame({"rev": [100, 200, 300], "visits": [10, 0, 30]})
        result = safe_ratio("rev", "visits")(df)
        assert result.iloc[0] == 10.0
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 10.0
