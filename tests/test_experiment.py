"""What the experiment design and measurement module promises.

The assertions here are mostly PROPERTIES rather than fixed numbers: that a
bigger required lift needs a smaller sample, that CUPED lowers variance
without moving the mean, that a degenerate covariate is handled instead of
dividing by zero. A test that pins an arbitrary float tells you the code did
not change; a property tells you it is still correct.

Where a number IS pinned it is one an external tool agrees on: the even-split
continuous sample size is checked against statsmodels directly, so the test
would catch a wrong effect-size formula rather than merely a changed one.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import TTestIndPower

from otter.experiment import (
    ExperimentSize,
    assign_groups,
    compare_groups,
    cuped_adjust,
    detectable_lift_for_mean,
    detectable_lift_for_proportion,
    pretest_balance,
    sample_size_for_mean,
    sample_size_for_proportion,
)


# ── sizing a continuous metric ────────────────────────────────────────────


def test_an_even_split_sizes_both_arms_the_same():
    size = sample_size_for_mean(100.0, 50.0, 5.0)
    assert size.treatment_n == size.control_n
    assert size.total_n == size.treatment_n + size.control_n


def test_the_sample_size_agrees_with_statsmodels():
    """Pinned against the library that owns the calculation, not a snapshot."""
    size = sample_size_for_mean(100.0, 50.0, 10.0, power=0.8, alpha=0.05)
    expected = TTestIndPower().solve_power(
        effect_size=100.0 * 0.10 / 50.0, power=0.8, nobs1=None, ratio=1.0, alpha=0.05
    )
    assert size.treatment_n == int(np.ceil(expected))


def test_a_bigger_required_lift_needs_fewer_units():
    small = sample_size_for_mean(100.0, 50.0, 1.0)
    large = sample_size_for_mean(100.0, 50.0, 10.0)
    assert large.total_n < small.total_n


def test_more_power_needs_more_units():
    lower = sample_size_for_mean(100.0, 50.0, 5.0, power=0.80)
    higher = sample_size_for_mean(100.0, 50.0, 5.0, power=0.95)
    assert higher.total_n > lower.total_n


def test_an_uneven_split_puts_the_named_fraction_in_treatment():
    size = sample_size_for_mean(100.0, 50.0, 5.0, treatment_share=0.2)
    # a fifth in treatment means control is about four times as large
    assert size.control_n > size.treatment_n
    assert size.control_n / size.treatment_n == pytest.approx(4.0, rel=0.01)


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5, 50])
def test_a_share_outside_the_unit_interval_is_refused(share):
    """50 is the commonest mistake: a percent where a fraction belongs."""
    with pytest.raises(ValueError, match="fraction"):
        sample_size_for_mean(100.0, 50.0, 5.0, treatment_share=share)


def test_a_zero_baseline_mean_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="percentage lift has no meaning"):
        sample_size_for_mean(0.0, 50.0, 5.0)


def test_a_non_positive_sd_is_refused():
    with pytest.raises(ValueError, match="baseline_sd"):
        sample_size_for_mean(100.0, 0.0, 5.0)


# ── sizing a proportion ───────────────────────────────────────────────────


def test_a_proportion_sample_size_is_positive_and_even_when_split_evenly():
    size = sample_size_for_proportion(0.10, 10.0)
    assert size.treatment_n > 0
    assert size.treatment_n == pytest.approx(size.control_n, rel=0.01)


def test_a_rarer_event_needs_a_bigger_sample():
    common = sample_size_for_proportion(0.30, 10.0)
    rare = sample_size_for_proportion(0.01, 10.0)
    assert rare.total_n > common.total_n


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.2, 1.2])
def test_a_rate_outside_zero_to_one_is_refused(rate):
    with pytest.raises(ValueError, match="proportion"):
        sample_size_for_proportion(rate, 10.0)


def test_a_lift_that_pushes_the_rate_past_one_is_refused():
    """0.9 plus 50% is 1.35, which is not a rate. Say so, do not compute it."""
    with pytest.raises(ValueError, match="not a proportion"):
        sample_size_for_proportion(0.9, 50.0)


# ── the sensitivity direction ─────────────────────────────────────────────


def test_a_bigger_sample_detects_a_smaller_lift():
    small = detectable_lift_for_mean(100.0, 50.0, total_n=1_000)
    large = detectable_lift_for_mean(100.0, 50.0, total_n=100_000)
    assert large < small


def test_sizing_and_sensitivity_are_inverses():
    """Size for a lift, then ask what that sample detects: the same lift."""
    size = sample_size_for_mean(100.0, 50.0, 5.0)
    back = detectable_lift_for_mean(100.0, 50.0, total_n=size.total_n)
    assert back == pytest.approx(5.0, rel=0.02)


def test_a_bigger_sample_detects_a_smaller_lift_on_a_proportion():
    small = detectable_lift_for_proportion(0.10, total_n=5_000)
    large = detectable_lift_for_proportion(0.10, total_n=500_000)
    assert large < small


def test_a_tiny_total_is_refused():
    with pytest.raises(ValueError, match="total_n"):
        detectable_lift_for_mean(100.0, 50.0, total_n=2)


# ── assignment ────────────────────────────────────────────────────────────


def test_every_unit_is_assigned_exactly_once():
    frame = assign_groups(range(500), {"control": 0.5, "treatment": 0.5}, seed=0)
    assert len(frame) == 500
    assert frame["unit"].nunique() == 500


def test_assignment_is_reproducible_from_a_seed():
    a = assign_groups(range(200), {"c": 0.5, "t": 0.5}, seed=7)
    b = assign_groups(range(200), {"c": 0.5, "t": 0.5}, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_gives_a_different_assignment():
    a = assign_groups(range(200), {"c": 0.5, "t": 0.5}, seed=1)
    b = assign_groups(range(200), {"c": 0.5, "t": 0.5}, seed=2)
    assert not a["group"].equals(b["group"])


def test_the_split_lands_near_the_requested_ratio():
    frame = assign_groups(range(20_000), {"control": 0.9, "treatment": 0.1}, seed=3)
    share = (frame["group"] == "treatment").mean()
    assert share == pytest.approx(0.1, abs=0.01)


def test_duplicate_units_are_refused():
    """The same unit in two arms is silent corruption, so it is a hard error."""
    with pytest.raises(ValueError, match="duplicate"):
        assign_groups([1, 2, 2, 3], {"c": 0.5, "t": 0.5})


def test_ratios_that_do_not_sum_to_one_are_refused():
    with pytest.raises(ValueError, match="sum to 1"):
        assign_groups(range(10), {"c": 50, "t": 50})


def test_no_units_is_refused():
    with pytest.raises(ValueError, match="no units"):
        assign_groups([], {"c": 1.0})


# ── CUPED ─────────────────────────────────────────────────────────────────


@pytest.fixture
def correlated():
    rng = np.random.default_rng(11)
    pre = rng.normal(100.0, 20.0, size=4_000)
    post = pre * 0.8 + rng.normal(20.0, 5.0, size=4_000)
    return pd.DataFrame({"pre": pre, "post": post})


def test_cuped_lowers_variance(correlated):
    adjusted = cuped_adjust(correlated, "post", "pre")
    assert adjusted.var() < correlated["post"].var()


def test_cuped_does_not_move_the_mean(correlated):
    """The point is variance reduction. Shifting the estimate would be a bug."""
    adjusted = cuped_adjust(correlated, "post", "pre")
    assert adjusted.mean() == pytest.approx(correlated["post"].mean(), rel=1e-9)


def test_a_constant_covariate_returns_the_metric_unchanged():
    """No variance means no information, and must not be a divide by zero."""
    frame = pd.DataFrame({"pre": [5.0] * 50, "post": np.arange(50.0)})
    pd.testing.assert_series_equal(
        cuped_adjust(frame, "post", "pre"), frame["post"], check_names=False
    )


def test_a_missing_column_is_named(correlated):
    with pytest.raises(KeyError, match="nope"):
        cuped_adjust(correlated, "post", "nope")


# ── comparing arms ────────────────────────────────────────────────────────


@pytest.fixture
def experiment():
    rng = np.random.default_rng(5)
    n = 2_000
    group = np.where(np.arange(n) % 2 == 0, "control", "treatment")
    pre = rng.normal(100.0, 25.0, size=n)
    revenue = pre * 0.5 + rng.normal(50.0, 10.0, size=n)
    revenue = revenue + (group == "treatment") * 8.0   # a real lift
    flat = rng.normal(10.0, 2.0, size=n)               # no lift
    return pd.DataFrame(
        {"group": group, "pre_revenue": pre, "revenue": revenue, "flat": flat}
    )


def test_a_real_lift_is_detected(experiment):
    result = compare_groups(
        experiment, "group", ["revenue"], control="control", treatment="treatment"
    )
    row = result.iloc[0]
    assert row["p_value"] < 0.01
    assert row["mean_difference"] > 0


def test_a_metric_with_no_lift_is_not_flagged(experiment):
    result = compare_groups(
        experiment, "group", ["flat"], control="control", treatment="treatment"
    )
    assert result.iloc[0]["p_value"] > 0.05


def test_the_confidence_interval_brackets_the_difference(experiment):
    row = compare_groups(
        experiment, "group", ["revenue"], control="control", treatment="treatment"
    ).iloc[0]
    assert row["ci_low"] < row["mean_difference"] < row["ci_high"]


def test_one_row_per_metric_and_a_correction_column(experiment):
    result = compare_groups(
        experiment,
        "group",
        ["revenue", "flat"],
        control="control",
        treatment="treatment",
    )
    assert list(result["metric"]) == ["revenue", "flat"]
    assert "p_value_corrected" in result.columns
    assert (result["p_value_corrected"] >= result["p_value"]).all()


def test_the_correction_can_be_switched_off(experiment):
    result = compare_groups(
        experiment,
        "group",
        ["revenue", "flat"],
        control="control",
        treatment="treatment",
        correction=None,
    )
    assert "p_value_corrected" not in result.columns


def test_cuped_tightens_the_interval_without_moving_the_estimate(experiment):
    """The whole value of CUPED, asserted rather than asserted about."""
    plain = compare_groups(
        experiment, "group", ["revenue"], control="control", treatment="treatment"
    ).iloc[0]
    reduced = compare_groups(
        experiment,
        "group",
        ["revenue"],
        control="control",
        treatment="treatment",
        cuped_pre={"revenue": "pre_revenue"},
    ).iloc[0]
    plain_width = plain["ci_high"] - plain["ci_low"]
    reduced_width = reduced["ci_high"] - reduced["ci_low"]
    assert reduced_width < plain_width
    assert reduced["mean_difference"] == pytest.approx(
        plain["mean_difference"], rel=0.15
    )
    assert reduced["cuped"] is True or reduced["cuped"] == True  # noqa: E712


def test_an_absent_group_is_named(experiment):
    with pytest.raises(ValueError, match="holdout"):
        compare_groups(
            experiment, "group", ["revenue"], control="holdout", treatment="treatment"
        )


def test_an_absent_metric_is_named(experiment):
    with pytest.raises(KeyError, match="missing_metric"):
        compare_groups(
            experiment,
            "group",
            ["missing_metric"],
            control="control",
            treatment="treatment",
        )


def test_an_arm_with_one_observation_is_refused():
    """A single observation has no variance, so a t-test on it is nonsense."""
    frame = pd.DataFrame(
        {"group": ["control", "control", "treatment"], "y": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError, match="at least 2 observations"):
        compare_groups(frame, "group", ["y"], control="control", treatment="treatment")


# ── pre-test balance ──────────────────────────────────────────────────────


def test_a_balanced_split_shows_no_pretest_difference(experiment):
    result = pretest_balance(
        experiment, "group", ["pre_revenue"], control="control", treatment="treatment"
    )
    assert bool(result.iloc[0]["uninformative"]) is False
    assert result.iloc[0]["p_value"] > 0.05


def test_an_empty_pre_period_is_reported_not_tested():
    """New units have no history. That is a legitimate state, not a result."""
    frame = pd.DataFrame(
        {
            "group": ["control"] * 10 + ["treatment"] * 10,
            "pre_spend": [0.0] * 20,
        }
    )
    result = pretest_balance(
        frame, "group", ["pre_spend"], control="control", treatment="treatment"
    )
    assert bool(result.iloc[0]["uninformative"]) is True
    assert "p_value" not in result.columns or pd.isna(result.iloc[0].get("p_value"))


def test_a_deliberately_unbalanced_split_is_caught():
    """Break randomisation on purpose and the check must go red."""
    rng = np.random.default_rng(2)
    pre = np.concatenate([rng.normal(100.0, 10.0, 500), rng.normal(130.0, 10.0, 500)])
    frame = pd.DataFrame(
        {"group": ["control"] * 500 + ["treatment"] * 500, "pre_spend": pre}
    )
    result = pretest_balance(
        frame, "group", ["pre_spend"], control="control", treatment="treatment"
    )
    assert result.iloc[0]["p_value"] < 0.001


def test_the_result_keeps_the_requested_column_order():
    frame = pd.DataFrame(
        {
            "group": ["control"] * 20 + ["treatment"] * 20,
            "a": np.random.default_rng(1).normal(size=40),
            "b": [0.0] * 40,
            "c": np.random.default_rng(4).normal(size=40),
        }
    )
    result = pretest_balance(
        frame, "group", ["a", "b", "c"], control="control", treatment="treatment"
    )
    assert list(result["metric"]) == ["a", "b", "c"]


def test_experiment_size_exposes_a_total():
    size = ExperimentSize(treatment_n=10, control_n=30, effect_size=0.2,
                          detectable_lift_pct=5.0)
    assert size.total_n == 40
