"""Designing and measuring controlled experiments.

The rest of otter describes a study you already have data for. This module
covers the two questions that come before and after that: how big does the
experiment need to be, and what did it actually show.

    design      sample_size_for_mean, sample_size_for_proportion
                detectable_lift_for_mean, detectable_lift_for_proportion
    assign      assign_groups
    measure     compare_groups, cuped_adjust, pretest_balance

Everything takes and returns plain pandas objects, so it composes with
``Pond`` without needing to know about it.

A worked example, from planning to result::

    from otter.experiment import sample_size_for_mean, assign_groups, compare_groups

    # How many units do we need to see a 2% lift on a metric averaging 120
    # with a standard deviation of 300, splitting the sample evenly?
    size = sample_size_for_mean(
        baseline_mean=120.0, baseline_sd=300.0, min_lift_pct=2.0,
    )
    size.treatment_n, size.control_n

    # Randomise, then measure once the data is in.
    assignment = assign_groups(user_ids, {"control": 0.5, "treatment": 0.5}, seed=1)
    compare_groups(
        data, group_col="group", metric_cols=["revenue"],
        control="control", treatment="treatment",
    )

ON THE STATISTICS. Continuous comparisons use Welch's t-test, which does not
assume equal variances between arms, with Satterthwaite degrees of freedom.
Proportion sample sizes use the normal approximation, and the sensitivity
direction uses the arcsine variance-stabilising transform. Variance reduction
is CUPED (Deng, Xu, Kohavi and Walker, 2013), which regresses out a
pre-period covariate. When several metrics are compared at once the p-values
are corrected for multiplicity, Benjamini-Hochberg by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

Alternative = Literal["two-sided", "larger", "smaller"]

__all__ = [
    "ExperimentSize",
    "assign_groups",
    "compare_groups",
    "cuped_adjust",
    "detectable_lift_for_mean",
    "detectable_lift_for_proportion",
    "pretest_balance",
    "sample_size_for_mean",
    "sample_size_for_proportion",
]


@dataclass(frozen=True)
class ExperimentSize:
    """How many units each arm needs, and the effect that buys.

    ``effect_size`` is Cohen's d for a mean and the arcsine-transformed
    effect for a proportion, so it is comparable within a metric type and not
    across the two.
    """

    treatment_n: int
    control_n: int
    effect_size: float
    detectable_lift_pct: float

    @property
    def total_n(self) -> int:
        """Units needed across both arms."""
        return self.treatment_n + self.control_n


def _check_share(treatment_share: float) -> float:
    """The control-to-treatment ratio, or a named error.

    ``treatment_share`` is the FRACTION of units in treatment, so 0.5 is an
    even split and 0.1 puts a tenth in treatment. The source this replaced
    called it a holdout percent and then solved for the arm it named
    treatment, which read backwards; naming the fraction explicitly removes
    the ambiguity rather than documenting it.
    """
    if not 0.0 < treatment_share < 1.0:
        raise ValueError(
            f"treatment_share must be strictly between 0 and 1, got "
            f"{treatment_share!r}. It is a fraction, so an even split is 0.5, "
            f"not 50."
        )
    return (1.0 - treatment_share) / treatment_share


def _z(alpha: float, power: float, alternative: Alternative) -> tuple[float, float]:
    """The critical values for a normal-approximation sample size."""
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    elif alternative in ("larger", "smaller"):
        z_alpha = stats.norm.ppf(1.0 - alpha)
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger' or 'smaller', got "
            f"{alternative!r}"
        )
    return float(z_alpha), float(stats.norm.ppf(power))


def sample_size_for_mean(
    baseline_mean: float,
    baseline_sd: float,
    min_lift_pct: float,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ExperimentSize:
    """Units per arm needed to detect ``min_lift_pct`` on a continuous metric.

    ``min_lift_pct`` is a percentage of the baseline mean, so 2.0 means a 2%
    relative lift, which is how experiment targets are usually stated.

    >>> size = sample_size_for_mean(100.0, 50.0, 5.0)
    >>> size.treatment_n == size.control_n  # an even split by default
    True
    """
    if baseline_sd <= 0:
        raise ValueError(f"baseline_sd must be positive, got {baseline_sd!r}")
    if baseline_mean == 0:
        raise ValueError(
            "baseline_mean is zero, so a percentage lift has no meaning. "
            "Use an absolute effect size instead."
        )
    ratio = _check_share(treatment_share)
    cohen_d = baseline_mean * (min_lift_pct / 100.0) / baseline_sd
    treatment_n = TTestIndPower().solve_power(
        effect_size=cohen_d,
        power=power,
        nobs1=None,
        ratio=ratio,
        alpha=alpha,
        alternative=alternative,
    )
    return ExperimentSize(
        treatment_n=int(np.ceil(treatment_n)),
        control_n=int(np.ceil(treatment_n * ratio)),
        effect_size=float(cohen_d),
        detectable_lift_pct=float(min_lift_pct),
    )


def sample_size_for_proportion(
    baseline_rate: float,
    min_lift_pct: float,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ExperimentSize:
    """Units per arm needed to detect ``min_lift_pct`` on a rate.

    ``baseline_rate`` is a proportion in (0, 1), and ``min_lift_pct`` is
    relative to it: a baseline of 0.10 with a 10% lift targets 0.11, not 0.20.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(
            f"baseline_rate is a proportion and must be strictly between 0 "
            f"and 1, got {baseline_rate!r}"
        )
    ratio = _check_share(treatment_share)
    z_alpha, z_beta = _z(alpha, power, alternative)
    treated_rate = baseline_rate * (1.0 + min_lift_pct / 100.0)
    if not 0.0 < treated_rate < 1.0:
        raise ValueError(
            f"a {min_lift_pct}% lift on a baseline of {baseline_rate} implies "
            f"a rate of {treated_rate:.4f}, which is not a proportion"
        )
    var_sum = (
        baseline_rate * (1.0 - baseline_rate)
        + treated_rate * (1.0 - treated_rate)
    )
    n = var_sum / (baseline_rate - treated_rate) ** 2 * (z_alpha + z_beta) ** 2 * 2.0
    n_prime = n * (1.0 + ratio) ** 2 / (4.0 * ratio)
    treatment_n = n_prime / (1.0 + ratio)
    pooled = (baseline_rate + treated_rate) / 2.0
    effect = abs(treated_rate - baseline_rate) / np.sqrt(pooled * (1.0 - pooled))
    return ExperimentSize(
        treatment_n=int(np.ceil(treatment_n)),
        control_n=int(np.ceil(n_prime - treatment_n)),
        effect_size=float(effect),
        detectable_lift_pct=float(min_lift_pct),
    )


def detectable_lift_for_mean(
    baseline_mean: float,
    baseline_sd: float,
    total_n: int,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> float:
    """The smallest relative lift a fixed sample can detect, as a percentage.

    The other direction of ``sample_size_for_mean``: the sample is what it is,
    so the question is whether the experiment can see anything worth seeing.
    """
    if baseline_sd <= 0:
        raise ValueError(f"baseline_sd must be positive, got {baseline_sd!r}")
    if total_n <= 2:
        raise ValueError(f"total_n must be greater than 2, got {total_n!r}")
    ratio = _check_share(treatment_share)
    treatment_n = total_n * treatment_share
    effect = TTestIndPower().solve_power(
        effect_size=None,
        power=power,
        nobs1=treatment_n,
        ratio=ratio,
        alpha=alpha,
        alternative=alternative,
    )
    return float(effect * baseline_sd / baseline_mean * 100.0)


def detectable_lift_for_proportion(
    baseline_rate: float,
    total_n: int,
    treatment_share: float = 0.5,
    power: float = 0.80,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> float:
    """The smallest relative lift on a rate a fixed sample can detect.

    Uses the arcsine variance-stabilising transform, which behaves near 0 and
    1 where the plain normal approximation does not.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(
            f"baseline_rate is a proportion and must be strictly between 0 "
            f"and 1, got {baseline_rate!r}"
        )
    if total_n <= 2:
        raise ValueError(f"total_n must be greater than 2, got {total_n!r}")
    _check_share(treatment_share)
    z_alpha, z_beta = _z(alpha, power, alternative)
    treatment_n = total_n * treatment_share
    control_n = total_n - treatment_n
    harmonic = 2.0 * treatment_n * control_n / (treatment_n + control_n)
    effect = (z_alpha + z_beta) / np.sqrt(harmonic / 2.0)
    treated = np.sin(2.0 * np.arcsin(np.sqrt(baseline_rate)) / 2.0 + effect / 2.0) ** 2
    return float((treated - baseline_rate) / baseline_rate * 100.0)


def assign_groups(
    units: Iterable[object],
    ratios: Mapping[str, float],
    seed: int | None = None,
) -> pd.DataFrame:
    """Randomly assign units to named groups in the given proportions.

    Returns a frame of ``unit`` and ``group``. Pass ``seed`` to make an
    assignment reproducible, which is what makes a published experiment
    checkable by somebody else.

    >>> frame = assign_groups(range(100), {"control": 0.5, "treatment": 0.5}, seed=0)
    >>> sorted(frame["group"].unique())
    ['control', 'treatment']
    """
    unit_list = list(units)
    if not unit_list:
        raise ValueError("no units to assign")
    duplicates = len(unit_list) - len(set(unit_list))
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate unit(s) in the input. Assigning the same "
            f"unit twice puts it in two arms, which silently breaks the test."
        )
    names = list(ratios)
    weights = np.asarray([ratios[name] for name in names], dtype=float)
    if np.any(weights <= 0):
        raise ValueError(f"every ratio must be positive, got {dict(ratios)!r}")
    total = weights.sum()
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"ratios must sum to 1, got {total!r}. They are proportions of the "
            f"sample, not counts."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(names), size=len(unit_list), p=weights)
    return pd.DataFrame(
        {"unit": unit_list, "group": [names[i] for i in chosen]}
    )


def cuped_adjust(
    data: pd.DataFrame,
    metric_col: str,
    pre_col: str,
) -> pd.Series:
    """Variance-reduced metric using a pre-period covariate (CUPED).

    Returns an adjusted copy of ``metric_col`` with the same mean and lower
    variance, so the same experiment detects a smaller effect. The covariate
    must be measured BEFORE assignment: a post-period covariate can be
    affected by the treatment, and adjusting on it biases the result.

    A covariate with no variance carries no information, so the metric is
    returned unchanged rather than raising, and a constant column cannot
    silently produce a divide by zero.
    """
    for column in (metric_col, pre_col):
        if column not in data.columns:
            raise KeyError(f"{column!r} is not a column of the data")
    pre = data[pre_col].astype(float)
    metric = data[metric_col].astype(float)
    pre_var = float(pre.var(ddof=1))
    if not np.isfinite(pre_var) or pre_var < 1e-12:
        return metric.copy()
    theta = float(pre.cov(metric)) / pre_var
    return metric - theta * (pre - pre.mean())


def _welch(
    treated: "pd.Series[float]", control: "pd.Series[float]", alpha: float,
    alternative: Alternative,
) -> dict[str, object]:
    """One Welch comparison, as a row."""
    n1, n2 = int(treated.count()), int(control.count())
    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"each arm needs at least 2 observations to have a variance, got "
            f"treatment={n1} and control={n2}"
        )
    m1, m2 = float(treated.mean()), float(control.mean())
    s1, s2 = float(treated.std(ddof=1)), float(control.std(ddof=1))
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        t_stat, pval, dof = np.nan, np.nan, np.nan
    else:
        dof = (s1**2 / n1 + s2**2 / n2) ** 2 / (
            (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        )
        t_stat = (m1 - m2) / se
        if alternative == "two-sided":
            pval = float(2.0 * stats.t.sf(abs(t_stat), dof))
        elif alternative == "larger":
            pval = float(stats.t.sf(t_stat, dof))
        else:
            pval = float(stats.t.cdf(t_stat, dof))
    diff = m1 - m2
    if np.isnan(dof):
        low, high = np.nan, np.nan
    else:
        crit = stats.t.ppf(1.0 - alpha / 2.0, dof)
        low, high = diff - crit * se, diff + crit * se
    return {
        "treatment_n": n1,
        "control_n": n2,
        "treatment_mean": m1,
        "control_mean": m2,
        "treatment_sd": s1,
        "control_sd": s2,
        "mean_difference": diff,
        "ci_low": float(low),
        "ci_high": float(high),
        "percent_lift": (diff / m2 * 100.0) if m2 != 0 else np.nan,
        "t_statistic": float(t_stat),
        "p_value": float(pval),
    }


def compare_groups(
    data: pd.DataFrame,
    group_col: str,
    metric_cols: Sequence[str],
    control: str,
    treatment: str,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    correction: str | None = "fdr_bh",
    cuped_pre: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compare two arms across metrics, one row per metric.

    Welch's t-test throughout, so unequal variances between arms are handled
    rather than assumed away. ``correction`` applies a multiple-testing
    correction across the metrics and adds a ``p_value_corrected`` column;
    pass ``None`` to skip it, which is only right when there is one metric.

    ``cuped_pre`` maps a metric to its pre-period column and applies
    :func:`cuped_adjust` before testing, which lowers the variance without
    changing what is being estimated.
    """
    if group_col not in data.columns:
        raise KeyError(f"{group_col!r} is not a column of the data")
    if not metric_cols:
        raise ValueError("no metrics given to compare")
    present = set(data[group_col].unique())
    missing = {control, treatment} - present
    if missing:
        raise ValueError(
            f"group(s) {sorted(missing)!r} do not appear in {group_col!r}. "
            f"Present: {sorted(map(str, present))!r}"
        )

    rows = []
    for metric in metric_cols:
        if metric not in data.columns:
            raise KeyError(f"{metric!r} is not a column of the data")
        frame = data
        column = metric
        if cuped_pre and metric in cuped_pre:
            frame = data.assign(
                **{f"__cuped_{metric}": cuped_adjust(data, metric, cuped_pre[metric])}
            )
            column = f"__cuped_{metric}"
        treated = frame.loc[frame[group_col] == treatment, column]
        controlled = frame.loc[frame[group_col] == control, column]
        row: dict[str, object] = {
            "metric": metric,
            "treatment_group": treatment,
            "control_group": control,
            "cuped": bool(cuped_pre and metric in cuped_pre),
        }
        row.update(_welch(treated, controlled, alpha, alternative))
        rows.append(row)

    result = pd.DataFrame(rows)
    if correction is not None and len(result) > 1:
        usable = result["p_value"].notna()
        result["p_value_corrected"] = np.nan
        if usable.any():
            result.loc[usable, "p_value_corrected"] = multipletests(
                result.loc[usable, "p_value"].to_numpy(),
                alpha=alpha,
                method=correction,
            )[1]
    return result


def pretest_balance(
    data: pd.DataFrame,
    group_col: str,
    pre_cols: Sequence[str],
    control: str,
    treatment: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Check the arms looked the same BEFORE the treatment ran.

    Randomisation is supposed to make the arms exchangeable. Testing the
    pre-period metrics is how that gets checked rather than assumed: a
    significant difference here means the split is suspect and the headline
    result should not be read at face value.

    A pre-period column that is constant or all zero carries no information,
    which happens legitimately when the units are new and have no history. It
    is reported as ``uninformative`` rather than tested, because a degenerate
    t-test returns a confident-looking answer built on nothing.
    """
    rows = []
    informative = []
    for column in pre_cols:
        if column not in data.columns:
            raise KeyError(f"{column!r} is not a column of the data")
        series = data[column].astype(float)
        if float(series.std(ddof=0)) < 1e-12:
            rows.append(
                {
                    "metric": column,
                    "treatment_group": treatment,
                    "control_group": control,
                    "uninformative": True,
                }
            )
        else:
            informative.append(column)

    if informative:
        tested = compare_groups(
            data,
            group_col=group_col,
            metric_cols=informative,
            control=control,
            treatment=treatment,
            alpha=alpha,
        )
        tested["uninformative"] = False
        rows.extend(tested.to_dict("records"))

    result = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(pre_cols)}
    return result.sort_values("metric", key=lambda s: s.map(order)).reset_index(
        drop=True
    )
