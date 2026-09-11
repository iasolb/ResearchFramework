"""
VC Portfolio Monte Carlo Simulation
=====================================
Demonstrates: full simulation workflow using the simulations module -
              distribution fitting from observed data, manual specs,
              correlated draws, scenario comparison, sensitivity
              analysis, convergence diagnostics, and plotting.

Scenario:
    A venture fund holds 50 early-stage startups. We want to estimate
    total portfolio value after 3 years under uncertainty about growth
    rates, churn, and market multiples. We compare baseline, bull, and
    bear market scenarios.

Run:   python examples/monte_carlo_test.py
Data:  examples/data/startup_portfolio.csv
"""

import os
import numpy as np
import pandas as pd

from otter import (
    ConvergenceDiagnostics,
    DistributionSpec,
    InputManager,
    ModelFunction,
    MonteCarloEngine,
    SensitivityAnalyzer,
    Simulation,
    SimulationPlotter,
    SimulationResult,
    Scenario,
)
from otter.simulation import ScenarioComparator

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "startup_portfolio.csv")


# ---------------------------------------------------------------------------
# Portfolio valuation model
# ---------------------------------------------------------------------------


def portfolio_value(row):
    """
    Simplified 3-year forward valuation for a SaaS portfolio.

    Inputs (drawn from distributions):
        revenue_multiple   - market multiple applied to ARR
        growth_rate        - annualized portfolio revenue growth
        churn_rate         - annual customer churn (revenue drag)
        discount_rate      - rate used to discount terminal value

    Uses the portfolio's observed median MRR ($33K) as the baseline
    monthly revenue, then projects forward 3 years with growth and churn.
    """
    base_arr = 33.0 * 12  # median MRR * 12 -> baseline ARR ($K)
    years = 3

    # net growth after churn
    net_growth = row["growth_rate"] - row["churn_rate"]
    projected_arr = base_arr * (1 + net_growth) ** years

    # terminal value = ARR * market multiple, discounted back
    terminal = projected_arr * row["revenue_multiple"]
    present_value = terminal / (1 + row["discount_rate"]) ** years

    return present_value


# ---------------------------------------------------------------------------
# Step 1: Fit distributions from observed data
# ---------------------------------------------------------------------------


def step_1_fit_from_data():
    print("\n" + "=" * 60)
    print("STEP 1: FIT DISTRIBUTIONS FROM OBSERVED DATA")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    mgr = InputManager()

    # Fit growth and churn from the actual portfolio data
    specs = mgr.fit_from_data(df, ["growth_rate", "churn_rate"], dist_type="normal")
    for spec in specs:
        print(f"  {spec.name}: {spec.dist_type} -> {spec.params}")

    # Infer correlation from the observed data
    corr = mgr.infer_correlation_from_data(df)
    print(f"\n  Empirical correlation matrix:")
    print(f"  {mgr.variable_names}")
    print(f"  {corr.round(3)}")

    # Draw and verify
    draws = mgr.draw(5_000, seed=42)
    empirical_corr = draws["growth_rate"].corr(draws["churn_rate"])
    print(f"\n  Simulated correlation (growth vs churn): {empirical_corr:.3f}")
    print(f"  Draw shape: {draws.shape}")

    return specs


# ---------------------------------------------------------------------------
# Step 2: Full simulation via the top-level facade
# ---------------------------------------------------------------------------


def step_2_run_simulation():
    print("\n" + "=" * 60)
    print("STEP 2: MONTE CARLO SIMULATION (10,000 iterations)")
    print("=" * 60)

    sim = Simulation(
        variables=[
            DistributionSpec("growth_rate", "normal", {"mean": 0.40, "std": 0.15}),
            DistributionSpec("churn_rate", "beta", {"a": 2, "b": 30}),
            DistributionSpec(
                "revenue_multiple", "triangular", {"left": 3, "mode": 8, "right": 20}
            ),
            DistributionSpec("discount_rate", "normal", {"mean": 0.12, "std": 0.03}),
        ],
        model=portfolio_value,
        n_iterations=10_000,
        seed=42,
    )

    result = sim.run()
    summary = result.summarize()

    print(f"  Mean portfolio value:   ${summary['mean']:>12,.0f}K")
    print(f"  Median:                 ${summary['median']:>12,.0f}K")
    print(f"  Std deviation:          ${summary['std']:>12,.0f}K")
    print(
        f"  95% CI:                 [${summary['ci_lower']:>,.0f}K, ${summary['ci_upper']:>,.0f}K]"
    )
    print(f"  5th percentile:         ${summary['percentiles'][5]:>12,.0f}K")
    print(f"  95th percentile:        ${summary['percentiles'][95]:>12,.0f}K")

    return sim, result


# ---------------------------------------------------------------------------
# Step 3: Sensitivity analysis
# ---------------------------------------------------------------------------


def step_3_sensitivity(sim):
    print("\n" + "=" * 60)
    print("STEP 3: SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Tornado chart data
    print("\n  --- Tornado (10th-90th percentile) ---")
    tornado = sim.sensitivity.tornado()
    for _, row in tornado.iterrows():
        print(
            f"  {row['variable']:>20s}:  "
            f"swing = ${row['swing']:>10,.0f}K  "
            f"[{row['low_outcome']:>10,.0f} -> {row['high_outcome']:>10,.0f}]"
        )

    # One-at-a-time for the biggest driver
    top_var = tornado.iloc[0]["variable"]
    print(f"\n  --- One-at-a-time sweep: {top_var} ---")
    oat = sim.sensitivity.one_at_a_time(top_var, n_steps=6)
    for _, row in oat.iterrows():
        print(
            f"  {top_var} = {row['variable_value']:>8.2f}  ->  ${row['outcome']:>10,.0f}K"
        )

    # Sobol indices (smaller n for speed in example)
    print("\n  --- Sobol first-order indices ---")
    sobol = sim.sensitivity.sobol_indices(n_samples=2_000, seed=99)
    for _, row in sobol.iterrows():
        print(
            f"  {row['variable']:>20s}:  S1 = {row['S1']:.3f}  "
            f"(+/-{row['S1_conf']:.3f})"
        )

    return tornado


# ---------------------------------------------------------------------------
# Step 4: Scenario comparison
# ---------------------------------------------------------------------------


def step_4_scenarios(sim):
    print("\n" + "=" * 60)
    print("STEP 4: SCENARIO COMPARISON")
    print("=" * 60)

    scenarios = [
        Scenario(
            "bull_market",
            overrides={
                "revenue_multiple": {"left": 8, "mode": 15, "right": 30},
                "growth_rate": {"mean": 0.55},
            },
        ),
        Scenario(
            "bear_market",
            overrides={
                "revenue_multiple": {"left": 2, "mode": 4, "right": 8},
                "growth_rate": {"mean": 0.20},
                "discount_rate": {"mean": 0.18},
            },
        ),
        Scenario(
            "high_churn_crisis",
            overrides={
                "churn_rate": {"a": 3, "b": 10},
            },
        ),
    ]

    summary = sim.compare_scenarios_summary(scenarios)
    print("\n" + summary.to_string(index=False))

    return scenarios


# ---------------------------------------------------------------------------
# Step 5: Convergence diagnostics
# ---------------------------------------------------------------------------


def step_5_convergence(sim, result):
    print("\n" + "=" * 60)
    print("STEP 5: CONVERGENCE DIAGNOSTICS")
    print("=" * 60)

    report = sim.check_convergence(result)
    print(f"  Converged:     {report['is_converged']}")
    print(f"  Relative SE:   {report['relative_se']:.4f}")
    print(f"  Current N:     {report['current_n']:,}")
    print(f"  Suggested N:   {report['suggested_n']:,}")

    # Convergence snapshots
    print("\n  --- Convergence snapshots ---")
    snapshots = sim.engine.run_convergence()
    for snap in snapshots:
        print(
            f"  n={snap.n_iterations:>6,}  "
            f"mean=${snap.mean:>10,.0f}K  "
            f"std=${snap.std:>10,.0f}K"
        )


# ---------------------------------------------------------------------------
# Step 6: Empirical resampling from observed data
# ---------------------------------------------------------------------------


def step_6_empirical():
    print("\n" + "=" * 60)
    print("STEP 6: EMPIRICAL RESAMPLING FROM OBSERVED DATA")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    observed_growth = df["growth_rate"].dropna().values

    sim = Simulation(
        variables=[
            # resample growth from actual portfolio instead of assuming normal
            DistributionSpec(
                "growth_rate", "empirical", empirical_data=observed_growth
            ),
            DistributionSpec("churn_rate", "beta", {"a": 2, "b": 30}),
            DistributionSpec(
                "revenue_multiple", "triangular", {"left": 3, "mode": 8, "right": 20}
            ),
            DistributionSpec("discount_rate", "normal", {"mean": 0.12, "std": 0.03}),
        ],
        model=portfolio_value,
        n_iterations=10_000,
        seed=42,
    )

    result = sim.run()
    print(f"  Mean (empirical growth):  ${result.mean:>12,.0f}K")
    print(f"  Std:                      ${result.std:>12,.0f}K")
    print(f"  95% CI: [${result.ci_lower:>,.0f}K, ${result.ci_upper:>,.0f}K]")


# ---------------------------------------------------------------------------
# Step 7: Plots (saved to disk)
# ---------------------------------------------------------------------------


def step_7_plots(sim, result, tornado_data, scenarios):
    print("\n" + "=" * 60)
    print("STEP 7: GENERATING PLOTS")
    print("=" * 60)

    # matplotlib is NOT a declared dependency of otter (the package ships
    # plotly), so this step is optional rather than fatal. It used to raise
    # ModuleNotFoundError and take the whole example down at the last step,
    # after six steps of real output had already succeeded.
    try:
        import matplotlib
    except ImportError:
        print("  SKIPPED: matplotlib is not installed, and it is not a")
        print("  dependency of otter. Install it to write the plots:")
        print("      pip install matplotlib")
        return

    matplotlib.use("Agg")

    out_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(out_dir, exist_ok=True)

    fig = sim.plot.histogram(result)
    path = os.path.join(out_dir, "histogram.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")

    fig = sim.plot.cumulative_density(result)
    path = os.path.join(out_dir, "cdf.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")

    fig = sim.plot.convergence_plot(result.outcomes)
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")

    fig = sim.plot.tornado_chart(tornado_data)
    path = os.path.join(out_dir, "tornado.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")

    scenario_results = sim.compare_scenarios(scenarios)
    fig = sim.plot.scenario_comparison(scenario_results)
    path = os.path.join(out_dir, "scenarios.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def ensure_data():
    """Write the portfolio CSV if it is not there.

    The README promises every example generates its own data and runs on a
    fresh clone. This one read a file that was never committed, so it failed
    immediately with a FileNotFoundError. Growth and churn are drawn
    correlated on purpose, because step 1 infers a correlation matrix from
    them and independent columns would make that step demonstrate nothing.
    """
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    rng = np.random.default_rng(42)
    n = 240
    growth = rng.normal(0.22, 0.09, n)
    # faster-growing companies churn a little less, so the two are negatively
    # correlated rather than independent
    churn = np.clip(0.09 - 0.18 * (growth - 0.22) + rng.normal(0, 0.02, n), 0.005, None)
    pd.DataFrame(
        {
            "company_id": [f"co_{i:03d}" for i in range(n)],
            "mrr": rng.lognormal(np.log(33.0), 0.8, n).round(1),
            "growth_rate": growth.round(4),
            "churn_rate": churn.round(4),
        }
    ).to_csv(DATA_PATH, index=False)
    print(f"Generated synthetic portfolio data: {DATA_PATH}")


def main():
    ensure_data()
    fitted_specs = step_1_fit_from_data()
    sim, result = step_2_run_simulation()
    tornado_data = step_3_sensitivity(sim)
    scenarios = step_4_scenarios(sim)
    step_5_convergence(sim, result)
    step_6_empirical()
    step_7_plots(sim, result, tornado_data, scenarios)

    print("\n" + "=" * 60)
    print("DONE - all steps completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
