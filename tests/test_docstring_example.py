"""Run the usage example from Simulation's docstring.

A documented example that does not run is worse than no example: it looks
authoritative and wastes the reader's time. This executes the exact shape shown
in `Simulation`'s docstring, so the docs cannot drift from the API silently.

Offline: no network in any of these paths.
"""

import numpy as np
import pandas as pd

from research_framework.simulation import (
    ConvergenceDiagnostics,
    DistributionSpec,
    Scenario,
    Simulation,
)


def _sim(**kwargs):
    return Simulation(
        variables=[
            DistributionSpec("price", "normal", {"mean": 10, "std": 2}),
            DistributionSpec(
                "units", "triangular", {"left": 80, "mode": 100, "right": 150}
            ),
        ],
        model=lambda row: row["price"] * row["units"],
        seed=42,
        n_iterations=500,
        **kwargs,
    )


def test_the_documented_example_runs_end_to_end():
    sim = _sim()
    result = sim.run()

    assert result.n_iterations == 500
    assert result.mean is not None          # run() summarises, as documented
    assert result.ci_lower <= result.median <= result.ci_upper

    tornado = sim.sensitivity.tornado()
    assert set(tornado["variable"]) == {"price", "units"}
    # documented as biggest swing first
    assert tornado["swing"].is_monotonic_decreasing

    conv = sim.check_convergence(result)
    assert set(conv) == {"is_converged", "suggested_n", "current_n", "relative_se"}
    assert conv["current_n"] == 500


def test_seed_makes_a_run_repeatable_as_documented():
    a = _sim().run()
    b = _sim().run()
    assert a.mean == b.mean


def test_vectorized_flag_takes_the_whole_frame():
    """Documented on Simulation as the faster path."""
    seen = {}

    def model(df):
        seen["type"] = type(df)
        return df["price"] * df["units"]

    sim = Simulation(
        variables=[
            DistributionSpec("price", "normal", {"mean": 10, "std": 2}),
            DistributionSpec("units", "normal", {"mean": 100, "std": 10}),
        ],
        model=model,
        vectorized=True,
        seed=1,
        n_iterations=200,
    )
    sim.run()
    assert seen["type"] is pd.DataFrame


def test_correlation_matrix_preserves_the_marginals():
    """The copula docstring on _draw_correlated claims each marginal survives.

    A lognormal drawn under correlation must stay strictly positive; if the
    implementation returned the underlying normals it would not.
    """
    sim = Simulation(
        variables=[
            DistributionSpec("a", "lognormal", {"mean": 0.0, "sigma": 0.5}),
            DistributionSpec("b", "normal", {"mean": 0.0, "std": 1.0}),
        ],
        model=lambda row: row["a"] + row["b"],
        correlation_matrix=np.array([[1.0, 0.6], [0.6, 1.0]]),
        seed=7,
        n_iterations=2000,
    )
    result = sim.run()
    draws = result.draws
    assert (draws["a"] > 0).all()
    # and the requested correlation actually shows up, loosely
    assert draws["a"].corr(draws["b"]) > 0.3


def test_scenario_overrides_merge_into_existing_params():
    """Scenario's docstring says an override can change a mean without
    restating the standard deviation."""
    sim = _sim()
    results = sim.compare_scenarios(
        [Scenario("higher_price", {"price": {"mean": 20}})]
    )
    assert set(results) == {"baseline", "higher_price"}
    assert results["higher_price"].mean > results["baseline"].mean


def test_suggest_n_never_suggests_less_than_you_ran():
    outcomes = np.random.default_rng(0).normal(100, 15, size=400)
    assert ConvergenceDiagnostics.suggest_n(outcomes) >= len(outcomes)


def test_is_converged_is_false_on_too_little_data():
    """Documented: too little to judge reads as not converged."""
    assert ConvergenceDiagnostics.is_converged(np.array([1.0, 2.0, 3.0])) is False


def test_run_convergence_snapshots_are_nested_prefixes():
    """run_convergence's docstring explains it slices ONE run rather than
    running several, which is why the snapshots nest."""
    sim = _sim()
    snaps = sim.engine.run_convergence(checkpoints=[100, 200, 500])
    assert [s.n_iterations for s in snaps] == [100, 200, 500]
    biggest = np.asarray(snaps[-1].outcomes)
    for snap in snaps[:-1]:
        prefix = np.asarray(snap.outcomes)
        assert np.allclose(prefix, biggest[: len(prefix)])
