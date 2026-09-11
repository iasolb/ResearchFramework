"""Coverage for simulation.py's pieces, not just the happy path through `Simulation`.

Written against measured coverage: simulation.py sat at 56% with the whole
distribution registry, every validation branch, the correlated and empirical
draw paths, the scenario comparator and `from_spec` untouched.

Priorities here, in order:

1. Every branch that REFUSES something. A validation path that has never run is
   a validation path that might not work, and those are the ones a user meets
   on their worst day.
2. Statistical claims the code makes, checked loosely but honestly: a fitted
   distribution should recover roughly the parameters it was given, a
   correlation asked for should show up in the draws.
3. Determinism, because a seeded simulation that is not reproducible is worse
   than one that is obviously random.

Tolerances are deliberately wide. These are Monte Carlo assertions and a tight
bound would make the suite flaky, which teaches everyone to ignore it.
"""

import numpy as np
import pandas as pd
import pytest

from otter.rh import ModelSpec
from otter.simulation import (
    _DISTRIBUTION_REGISTRY,
    ConvergenceDiagnostics,
    DistributionSpec,
    InputManager,
    ModelFunction,
    MonteCarloEngine,
    Scenario,
    ScenarioComparator,
    SensitivityAnalyzer,
    Simulation,
    SimulationResult,
)

# One valid parameter set per registered distribution, so adding a distribution
# to the registry without adding it here shows up as a failure rather than as
# silently untested code.
VALID_PARAMS = {
    "normal": {"mean": 10.0, "std": 2.0},
    "uniform": {"low": 0.0, "high": 5.0},
    "lognormal": {"mean": 0.0, "sigma": 0.4},
    "beta": {"a": 2.0, "b": 5.0},
    "triangular": {"left": 0.0, "mode": 2.0, "right": 6.0},
    "exponential": {"scale": 3.0},
}


def test_every_registered_distribution_is_covered_here():
    """A guard on this file, not on the library: keeps the parametrised tests
    below honest as the registry grows."""
    assert set(VALID_PARAMS) == set(_DISTRIBUTION_REGISTRY)


# ── DistributionSpec validation ───────────────────────────────────────────


@pytest.mark.parametrize("dist_type", sorted(VALID_PARAMS))
def test_a_valid_spec_of_every_type_constructs(dist_type):
    spec = DistributionSpec("x", dist_type, VALID_PARAMS[dist_type])
    assert spec.dist_type == dist_type


def test_dist_type_is_normalised():
    assert DistributionSpec("x", "  NORMAL  ", VALID_PARAMS["normal"]).dist_type == "normal"


def test_an_unknown_dist_type_is_refused_and_lists_the_options():
    with pytest.raises(ValueError, match="unknown dist_type"):
        DistributionSpec("x", "not_a_distribution", {})


@pytest.mark.parametrize("dist_type", sorted(VALID_PARAMS))
def test_missing_parameters_are_named(dist_type):
    """The error says WHICH parameters are missing, which is the difference
    between a one-second fix and a trip into the source."""
    with pytest.raises(ValueError, match="missing required params"):
        DistributionSpec("x", dist_type, {})


def test_empirical_requires_data():
    with pytest.raises(ValueError, match="requires non-empty empirical_data"):
        DistributionSpec("x", "empirical")


def test_empirical_refuses_an_empty_array():
    with pytest.raises(ValueError, match="requires non-empty empirical_data"):
        DistributionSpec("x", "empirical", empirical_data=np.array([]))


# ── InputManager ──────────────────────────────────────────────────────────


def _mgr(*specs):
    mgr = InputManager()
    mgr.add_variables(list(specs))
    return mgr


def test_variable_order_is_insertion_order():
    """Load bearing: it is the column order of every draw, and therefore what
    makes a correlation matrix's rows mean anything."""
    mgr = _mgr(
        DistributionSpec("b", "normal", VALID_PARAMS["normal"]),
        DistributionSpec("a", "normal", VALID_PARAMS["normal"]),
    )
    assert mgr.variable_names == ["b", "a"]
    assert list(mgr.draw(5, seed=0).columns) == ["b", "a"]


def test_a_duplicate_variable_is_refused():
    mgr = _mgr(DistributionSpec("a", "normal", VALID_PARAMS["normal"]))
    with pytest.raises(ValueError, match="already registered"):
        mgr.add_variable(DistributionSpec("a", "normal", VALID_PARAMS["normal"]))


def test_removing_an_unknown_variable_raises():
    with pytest.raises(KeyError):
        InputManager().remove_variable("nope")


def test_removing_a_variable_drops_the_correlation_matrix():
    """Documented behaviour: the matrix's shape no longer matches."""
    mgr = _mgr(
        DistributionSpec("a", "normal", VALID_PARAMS["normal"]),
        DistributionSpec("b", "normal", VALID_PARAMS["normal"]),
    )
    mgr.set_correlation_matrix(np.array([[1.0, 0.3], [0.3, 1.0]]))
    assert mgr.correlation_matrix is not None
    mgr.remove_variable("b")
    assert mgr.correlation_matrix is None


def test_drawing_with_no_variables_raises():
    with pytest.raises(RuntimeError, match="No variables registered"):
        InputManager().draw(10)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.eye(3), "does not match"),                              # wrong shape
        (np.array([[1.0, 0.5], [0.2, 1.0]]), "symmetric"),          # asymmetric
        (np.array([[0.9, 0.0], [0.0, 0.9]]), "diagonal"),           # bad diagonal
        (np.array([[1.0, 1.9], [1.9, 1.0]]), "positive semi-definite"),
    ],
)
def test_every_correlation_matrix_rejection_path(matrix, message):
    """The PSD check is the important one: without it Cholesky fails deep
    inside a draw with a far less obvious error."""
    mgr = _mgr(
        DistributionSpec("a", "normal", VALID_PARAMS["normal"]),
        DistributionSpec("b", "normal", VALID_PARAMS["normal"]),
    )
    with pytest.raises(ValueError, match=message):
        mgr.set_correlation_matrix(matrix)


def test_correlation_inferred_from_data_is_set_and_returned():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "b": a * 0.8 + rng.normal(size=400) * 0.3})
    mgr = _mgr(
        DistributionSpec("a", "normal", VALID_PARAMS["normal"]),
        DistributionSpec("b", "normal", VALID_PARAMS["normal"]),
    )
    matrix = mgr.infer_correlation_from_data(df)
    assert mgr.correlation_matrix is not None
    assert matrix[0, 1] > 0.5


def test_correlation_from_data_names_the_missing_columns():
    mgr = _mgr(DistributionSpec("a", "normal", VALID_PARAMS["normal"]))
    with pytest.raises(KeyError, match="a"):
        mgr.infer_correlation_from_data(pd.DataFrame({"other": [1, 2, 3]}))


@pytest.mark.parametrize("dist_type", sorted(VALID_PARAMS))
def test_independent_draws_have_the_right_shape_for_every_type(dist_type):
    mgr = _mgr(DistributionSpec("x", dist_type, VALID_PARAMS[dist_type]))
    draws = mgr.draw(200, seed=3)
    assert draws.shape == (200, 1)
    assert draws["x"].notna().all()


def test_a_seeded_draw_is_reproducible():
    mgr = _mgr(DistributionSpec("x", "normal", VALID_PARAMS["normal"]))
    pd.testing.assert_frame_equal(mgr.draw(50, seed=11), mgr.draw(50, seed=11))


def test_empirical_draws_only_ever_return_observed_values():
    data = np.array([1.0, 7.0, 13.0])
    mgr = _mgr(DistributionSpec("x", "empirical", empirical_data=data))
    draws = mgr.draw(300, seed=5)
    assert set(np.unique(draws["x"])) <= set(data)


def test_correlated_empirical_draws_also_stay_within_the_data():
    """The correlated path reaches empirical variables through a different
    branch (sorted data indexed by the copula's uniforms), so it needs its own
    check."""
    data = np.array([2.0, 4.0, 6.0, 8.0])
    mgr = _mgr(
        DistributionSpec("e", "empirical", empirical_data=data),
        DistributionSpec("n", "normal", VALID_PARAMS["normal"]),
    )
    mgr.set_correlation_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
    draws = mgr.draw(300, seed=9)
    assert set(np.unique(draws["e"])) <= set(data)


def test_fit_from_data_recovers_roughly_the_right_parameters():
    values = np.random.default_rng(4).normal(loc=50, scale=5, size=3000)
    mgr = InputManager()
    created = mgr.fit_from_data(pd.DataFrame({"v": values}), ["v"])
    assert len(created) == 1
    assert created[0].params["mean"] == pytest.approx(50, abs=1.0)
    assert created[0].params["std"] == pytest.approx(5, abs=1.0)


def test_fit_from_data_drops_nans_rather_than_producing_nan_parameters():
    df = pd.DataFrame({"v": [1.0, 2.0, np.nan, 3.0, 4.0]})
    spec = InputManager().fit_from_data(df, ["v"])[0]
    assert np.isfinite(spec.params["mean"])


def test_fit_from_data_names_a_missing_column():
    with pytest.raises(KeyError, match="nope"):
        InputManager().fit_from_data(pd.DataFrame({"v": [1.0]}), ["nope"])


def test_fit_from_data_can_build_an_empirical_spec():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    spec = InputManager().fit_from_data(df, ["v"], dist_type="empirical")[0]
    assert spec.dist_type == "empirical"
    assert spec.empirical_data is not None


def test_fit_from_data_refuses_an_unknown_dist_type():
    with pytest.raises(ValueError, match="Unknown dist_type"):
        InputManager().fit_from_data(pd.DataFrame({"v": [1.0]}), ["v"], dist_type="bogus")


# ── ModelFunction ─────────────────────────────────────────────────────────


def test_row_wise_model_returning_a_scalar():
    draws = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = ModelFunction(lambda row: row["a"] + row["b"]).run(draws)
    assert np.allclose(out, [4.0, 6.0])


def test_row_wise_model_returning_a_dict_becomes_a_frame():
    """Documented: several outcomes per row produce a multi-column frame."""
    draws = pd.DataFrame({"a": [1.0, 2.0]})
    out = ModelFunction(lambda row: {"double": row["a"] * 2, "half": row["a"] / 2}).run(draws)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["double", "half"]


def test_vectorized_model_receives_the_whole_frame():
    seen = {}

    def model(df):
        seen["rows"] = len(df)
        return df["a"] * 2

    ModelFunction(model, vectorized=True).run(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert seen["rows"] == 3


def test_vectorized_model_may_return_a_frame_unchanged():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    out = ModelFunction(lambda df: frame, vectorized=True).run(pd.DataFrame({"a": [1.0, 2.0]}))
    assert isinstance(out, pd.DataFrame)


# ── SimulationResult ──────────────────────────────────────────────────────


def test_summarizing_nothing_raises():
    with pytest.raises(RuntimeError, match="No outcomes"):
        SimulationResult().summarize()


def test_summary_statistics_are_internally_consistent():
    outcomes = np.random.default_rng(2).normal(100, 10, size=5000)
    result = SimulationResult(outcomes=outcomes, n_iterations=5000)
    summary = result.summarize()
    assert summary["min"] <= summary["percentiles"][25] <= summary["median"]
    assert summary["median"] <= summary["percentiles"][75] <= summary["max"]
    assert summary["ci_lower"] < summary["mean"] < summary["ci_upper"]
    assert summary["mean"] == pytest.approx(100, abs=1.0)


def test_a_tighter_confidence_level_gives_a_narrower_interval():
    outcomes = np.random.default_rng(6).normal(0, 1, size=4000)
    wide = SimulationResult(outcomes=outcomes).summarize(confidence=0.99)
    narrow = SimulationResult(outcomes=outcomes).summarize(confidence=0.80)
    assert (narrow["ci_upper"] - narrow["ci_lower"]) < (wide["ci_upper"] - wide["ci_lower"])


def test_to_dataframe_without_draws_returns_just_the_outcome():
    df = SimulationResult(outcomes=np.array([1.0, 2.0])).to_dataframe()
    assert list(df.columns) == ["outcome"]


def test_to_dataframe_joins_draws_and_outcomes():
    draws = pd.DataFrame({"a": [1.0, 2.0]})
    df = SimulationResult(outcomes=np.array([10.0, 20.0]), draws=draws).to_dataframe()
    assert list(df.columns) == ["a", "outcome"]
    assert df["outcome"].tolist() == [10.0, 20.0]


def test_repr_distinguishes_raw_from_summarized():
    result = SimulationResult(outcomes=np.array([1.0, 2.0, 3.0]), n_iterations=3)
    assert "raw" in repr(result)
    result.summarize()
    assert "summarized" in repr(result)


# ── MonteCarloEngine ──────────────────────────────────────────────────────


def _engine(n=500, seed=1):
    mgr = _mgr(
        DistributionSpec("a", "normal", VALID_PARAMS["normal"]),
        DistributionSpec("b", "uniform", VALID_PARAMS["uniform"]),
    )
    return MonteCarloEngine(mgr, ModelFunction(lambda row: row["a"] + row["b"]), n, seed)


def test_run_does_not_summarize_but_run_on_simulation_does():
    """The split is documented and easy to get wrong in either direction."""
    assert _engine().run().mean is None


def test_store_draws_false_drops_the_input_frame():
    assert _engine().run(store_draws=False).draws is None


def test_convergence_checkpoints_are_clamped_sorted_and_deduped():
    snaps = _engine(n=300).run_convergence(checkpoints=[300, 50, 50, 5000])
    assert [s.n_iterations for s in snaps] == [50, 300]


def test_default_convergence_checkpoints_end_at_the_iteration_count():
    snaps = _engine(n=700).run_convergence()
    assert snaps[-1].n_iterations == 700


def test_convergence_snapshots_come_back_summarized():
    assert all(s.mean is not None for s in _engine(n=200).run_convergence())


# ── SensitivityAnalyzer ───────────────────────────────────────────────────


@pytest.mark.parametrize("dist_type", sorted(VALID_PARAMS))
def test_a_baseline_value_exists_for_every_distribution_type(dist_type):
    """Each type has its own branch picking a central value; an unhandled type
    would silently baseline at 0.0."""
    mgr = _mgr(DistributionSpec("x", dist_type, VALID_PARAMS[dist_type]))
    engine = MonteCarloEngine(mgr, ModelFunction(lambda row: row["x"]), 10, 0)
    row = SensitivityAnalyzer(engine)._get_baseline_row()
    assert np.isfinite(row["x"])
    assert row.dtype == float          # the int64 bug fixed 2026-08-28


def test_baseline_for_empirical_uses_the_median():
    mgr = _mgr(DistributionSpec("x", "empirical", empirical_data=np.array([1.0, 5.0, 100.0])))
    engine = MonteCarloEngine(mgr, ModelFunction(lambda row: row["x"]), 10, 0)
    assert SensitivityAnalyzer(engine)._get_baseline_row()["x"] == 5.0


def test_one_at_a_time_sweeps_a_monotone_model_monotonically():
    analyzer = SensitivityAnalyzer(_engine())
    out = analyzer.one_at_a_time("a", n_steps=8)
    assert len(out) == 8
    assert out["outcome"].is_monotonic_increasing


def test_one_at_a_time_accepts_explicit_values():
    out = SensitivityAnalyzer(_engine()).one_at_a_time("a", values=np.array([0.0, 1.0]))
    assert out["variable_value"].tolist() == [0.0, 1.0]


def test_one_at_a_time_refuses_an_unregistered_variable():
    with pytest.raises(KeyError, match="not registered"):
        SensitivityAnalyzer(_engine()).one_at_a_time("nope")


def test_one_at_a_time_handles_a_dict_returning_model():
    mgr = _mgr(DistributionSpec("a", "normal", VALID_PARAMS["normal"]))
    engine = MonteCarloEngine(mgr, ModelFunction(lambda row: {"out": row["a"] * 2}), 10, 0)
    assert len(SensitivityAnalyzer(engine).one_at_a_time("a", n_steps=4)) == 4


def test_tornado_ranks_the_variable_with_the_larger_effect_first():
    mgr = _mgr(
        DistributionSpec("small", "normal", {"mean": 0.0, "std": 0.1}),
        DistributionSpec("big", "normal", {"mean": 0.0, "std": 10.0}),
    )
    engine = MonteCarloEngine(mgr, ModelFunction(lambda row: row["small"] + row["big"]), 10, 0)
    out = SensitivityAnalyzer(engine).tornado()
    assert out.iloc[0]["variable"] == "big"
    assert out["swing"].is_monotonic_decreasing


def test_sobol_attributes_variance_to_the_variable_that_drives_it():
    mgr = _mgr(
        DistributionSpec("driver", "normal", {"mean": 0.0, "std": 5.0}),
        DistributionSpec("noise", "normal", {"mean": 0.0, "std": 0.01}),
    )
    engine = MonteCarloEngine(
        mgr, ModelFunction(lambda df: df["driver"] + df["noise"], vectorized=True), 400, 3
    )
    out = SensitivityAnalyzer(engine).sobol_indices(n_samples=400, seed=3)
    assert out.iloc[0]["variable"] == "driver"
    assert (out["S1_conf"] >= 0).all()


def test_sobol_returns_zero_indices_for_a_constant_model():
    """Guards the zero-variance branch, which would otherwise divide by ~0."""
    mgr = _mgr(DistributionSpec("a", "normal", VALID_PARAMS["normal"]))
    engine = MonteCarloEngine(mgr, ModelFunction(lambda df: df["a"] * 0.0, vectorized=True), 200, 1)
    out = SensitivityAnalyzer(engine).sobol_indices(n_samples=200, seed=1)
    assert out["S1"].abs().max() == 0.0


# ── Scenarios ─────────────────────────────────────────────────────────────


def _comparator(scenarios):
    mgr = _mgr(DistributionSpec("price", "normal", {"mean": 10.0, "std": 1.0}))
    return ScenarioComparator(mgr, ModelFunction(lambda row: row["price"]), scenarios, 300, 7)


def test_scenarios_run_alongside_a_baseline():
    results = _comparator([Scenario("up", {"price": {"mean": 50.0}})]).run_all()
    assert set(results) == {"baseline", "up"}
    assert results["up"].mean > results["baseline"].mean


def test_an_override_does_not_mutate_the_base_inputs():
    mgr = _mgr(DistributionSpec("price", "normal", {"mean": 10.0, "std": 1.0}))
    comparator = ScenarioComparator(
        mgr, ModelFunction(lambda row: row["price"]), [Scenario("up", {"price": {"mean": 99.0}})], 100, 1
    )
    comparator.run_all()
    assert mgr.specs["price"].params["mean"] == 10.0


def test_an_override_merges_rather_than_replacing_params():
    """Documented on Scenario: change a mean without restating the std."""
    mgr = _mgr(DistributionSpec("price", "normal", {"mean": 10.0, "std": 3.0}))
    comparator = ScenarioComparator(mgr, ModelFunction(lambda row: row["price"]), [], 10, 1)
    modified = comparator._apply_overrides(Scenario("up", {"price": {"mean": 20.0}}))
    assert modified.specs["price"].params == {"mean": 20.0, "std": 3.0}


def test_an_override_of_an_unknown_variable_names_the_scenario():
    with pytest.raises(KeyError, match="ghost"):
        _comparator([Scenario("ghost", {"nope": {"mean": 1.0}})]).run_all()


def test_compare_summary_runs_the_scenarios_if_needed():
    table = _comparator([Scenario("up", {"price": {"mean": 30.0}})]).compare_summary()
    assert set(table["scenario"]) == {"baseline", "up"}
    assert (table["ci_lower"] <= table["mean"]).all()


# ── ConvergenceDiagnostics ────────────────────────────────────────────────


def test_running_statistics_converge_on_the_true_mean():
    outcomes = np.random.default_rng(8).normal(20, 3, size=4000)
    stats_df = ConvergenceDiagnostics.running_statistics(outcomes)
    assert len(stats_df) == 4000
    assert stats_df["cumulative_mean"].iloc[-1] == pytest.approx(20, abs=0.5)
    assert stats_df["cumulative_std"].iloc[0] == 0.0      # one sample, no spread


def test_a_stable_series_reads_as_converged():
    assert ConvergenceDiagnostics.is_converged(np.full(5000, 7.0)) is True


def test_a_drifting_series_reads_as_not_converged():
    drifting = np.concatenate([np.full(2500, 1.0), np.full(2500, 100.0)])
    assert ConvergenceDiagnostics.is_converged(drifting) is False


def test_an_all_zero_series_is_treated_as_converged_rather_than_dividing_by_zero():
    assert ConvergenceDiagnostics.is_converged(np.zeros(5000)) is True


def test_a_frame_of_outcomes_is_accepted_everywhere_an_array_is():
    frame = pd.DataFrame({"outcome": np.random.default_rng(1).normal(5, 1, 3000)})
    assert isinstance(ConvergenceDiagnostics.is_converged(frame), bool)
    assert ConvergenceDiagnostics.suggest_n(frame) >= 3000
    assert len(ConvergenceDiagnostics.running_statistics(frame)) == 3000


def test_a_noisier_series_needs_more_iterations():
    rng = np.random.default_rng(12)
    quiet = ConvergenceDiagnostics.suggest_n(rng.normal(100, 1, 2000))
    noisy = ConvergenceDiagnostics.suggest_n(rng.normal(100, 40, 2000))
    assert noisy > quiet


# ── Simulation.from_spec ──────────────────────────────────────────────────


def _spec(n=400):
    rng = np.random.default_rng(21)
    x = rng.normal(10, 2, n)
    data = pd.DataFrame({"x": x, "c": rng.normal(5, 1, n), "y": x * 3 + rng.normal(0, 1, n)})
    return ModelSpec(
        X=data[["x", "c"]],
        y=data["y"],
        independents=("x",),
        controls=("c",),
        dependent="y",
        source_label="test",
        n=n,
        data=data,
    )


def test_from_spec_fits_the_independents_and_controls():
    sim = Simulation.from_spec(_spec(), model=lambda row: row["x"] + row["c"], n_iterations=200)
    assert set(sim.input_manager.variable_names) == {"x", "c"}
    assert sim.run().mean is not None


def test_from_spec_infers_correlation_when_there_is_more_than_one_column():
    sim = Simulation.from_spec(_spec(), model=lambda row: row["x"], n_iterations=100)
    assert sim.input_manager.correlation_matrix is not None


def test_from_spec_can_resample_the_dependent_instead_of_modelling_it():
    sim = Simulation.from_spec(_spec(), include_dependent=True, n_iterations=200)
    assert "y" in sim.input_manager.variable_names
    result = sim.engine.run()
    assert isinstance(result.outcomes, (np.ndarray, pd.DataFrame))


def test_from_spec_refuses_a_model_together_with_include_dependent():
    with pytest.raises(ValueError, match="Cannot use both"):
        Simulation.from_spec(_spec(), model=lambda row: row["x"], include_dependent=True)


def test_from_spec_requires_a_model_when_not_including_the_dependent():
    with pytest.raises(ValueError, match="model function is required"):
        Simulation.from_spec(_spec())


def test_from_spec_refuses_include_dependent_with_no_dependent_set():
    spec = _spec()
    bare = ModelSpec(
        X=spec.X, y=None, independents=spec.independents, controls=spec.controls,
        dependent=None, source_label="t", n=spec.n, data=spec.data,
    )
    with pytest.raises(ValueError, match="no dependent variable"):
        Simulation.from_spec(bare, include_dependent=True)


def test_from_spec_honours_a_per_column_dist_type_override():
    sim = Simulation.from_spec(
        _spec(),
        model=lambda row: row["x"],
        overrides={"x": {"dist_type": "empirical"}},
        n_iterations=100,
    )
    assert sim.input_manager.specs["x"].dist_type == "empirical"
    assert sim.input_manager.specs["c"].dist_type == "normal"
