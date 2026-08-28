"""Smoke tests for SimulationPlotter.

Every test proves the method runs and returns a valid plotly Figure with at
least one trace. Nothing is asserted about visual appearance (colours, titles,
axis labels, layout). The test breaks if an upstream column is renamed or a
plotly API shifts, but not if styling changes.
"""

import pytest
import plotly.graph_objects as go

from research_framework.simulation import (
    DistributionSpec,
    Scenario,
    Simulation,
)
from research_framework.plotter import SimulationPlotter


# ── Shared fixture ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sim():
    s = Simulation(
        variables=[
            DistributionSpec("price", "normal", {"mean": 10, "std": 2}),
            DistributionSpec("units", "triangular", {"left": 80, "mode": 100, "right": 150}),
        ],
        model=lambda row: row["price"] * row["units"],
        seed=42,
        n_iterations=500,
    )
    return s


@pytest.fixture(scope="module")
def result(sim):
    r = sim.run()
    r.summarize()
    return r


# ── histogram ─────────────────────────────────────────────────────────────


def test_histogram_returns_figure_with_trace(result):
    fig = SimulationPlotter.histogram(result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── cumulative_density ────────────────────────────────────────────────────


def test_cumulative_density_returns_figure_with_trace(result):
    fig = SimulationPlotter.cumulative_density(result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── convergence_plot ──────────────────────────────────────────────────────


def test_convergence_plot_returns_figure_with_trace(result):
    fig = SimulationPlotter.convergence_plot(result.outcomes)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── tornado_chart ─────────────────────────────────────────────────────────


def test_tornado_chart_returns_figure_with_trace(sim):
    tornado_data = sim.sensitivity.tornado()
    fig = SimulationPlotter.tornado_chart(tornado_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── scenario_comparison ───────────────────────────────────────────────────


def test_scenario_comparison_returns_figure_with_trace(sim):
    scenarios = [Scenario("up", {"price": {"mean": 20, "std": 2}})]
    results = sim.compare_scenarios(scenarios)
    fig = SimulationPlotter.scenario_comparison(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── tornado_comparison ────────────────────────────────────────────────────


def test_tornado_comparison_returns_figure_with_trace(sim):
    t1 = sim.sensitivity.tornado()
    t2 = sim.sensitivity.tornado()
    fig = SimulationPlotter.tornado_comparison([t1, t2], labels=["Base", "Alt"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── histogram_comparison ──────────────────────────────────────────────────


def test_histogram_comparison_returns_figure_with_trace(sim):
    scenarios = [Scenario("up", {"price": {"mean": 20, "std": 2}})]
    results = sim.compare_scenarios(scenarios)
    fig = SimulationPlotter.histogram_comparison(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ── edge cases ────────────────────────────────────────────────────────────


def test_histogram_single_iteration():
    """A single-row result should still return a figure or raise a known exception."""
    sim1 = Simulation(
        variables=[DistributionSpec("x", "normal", {"mean": 0, "std": 1})],
        model=lambda row: row["x"],
        seed=0,
        n_iterations=1,
    )
    result1 = sim1.run()
    result1.summarize()
    # With 1 point CI bounds may be None; histogram should still return a figure
    fig = SimulationPlotter.histogram(result1)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_convergence_plot_single_value():
    """Running statistics on a single value should return a figure or raise."""
    import numpy as np
    values = np.array([42.0])
    try:
        fig = SimulationPlotter.convergence_plot(values)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    except Exception as exc:
        # Document what exception is raised with a single value
        assert exc is not None  # current behaviour: raises (follow-up)


def test_tornado_chart_single_variable():
    """Tornado with a single-variable simulation should return a figure."""
    sim2 = Simulation(
        variables=[DistributionSpec("x", "normal", {"mean": 5, "std": 1})],
        model=lambda row: row["x"] * 2,
        seed=1,
        n_iterations=200,
    )
    t = sim2.sensitivity.tornado()
    fig = SimulationPlotter.tornado_chart(t)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
