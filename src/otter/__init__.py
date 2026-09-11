"""A readable pandas-based framework for research workflows: dataset
handling with tracked model specifications, reusable column transforms,
and a Monte Carlo simulation module."""

from .rh import ModelSpec, ResearchHandler, ResearchHandlerLoadFailedError
from .simulation import (
    ConvergenceDiagnostics,
    DistributionSpec,
    InputManager,
    ModelFunction,
    MonteCarloEngine,
    Scenario,
    SensitivityAnalyzer,
    Simulation,
    SimulationResult,
)
from .transforms import (
    demean_by_group,
    interaction,
    log1p_transform,
    log_transform,
    mean_center,
    min_max_scale,
    rank_transform,
    row_mean,
    row_sum,
    safe_ratio,
    square,
    winsorize,
    z_score,
)
from .plotter import SimulationPlotter

__all__ = [
    "ConvergenceDiagnostics",
    "DistributionSpec",
    "InputManager",
    "ModelFunction",
    "ModelSpec",
    "MonteCarloEngine",
    "ResearchHandler",
    "ResearchHandlerLoadFailedError",
    "Scenario",
    "SensitivityAnalyzer",
    "Simulation",
    "SimulationPlotter",
    "SimulationResult",
    "demean_by_group",
    "interaction",
    "log1p_transform",
    "log_transform",
    "mean_center",
    "min_max_scale",
    "rank_transform",
    "row_mean",
    "row_sum",
    "safe_ratio",
    "square",
    "winsorize",
    "z_score",
]
