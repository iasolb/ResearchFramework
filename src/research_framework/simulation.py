"""
Ian Solberg
March '26
Simulation Extension of Research Framework
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from copy import deepcopy

# ---------------------------------------------------------------------------
# 1. INPUT SPECIFICATION
# ---------------------------------------------------------------------------

_DISTRIBUTION_REGISTRY: dict[str, dict] = {
    "normal": {
        "draw": lambda rng, n, p: rng.normal(loc=p["mean"], scale=p["std"], size=n),
        "param_keys": ("mean", "std"),
        "scipy": stats.norm,
        "scipy_fit_map": lambda loc, scale: {"mean": loc, "std": scale},
        "scipy_ppf_args": lambda p: (p["mean"], p["std"]),
    },
    "uniform": {
        "draw": lambda rng, n, p: rng.uniform(low=p["low"], high=p["high"], size=n),
        "param_keys": ("low", "high"),
        "scipy": stats.uniform,
        "scipy_fit_map": lambda loc, scale: {"low": loc, "high": loc + scale},
        "scipy_ppf_args": lambda p: (p["low"], p["high"] - p["low"]),
    },
    "lognormal": {
        "draw": lambda rng, n, p: rng.lognormal(mean=p["mean"], sigma=p["sigma"], size=n),
        "param_keys": ("mean", "sigma"),
        "scipy": stats.lognorm,
        "scipy_fit_map": lambda s, loc, scale: {"mean": np.log(scale), "sigma": s},
        "scipy_ppf_args": lambda p: (p["sigma"], 0, np.exp(p["mean"])),
    },
    "beta": {
        "draw": lambda rng, n, p: rng.beta(a=p["a"], b=p["b"], size=n),
        "param_keys": ("a", "b"),
        "scipy": stats.beta,
        "scipy_fit_map": lambda a, b, loc, scale: {"a": a, "b": b},
        "scipy_ppf_args": lambda p: (p["a"], p["b"]),
    },
    "triangular": {
        "draw": lambda rng, n, p: rng.triangular(left=p["left"], mode=p["mode"], right=p["right"], size=n),
        "param_keys": ("left", "mode", "right"),
        "scipy": stats.triang,
        "scipy_fit_map": lambda c, loc, scale: {"left": loc, "mode": loc + c * scale, "right": loc + scale},
        "scipy_ppf_args": lambda p: ((p["mode"] - p["left"]) / (p["right"] - p["left"]), p["left"], p["right"] - p["left"]),
    },
    "exponential": {
        "draw": lambda rng, n, p: rng.exponential(scale=p["scale"], size=n),
        "param_keys": ("scale",),
        "scipy": stats.expon,
        "scipy_fit_map": lambda loc, scale: {"scale": scale},
        "scipy_ppf_args": lambda p: (0, p["scale"]),
    },
}


@dataclass
class DistributionSpec:
    name: str
    dist_type: str
    params: dict = field(default_factory=dict)
    empirical_data: Optional[np.ndarray] = None

    def __post_init__(self):
        self.dist_type = self.dist_type.lower().strip()
        self._validate()

    def _validate(self) -> None:
        if self.dist_type == "empirical":
            if self.empirical_data is None or len(self.empirical_data) == 0:
                raise ValueError(f"Variable '{self.name}': empirical dist_type requires non-empty empirical_data array.")
            return
        if self.dist_type not in _DISTRIBUTION_REGISTRY:
            raise ValueError(f"Variable '{self.name}': unknown dist_type '{self.dist_type}'. Supported: {list(_DISTRIBUTION_REGISTRY.keys()) + ['empirical']}")
        required = set(_DISTRIBUTION_REGISTRY[self.dist_type]["param_keys"])
        provided = set(self.params.keys())
        missing = required - provided
        if missing:
            raise ValueError(f"Variable '{self.name}' ({self.dist_type}): missing required params: {missing}. Expected: {required}")


class InputManager:
    def __init__(self):
        self.specs: dict[str, DistributionSpec] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self._variable_order: list[str] = []

    @property
    def variable_names(self) -> list[str]:
        return list(self._variable_order)

    @property
    def n_variables(self) -> int:
        return len(self._variable_order)

    def add_variable(self, spec: DistributionSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"Variable '{spec.name}' is already registered.")
        self.specs[spec.name] = spec
        self._variable_order.append(spec.name)

    def add_variables(self, specs: list[DistributionSpec]) -> None:
        for spec in specs:
            self.add_variable(spec)

    def remove_variable(self, name: str) -> None:
        if name not in self.specs:
            raise KeyError(f"Variable '{name}' not found.")
        del self.specs[name]
        self._variable_order.remove(name)
        self.correlation_matrix = None

    def fit_from_data(self, df: pd.DataFrame, columns: list[str], dist_type: str = "normal") -> list[DistributionSpec]:
        created = []
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            values = df[col].dropna().values
            if dist_type == "empirical":
                spec = DistributionSpec(name=col, dist_type="empirical", empirical_data=np.array(values))
            else:
                if dist_type not in _DISTRIBUTION_REGISTRY:
                    raise ValueError(f"Unknown dist_type '{dist_type}'.")
                registry = _DISTRIBUTION_REGISTRY[dist_type]
                scipy_dist = registry["scipy"]
                fit_result = scipy_dist.fit(values)
                params = registry["scipy_fit_map"](*fit_result)
                spec = DistributionSpec(name=col, dist_type=dist_type, params=params)
            self.add_variable(spec)
            created.append(spec)
        return created

    def set_correlation_matrix(self, matrix: np.ndarray) -> None:
        n = self.n_variables
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (n, n):
            raise ValueError(f"Correlation matrix shape {matrix.shape} does not match number of variables ({n}).")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Correlation matrix must be symmetric.")
        if not np.allclose(np.diag(matrix), 1.0):
            raise ValueError("Correlation matrix diagonal must be all 1s.")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues < -1e-8):
            raise ValueError(f"Correlation matrix is not positive semi-definite. Smallest eigenvalue: {eigenvalues.min():.6e}")
        self.correlation_matrix = matrix

    def infer_correlation_from_data(self, df: pd.DataFrame) -> np.ndarray:
        missing = [v for v in self.variable_names if v not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")
        matrix = df[self.variable_names].corr().values
        self.set_correlation_matrix(matrix)
        return matrix

    def _draw_independent(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        columns = {}
        for name in self._variable_order:
            spec = self.specs[name]
            if spec.dist_type == "empirical":
                if spec.empirical_data is not None:
                    columns[name] = rng.choice(spec.empirical_data, size=n, replace=True)
            else:
                draw_fn = _DISTRIBUTION_REGISTRY[spec.dist_type]["draw"]
                columns[name] = draw_fn(rng, n, spec.params)
        return pd.DataFrame(columns)

    def _draw_correlated(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        k = self.n_variables
        z = rng.standard_normal(size=(n, k))
        if self.correlation_matrix is not None:
            L = np.linalg.cholesky(self.correlation_matrix)
        else:
            L = None
        assert L is not None, "Cholesky Failed"
        correlated_z = z @ L.T
        u = stats.norm.cdf(correlated_z)
        columns = {}
        for i, name in enumerate(self._variable_order):
            spec = self.specs[name]
            if spec.dist_type == "empirical":
                assert spec.empirical_data is not None, "Missing Empirical Data."
                sorted_data = np.sort(spec.empirical_data)
                indices = np.clip((u[:, i] * len(sorted_data)).astype(int), 0, len(sorted_data) - 1)
                columns[name] = sorted_data[indices]
            else:
                registry = _DISTRIBUTION_REGISTRY[spec.dist_type]
                scipy_dist = registry["scipy"]
                ppf_args = registry["scipy_ppf_args"](spec.params)
                columns[name] = scipy_dist.ppf(u[:, i], *ppf_args)
        return pd.DataFrame(columns)

    def draw(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        if self.n_variables == 0:
            raise RuntimeError("No variables registered. Use add_variable() first.")
        rng = np.random.default_rng(seed)
        if self.correlation_matrix is not None:
            return self._draw_correlated(n, rng)
        else:
            return self._draw_independent(n, rng)


class ModelFunction:
    def __init__(self, func: Callable, vectorized: bool = False):
        self.func = func
        self.vectorized = vectorized

    def run(self, draws: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        if self.vectorized:
            result = self.func(draws)
            if isinstance(result, pd.DataFrame):
                return result
            return np.asarray(result)
        raw = draws.apply(self.func, axis=1)
        if isinstance(raw.iloc[0], dict):
            return pd.DataFrame(raw.tolist(), index=draws.index)
        return raw.values.astype(float)


@dataclass
class SimulationResult:
    outcomes: np.ndarray = field(default_factory=lambda: np.array([]))
    draws: Optional[pd.DataFrame] = None
    n_iterations: int = 0
    seed: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    percentiles: Optional[dict[int, float]] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    def summarize(self, confidence: float = 0.95) -> dict:
        if len(self.outcomes) == 0:
            raise RuntimeError("No outcomes to summarize.")
        if isinstance(self.outcomes, pd.DataFrame):
            values = np.array(self.outcomes.iloc[:, 0].values)
        else:
            values = np.array(self.outcomes)
        self.mean = float(np.mean(values))
        self.median = float(np.median(values))
        self.std = float(np.std(values, ddof=1))
        pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(values, pct_keys)
        self.percentiles = dict(zip(pct_keys, [float(v) for v in pct_values]))
        alpha = 1 - confidence
        self.ci_lower = float(np.percentile(values, 100 * alpha / 2))
        self.ci_upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
        return {"mean": self.mean, "median": self.median, "std": self.std, "min": float(np.min(values)), "max": float(np.max(values)), "percentiles": self.percentiles, "ci_lower": self.ci_lower, "ci_upper": self.ci_upper, "confidence": confidence, "n_iterations": self.n_iterations}

    def to_dataframe(self) -> pd.DataFrame:
        if self.draws is None:
            if isinstance(self.outcomes, pd.DataFrame):
                return self.outcomes.copy()
            return pd.DataFrame({"outcome": self.outcomes})
        df = self.draws.copy()
        if isinstance(self.outcomes, pd.DataFrame):
            for col in self.outcomes.columns:
                df[col] = pd.DataFrame(self.outcomes[col]).values
        else:
            df["outcome"] = self.outcomes
        return df

    def __repr__(self) -> str:
        status = "summarized" if self.mean is not None else "raw"
        ci = ""
        if self.ci_lower is not None:
            ci = f", 95% CI=[{self.ci_lower:,.2f}, {self.ci_upper:,.2f}]"
        mean_str = f", mean={self.mean:,.4f}" if self.mean is not None else ""
        return f"SimulationResult(n={self.n_iterations}, status={status}{mean_str}{ci})"


class MonteCarloEngine:
    def __init__(self, inputs: InputManager, model: ModelFunction, n_iterations: int = 10_000, seed: Optional[int] = None):
        self.inputs = inputs
        self.model = model
        self.n_iterations = n_iterations
        self.seed = seed

    def run(self, store_draws: bool = True) -> SimulationResult:
        draws = self.inputs.draw(self.n_iterations, seed=self.seed)
        outcomes = np.array(self.model.run(draws))
        return SimulationResult(outcomes=outcomes, draws=draws if store_draws else None, n_iterations=self.n_iterations, seed=self.seed)

    def run_convergence(self, checkpoints: list[int] | None = None) -> list[SimulationResult]:
        if checkpoints is None:
            candidates = [100, 500, 1_000, 2_500, 5_000, self.n_iterations]
            checkpoints = [c for c in candidates if c <= self.n_iterations]
            if self.n_iterations not in checkpoints:
                checkpoints.append(self.n_iterations)
        checkpoints = sorted(set(min(c, self.n_iterations) for c in checkpoints))
        max_n = checkpoints[-1]
        original_n = self.n_iterations
        self.n_iterations = max_n
        full_result = self.run(store_draws=True)
        self.n_iterations = original_n
        snapshots = []
        for cp in checkpoints:
            if isinstance(full_result.outcomes, pd.DataFrame):
                sliced_outcomes = full_result.outcomes.iloc[:cp]
            else:
                sliced_outcomes = full_result.outcomes[:cp]
            sliced_draws = full_result.draws.iloc[:cp] if full_result.draws is not None else None
            snap = SimulationResult(outcomes=sliced_outcomes, draws=sliced_draws, n_iterations=cp, seed=self.seed)
            snap.summarize()
            snapshots.append(snap)
        return snapshots


class SensitivityAnalyzer:
    def __init__(self, engine: MonteCarloEngine):
        self.engine = engine

    def _get_baseline_row(self) -> pd.Series:
        values = {}
        for name, spec in self.engine.inputs.specs.items():
            if spec.dist_type == "empirical":
                values[name] = float(np.median(np.array(spec.empirical_data)))
            elif spec.dist_type == "normal":
                values[name] = spec.params["mean"]
            elif spec.dist_type == "lognormal":
                values[name] = np.exp(spec.params["mean"])
            elif spec.dist_type == "uniform":
                values[name] = (spec.params["low"] + spec.params["high"]) / 2
            elif spec.dist_type == "beta":
                a, b = spec.params["a"], spec.params["b"]
                values[name] = a / (a + b)
            elif spec.dist_type == "triangular":
                values[name] = spec.params["mode"]
            elif spec.dist_type == "exponential":
                values[name] = spec.params["scale"]
            else:
                values[name] = 0.0
        return pd.Series(values)

    def _get_variable_range(self, spec: DistributionSpec, n_steps: int, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
        if spec.dist_type == "empirical":
            return np.linspace(np.percentile(np.array(spec.empirical_data), low_pct), np.percentile(np.array(spec.empirical_data), high_pct), n_steps)
        registry = _DISTRIBUTION_REGISTRY[spec.dist_type]
        scipy_dist = registry["scipy"]
        ppf_args = registry["scipy_ppf_args"](spec.params)
        lo = scipy_dist.ppf(low_pct / 100, *ppf_args)
        hi = scipy_dist.ppf(high_pct / 100, *ppf_args)
        return np.linspace(lo, hi, n_steps)

    def one_at_a_time(self, variable: str, values: np.ndarray | None = None, n_steps: int = 20) -> pd.DataFrame:
        if variable not in self.engine.inputs.specs:
            raise KeyError(f"Variable '{variable}' not registered.")
        spec = self.engine.inputs.specs[variable]
        baseline = self._get_baseline_row()
        if values is None:
            values = self._get_variable_range(spec, n_steps)
        outcomes = []
        for val in values:
            row = baseline.copy()
            row[variable] = val
            result = self.engine.model.func(row)
            if isinstance(result, dict):
                result = list(result.values())[0]
            outcomes.append(float(result))
        return pd.DataFrame({"variable_value": values, "outcome": outcomes})

    def tornado(self, low_pct: float = 10, high_pct: float = 90) -> pd.DataFrame:
        baseline = self._get_baseline_row()
        rows = []
        for name, spec in self.engine.inputs.specs.items():
            rng = self._get_variable_range(spec, 2, low_pct, high_pct)
            lo_val, hi_val = rng[0], rng[-1]
            row_lo = baseline.copy()
            row_lo[name] = lo_val
            out_lo = self.engine.model.func(row_lo)
            if isinstance(out_lo, dict):
                out_lo = list(out_lo.values())[0]
            row_hi = baseline.copy()
            row_hi[name] = hi_val
            out_hi = self.engine.model.func(row_hi)
            if isinstance(out_hi, dict):
                out_hi = list(out_hi.values())[0]
            rows.append({"variable": name, "low_value": lo_val, "high_value": hi_val, "low_outcome": float(out_lo), "high_outcome": float(out_hi), "swing": abs(float(out_hi) - float(out_lo))})
        return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)

    def sobol_indices(self, n_samples: int = 10_000, seed: Optional[int] = None) -> pd.DataFrame:
        mgr = self.engine.inputs
        k = mgr.n_variables
        rng = np.random.default_rng(seed or self.engine.seed)
        seed_a = rng.integers(0, 2**31)
        seed_b = rng.integers(0, 2**31)
        A = mgr.draw(n_samples, seed=int(seed_a))
        B = mgr.draw(n_samples, seed=int(seed_b))
        f_A = self.engine.model.run(A)
        f_B = self.engine.model.run(B)
        if isinstance(f_A, pd.DataFrame):
            f_A = np.array(f_A.iloc[:, 0].values)
        if isinstance(f_B, pd.DataFrame):
            f_B = np.array(f_B.iloc[:, 0].values)
        S1_values = []
        names = mgr.variable_names
        for i, name in enumerate(names):
            AB_i = A.copy()
            AB_i[name] = B[name].values
            f_AB_i = self.engine.model.run(AB_i)
            if isinstance(f_AB_i, pd.DataFrame):
                f_AB_i = f_AB_i.iloc[:, 0].values
            total_var = np.var(np.concatenate([f_A, f_B]), ddof=1)
            if total_var < 1e-15:
                S1_values.append(0.0)
                continue
            S1 = np.mean(f_B * (f_AB_i - f_A)) / total_var
            S1_values.append(float(S1))
        n_boot = 100
        S1_boot = np.zeros((n_boot, k))
        for b in range(n_boot):
            idx = rng.integers(0, n_samples, size=n_samples)
            f_A_b = f_A[idx]
            f_B_b = f_B[idx]
            total_var_b = np.var(np.concatenate([f_A_b, f_B_b]), ddof=1)
            if total_var_b < 1e-15:
                continue
            for i, name in enumerate(names):
                AB_i = A.iloc[idx].copy()
                AB_i[name] = B[name].values[idx]
                f_AB_i_b = self.engine.model.run(AB_i)
                if isinstance(f_AB_i_b, pd.DataFrame):
                    f_AB_i_b = f_AB_i_b.iloc[:, 0].values
                S1_boot[b, i] = np.mean(f_B_b * (f_AB_i_b - f_A_b)) / total_var_b
        S1_conf = 1.96 * np.std(S1_boot, axis=0, ddof=1)
        return pd.DataFrame({"variable": names, "S1": S1_values, "S1_conf": S1_conf.tolist()}).sort_values("S1", ascending=False).reset_index(drop=True)


@dataclass
class Scenario:
    name: str
    overrides: dict[str, dict] = field(default_factory=dict)


class ScenarioComparator:
    def __init__(self, base_inputs: InputManager, model: ModelFunction, scenarios: list[Scenario], n_iterations: int = 10_000, seed: Optional[int] = None):
        self.base_inputs = base_inputs
        self.model = model
        self.scenarios = scenarios
        self.n_iterations = n_iterations
        self.seed = seed
        self._results: dict[str, SimulationResult] = {}

    def _apply_overrides(self, scenario: Scenario) -> InputManager:
        modified = deepcopy(self.base_inputs)
        for var_name, param_overrides in scenario.overrides.items():
            if var_name not in modified.specs:
                raise KeyError(f"Scenario '{scenario.name}' overrides variable '{var_name}' which is not registered.")
            spec = modified.specs[var_name]
            spec.params.update(param_overrides)
        return modified

    def run_all(self) -> dict[str, SimulationResult]:
        engine = MonteCarloEngine(self.base_inputs, self.model, self.n_iterations, self.seed)
        baseline_result = engine.run()
        baseline_result.summarize()
        self._results["baseline"] = baseline_result
        for scenario in self.scenarios:
            modified_inputs = self._apply_overrides(scenario)
            engine = MonteCarloEngine(modified_inputs, self.model, self.n_iterations, self.seed)
            result = engine.run()
            result.summarize()
            self._results[scenario.name] = result
        return self._results

    def compare_summary(self) -> pd.DataFrame:
        if not self._results:
            self.run_all()
        rows = []
        for name, result in self._results.items():
            if result.mean is None:
                result.summarize()
            rows.append({"scenario": name, "mean": result.mean, "median": result.median, "std": result.std, "ci_lower": result.ci_lower, "ci_upper": result.ci_upper, "min": float(np.min(result.outcomes if not isinstance(result.outcomes, pd.DataFrame) else result.outcomes.iloc[:, 0])), "max": float(np.max(result.outcomes if not isinstance(result.outcomes, pd.DataFrame) else result.outcomes.iloc[:, 0]))})
        return pd.DataFrame(rows)


class ConvergenceDiagnostics:
    @staticmethod
    def running_statistics(outcomes: np.ndarray) -> pd.DataFrame:
        if isinstance(outcomes, pd.DataFrame):
            outcomes = np.array(outcomes.iloc[:, 0].values)
        n = len(outcomes)
        cum_sum = np.cumsum(outcomes)
        iterations = np.arange(1, n + 1)
        cum_mean = cum_sum / iterations
        cum_sq_sum = np.cumsum(outcomes**2)
        with np.errstate(invalid="ignore"):
            cum_var = (cum_sq_sum / iterations) - cum_mean**2
            cum_var = np.where(iterations > 1, cum_var * iterations / (iterations - 1), 0.0)
        cum_std = np.sqrt(np.maximum(cum_var, 0.0))
        return pd.DataFrame({"iteration": iterations, "cumulative_mean": cum_mean, "cumulative_std": cum_std})

    @staticmethod
    def is_converged(outcomes: np.ndarray, window: int = 1000, tolerance: float = 0.01) -> bool:
        if isinstance(outcomes, pd.DataFrame):
            outcomes = np.array(outcomes.iloc[:, 0].values)
        if len(outcomes) < window * 2:
            return False
        overall_mean = np.mean(outcomes)
        if abs(overall_mean) < 1e-15:
            return True
        trailing_mean = np.mean(outcomes[-window:])
        relative_diff = abs(trailing_mean - overall_mean) / abs(overall_mean)
        return bool(relative_diff < tolerance)

    @staticmethod
    def suggest_n(outcomes: np.ndarray, target_tolerance: float = 0.005, confidence: float = 0.95) -> int:
        if isinstance(outcomes, pd.DataFrame):
            outcomes = np.array(outcomes.iloc[:, 0].values)
        mean = np.mean(outcomes)
        std = np.std(outcomes, ddof=1)
        if abs(mean) < 1e-15:
            return len(outcomes)
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        required_n = int(np.ceil((z * std / (target_tolerance * abs(mean))) ** 2))
        return max(required_n, len(outcomes))


class Simulation:
    def __init__(self, variables: list[DistributionSpec], model: Callable, *, vectorized: bool = False, n_iterations: int = 10_000, seed: Optional[int] = None, correlation_matrix: Optional[np.ndarray] = None):
        self.input_manager = InputManager()
        self.model_fn = ModelFunction(model, vectorized=vectorized)
        self.n_iterations = n_iterations
        self.seed = seed
        self.input_manager.add_variables(variables)
        if correlation_matrix is not None:
            self.input_manager.set_correlation_matrix(correlation_matrix)
        self.engine = MonteCarloEngine(self.input_manager, self.model_fn, n_iterations, seed)
        self.sensitivity = SensitivityAnalyzer(self.engine)
        self.convergence = ConvergenceDiagnostics

    def run(self) -> SimulationResult:
        result = self.engine.run()
        result.summarize()
        return result

    def compare_scenarios(self, scenarios: list[Scenario]) -> dict[str, SimulationResult]:
        comparator = ScenarioComparator(base_inputs=self.input_manager, model=self.model_fn, scenarios=scenarios, n_iterations=self.n_iterations, seed=self.seed)
        return comparator.run_all()

    def compare_scenarios_summary(self, scenarios: list[Scenario]) -> pd.DataFrame:
        comparator = ScenarioComparator(base_inputs=self.input_manager, model=self.model_fn, scenarios=scenarios, n_iterations=self.n_iterations, seed=self.seed)
        comparator.run_all()
        return comparator.compare_summary()

    def check_convergence(self, result: SimulationResult) -> dict:
        values = result.outcomes
        if isinstance(values, pd.DataFrame):
            values = np.array(values.iloc[:, 0].values)
        is_conv = ConvergenceDiagnostics.is_converged(values)
        suggested = ConvergenceDiagnostics.suggest_n(values)
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        relative_se = se / abs(mean) if abs(mean) > 1e-15 else 0.0
        return {"is_converged": is_conv, "suggested_n": suggested, "current_n": len(values), "relative_se": relative_se}

    @classmethod
    def from_spec(cls, spec, model: Optional[Callable] = None, *, dist_type: str = "normal", overrides: Optional[dict[str, dict]] = None, include_dependent: bool = False, n_iterations: int = 10_000, seed: Optional[int] = None, vectorized: bool = False) -> "Simulation":
        if include_dependent and model is not None:
            raise ValueError("Cannot use both include_dependent=True and a model function.")
        if not include_dependent and model is None:
            raise ValueError("A model function is required when include_dependent=False.")
        if include_dependent and spec.dependent is None:
            raise ValueError("include_dependent=True but the ModelSpec has no dependent variable.")
        overrides = overrides or {}
        columns_to_fit = list(spec.independents) + list(spec.controls)
        if include_dependent:
            columns_to_fit = [spec.dependent] + columns_to_fit
        mgr = InputManager()
        for col in columns_to_fit:
            col_dist = overrides.get(col, {}).get("dist_type", dist_type)
            mgr.fit_from_data(spec.data, [col], dist_type=col_dist)
        if len(columns_to_fit) > 1:
            mgr.infer_correlation_from_data(spec.data)
        if include_dependent:
            identity = ModelFunction(lambda df: df, vectorized=True)
            instance = cls.__new__(cls)
            instance.input_manager = mgr
            instance.model_fn = identity
            instance.n_iterations = n_iterations
            instance.seed = seed
            instance.engine = MonteCarloEngine(mgr, identity, n_iterations, seed)
            instance.sensitivity = SensitivityAnalyzer(instance.engine)
            instance.convergence = ConvergenceDiagnostics
        else:
            assert model is not None, "No Model Passed."
            model_fn = ModelFunction(model, vectorized=vectorized)
            instance = cls.__new__(cls)
            instance.input_manager = mgr
            instance.model_fn = model_fn
            instance.n_iterations = n_iterations
            instance.seed = seed
            instance.engine = MonteCarloEngine(mgr, model_fn, n_iterations, seed)
            instance.sensitivity = SensitivityAnalyzer(instance.engine)
            instance.convergence = ConvergenceDiagnostics
        return instance
