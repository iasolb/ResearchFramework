import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Optional, Callable, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """
    Frozen snapshot of a ResearchHandler's variable specification.
    Produced by rh.get_spec(). Can be passed to Simulation.from_spec()
    to build a data-driven Monte Carlo simulation.

    Attributes:
        X:             design matrix (independents + controls)
        y:             dependent variable (None if not set)
        independents:  tuple of independent variable column names
        controls:      tuple of control variable column names
        dependent:     dependent variable column name (None if not set)
        source_label:  "full" or "subset" — which dataset the variables came from
        n:             number of observations
        data:          copy of the source DataFrame (for distribution fitting)
    """

    X: pd.DataFrame
    y: Optional[pd.Series]
    independents: tuple
    controls: tuple
    dependent: Optional[str]
    source_label: str
    n: int
    data: pd.DataFrame

    @property
    def columns(self) -> tuple:
        """All variable column names (independents + controls)."""
        return self.independents + self.controls

    @property
    def all_columns(self) -> tuple:
        """All column names including dependent (if set)."""
        if self.dependent is not None:
            return (self.dependent,) + self.independents + self.controls
        return self.independents + self.controls

    def __repr__(self) -> str:
        dep = self.dependent or "None"
        return (
            f"ModelSpec(n={self.n}, source={self.source_label}, "
            f"dependent={dep}, "
            f"independents={list(self.independents)}, "
            f"controls={list(self.controls)})"
        )


class ResearchHandler:
    def __init__(
        self,
        source: str | pd.DataFrame | gpd.GeoDataFrame,
        handler: Optional[Callable] = None,
        *,
        shapefile: bool = False,
    ):
        """
        Args:
            source: filepath (CSV or shapefile), DataFrame, or GeoDataFrame
            handler: optional transform applied after loading
            shapefile: if True, read source path as a shapefile

        Examples:
            ResearchHandler("data.csv")
            ResearchHandler("data.csv", lambda df: df.dropna())
            ResearchHandler("regions.shp", shapefile=True)
            ResearchHandler(existing_df)
        """
        self.data = self._load(source, handler, shapefile)
        self.subset = None
        self.dependent = None
        self.independents = []
        self.controls = []
        self._source_mode: Optional[str] = (
            None  # "full" or "subset", locked on first variable call
        )

    @staticmethod
    def _load(
        source: str | pd.DataFrame | gpd.GeoDataFrame,
        handler: Optional[Callable],
        shapefile: bool,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        if isinstance(source, (pd.DataFrame, gpd.GeoDataFrame)):
            raw = source
        elif isinstance(source, str):
            raw = gpd.read_file(source) if shapefile else pd.read_csv(source)
        else:
            raise TypeError(
                f"source must be a filepath, DataFrame, or GeoDataFrame — got {type(source).__name__}"
            )
        return handler(raw) if handler else raw

    def create_subset(self, condition: Callable) -> None:
        """
        Example Usage:

            handler.create_subset(lambda df: df["age"] > 30)
            handler.create_subset(lambda df: df["country"].isin(["US", "UK"]))
        """
        if self.data is not None:
            self.subset = self.data[condition(self.data)].copy()
        else:
            print("No full dataset available")
            return
        print(f"Subset created with {len(self.subset)} rows")

    def _enforce_source_mode(self, full: bool) -> None:
        """
        Lock the source mode on the first variable-setting call.
        Raises ValueError if a subsequent call uses a different mode.
        """
        mode = "full" if full else "subset"
        if self._source_mode is None:
            self._source_mode = mode
        elif self._source_mode != mode:
            raise ValueError(
                f"Source mode conflict: variables are being set from '{self._source_mode}' "
                f"but this call uses '{'full' if full else 'subset'}'. "
                f"Call clear_caches() before switching between full and subset."
            )

    def set_dependent(self, col: str, full: bool = True) -> None:
        """
        Example Usage:

            handler.set_dependent("income")
            handler.set_dependent("income", full=False)
        """
        self._enforce_source_mode(full)
        if full and self.data is not None:
            self.dependent = self.data[col]
        elif not full and self.subset is not None:
            self.dependent = self.subset[col]
        else:
            print("No valid dataset available")
            return
        print(f"Dependent variable set to: {col}")

    def add_independents(self, *cols: str, full: bool = True) -> None:
        """
        Example Usage:

            handler.add_independents("age", "education", "experience")
            handler.add_independents("age", "education", full=False)
        """
        self._enforce_source_mode(full)
        if full and self.data is not None:
            df = self.data
        elif not full and self.subset is not None:
            df = self.subset
        else:
            print("No valid dataset available")
            return
        for col in cols:
            self.independents.append(df[col])
        print(f"Independent variables: {[s.name for s in self.independents]}")

    def add_controls(self, *cols: str, full: bool = True) -> None:
        """
        Example Usage:

            handler.add_controls("gender", "region")
            handler.add_controls("gender", "region", full=False)
        """
        self._enforce_source_mode(full)
        if full and self.data is not None:
            df = self.data
        elif not full and self.subset is not None:
            df = self.subset
        else:
            print("No valid dataset available")
            return
        for col in cols:
            self.controls.append(df[col])
        print(f"Control variables: {[s.name for s in self.controls]}")

    def get_X(self) -> Optional[pd.DataFrame]:
        if not self.independents:
            print("No independent variables set")
            return None
        cols = self.independents + self.controls
        return pd.concat(cols, axis=1)

    def get_y(self) -> Optional[pd.Series]:
        if self.dependent is None:
            print("No dependent variable set")
            return None
        return self.dependent

    def attach(
        self,
        col_name: str,
        series: pd.Series,
        to_full: bool = True,
        quiet: bool = False,
    ) -> None:
        """
        Example Usage:

            handler.attach("log_income", np.log(handler.data["income"]))
            handler.attach("log_income", some_series, to_full=False)
        """
        if to_full and self.data is not None:
            self.data[col_name] = series
        elif not to_full and self.subset is not None:
            self.subset[col_name] = series.loc[self.subset.index]
        else:
            print("No valid dataset available")
            return
        if not quiet:
            print(f"Attached '{col_name}' to dataset")

    def normalize_and_attach(
        self,
        source_col: str,
        normalizing_function: Callable,
        new_colname: str,
        full: bool = True,
    ) -> None:
        """
        Pulls 1 column and attaches based on a normalizing Callable
        Example Usage:

            handler.normalize_and_attach("income", np.log, "log_income")
            handler.normalize_and_attach("score", lambda s: (s - s.mean()) / s.std(), "z_score", full=False)
        """
        if full and self.data is not None:
            result = normalizing_function(self.data[source_col])
            self.attach(col_name=new_colname, series=result, to_full=True, quiet=True)
        elif not full and self.subset is not None:
            result = normalizing_function(self.subset[source_col])
            self.attach(col_name=new_colname, series=result, to_full=False, quiet=True)
        else:
            print("No valid dataset available")
            return
        print(
            f"Created {new_colname} from {source_col} using function: {normalizing_function.__name__} and attached to dataset"
        )

    def calculate_and_attach(
        self,
        source_cols: list[str],
        func: Callable,
        new_colname: str,
        full: bool = True,
    ) -> None:
        """
        Pulls 2 or more columns for calculation, and attaches to dataset


            handler.calculate_and_attach(["price", "quantity"], lambda df: df["price"] * df["quantity"], "revenue")
            handler.calculate_and_attach(["math", "reading"], lambda df: df.mean(axis=1), "avg_score", full=False)
        little weird
        """
        if full and self.data is not None:
            result = func(self.data[source_cols])
            self.attach(col_name=new_colname, series=result, to_full=True, quiet=True)
        elif not full and self.subset is not None:
            result = func(self.subset[source_cols])
            self.attach(col_name=new_colname, series=result, to_full=False, quiet=True)
        else:
            print("No valid dataset available")
            return
        print(f"Created {new_colname} from {source_cols} and attached to dataset")

    def get_spec(self) -> ModelSpec:
        """
        Return a frozen snapshot of the current variable specification.

        The ModelSpec contains copies of the design matrix, dependent variable,
        column name metadata, and the source DataFrame (for distribution fitting
        in the simulation module).

        Raises:
            RuntimeError: if no independents have been set

        Example:
            rh.set_dependent("log_income")
            rh.add_independents("education", "experience")
            rh.add_controls("female")

            spec = rh.get_spec()
            spec.X              # DataFrame
            spec.y              # Series
            spec.independents   # ("education", "experience")
            spec.controls       # ("female",)
        """
        X = self.get_X()
        if X is None:
            raise RuntimeError("Cannot build ModelSpec: no independent variables set.")
        y = self.get_y()

        # determine source dataframe
        if self._source_mode == "subset":
            source_df = self.subset
        else:
            source_df = self.data

        return ModelSpec(
            X=X.copy(),
            y=y.copy() if y is not None else None,
            independents=tuple(s.name for s in self.independents),
            controls=tuple(s.name for s in self.controls),
            dependent=self.dependent.name if self.dependent is not None else None,
            source_label=self._source_mode or "full",
            n=len(X),
            data=source_df.copy(),
        )

    def reset_subset(self) -> None:
        self.subset = None
        print("Subset cleared")

    def clear_caches(self) -> None:
        self.dependent = None
        self.independents = []
        self.controls = []
        self._source_mode = None
        print("Caches cleared")
