"""
Plotter Module — ResearchFramework Visualization Layer
=======================================================

External visualization module for the ResearchFramework. Decoupled from
the core simulation engine so that simulation.py has zero viz dependencies.

Classes:
    SimulationPlotter   — Monte Carlo simulation results (histograms,
                          KDEs, tornado charts, convergence, scenarios)

Future:
    TimeSeriesPlotter   — time series diagnostics, regime highlighting
    RegressionPlotter   — coefficient plots, residual diagnostics
    ...

All plotter classes return plotly Figure objects.
Call .show() for interactive display or .write_image() for static export.

Dependencies:
    pip install plotly kaleido scipy

Usage:
    from plotter import SimulationPlotter

    plot = SimulationPlotter()
    fig = plot.histogram(result, outcome_label="CPI YoY (%)")
    fig.show()
    fig.write_image("chart.png", scale=2)
"""

import numpy as np
import pandas as pd
from typing import Optional
from .simulation import ConvergenceDiagnostics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


# ═══════════════════════════════════════════════════════════════════════════
# SHARED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Accessible, distinguishable, print-safe palette
PALETTE = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
    "#FF6692",  # pink
    "#B6E880",  # lime
]

# Common layout defaults applied to every figure
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    hoverlabel=dict(font_size=12),
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION PLOTTER
# ═══════════════════════════════════════════════════════════════════════════


class SimulationPlotter:
    """
    Plotly-based visualization suite for Monte Carlo simulation results.

    Every method returns a plotly.graph_objects.Figure.
    All methods accept an optional `outcome_label` kwarg to set the
    x-axis label (default "Outcome").

    Methods:
        histogram             — distribution with mean line and CI shading
        cumulative_density    — empirical CDF with percentile markers
        convergence_plot      — running mean ± SE band
        tornado_chart         — diverging sensitivity bars centered on baseline
        scenario_comparison   — overlaid KDE curves across scenarios
        tornado_comparison    — side-by-side tornado subplots
        histogram_comparison  — overlaid semi-transparent histograms
    """

    @staticmethod
    def _extract(result) -> np.ndarray:
        """Pull a 1-D array from outcomes (handles multi-output)."""
        if isinstance(result.outcomes, pd.DataFrame):
            return pd.DataFrame(np.array(result.outcomes.iloc[:, 0])).values
        return np.asarray(result.outcomes)

    # ── Histogram ─────────────────────────────────────────────────

    @staticmethod
    def histogram(
        result,
        bins: int = 80,
        outcome_label: str = "Outcome",
        title: str = "Simulation Outcome Distribution",
        **kwargs,
    ) -> go.Figure:
        """Distribution histogram with mean line and 95% CI shading."""

        values = SimulationPlotter._extract(result)
        if result.mean is None:
            result.summarize()

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=bins,
                histnorm="probability density",
                marker=dict(
                    color=PALETTE[0], opacity=0.7, line=dict(color="white", width=0.4)
                ),
                name="Outcome",
                hovertemplate="%{x:.2f}<extra></extra>",
            )
        )

        # CI shading
        if result.ci_lower is not None:
            fig.add_vrect(
                x0=result.ci_lower,
                x1=result.ci_upper,
                fillcolor=PALETTE[2],
                opacity=0.12,
                line_width=0,
                annotation_text=f"95% CI: [{result.ci_lower:,.2f}, {result.ci_upper:,.2f}]",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color=PALETTE[2],
            )

        # mean line
        fig.add_vline(
            x=result.mean,
            line_dash="dash",
            line_color=PALETTE[1],
            line_width=2,
            annotation_text=f"Mean: {result.mean:,.2f}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color=PALETTE[1],
        )

        fig.update_layout(
            **_LAYOUT,
            title=title,
            xaxis_title=outcome_label,
            yaxis_title="Density",
            showlegend=False,
        )
        return fig

    # ── Empirical CDF ─────────────────────────────────────────────

    @staticmethod
    def cumulative_density(
        result,
        outcome_label: str = "Outcome",
        title: str = "Empirical CDF",
        **kwargs,
    ) -> go.Figure:
        """Empirical CDF with percentile markers."""

        values = SimulationPlotter._extract(result)
        if result.mean is None:
            result.summarize()

        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sorted_vals,
                y=cdf,
                mode="lines",
                line=dict(color=PALETTE[0], width=2),
                name="CDF",
                hovertemplate="%{x:.2f} → P=%{y:.1%}<extra></extra>",
            )
        )

        for pct in [5, 25, 50, 75, 95]:
            val = result.percentiles[pct]
            fig.add_trace(
                go.Scatter(
                    x=[val],
                    y=[pct / 100],
                    mode="markers+text",
                    marker=dict(color=PALETTE[1], size=8),
                    text=[f"P{pct}: {val:,.2f}"],
                    textposition="middle right",
                    textfont=dict(size=10),
                    showlegend=False,
                    hovertemplate=f"P{pct}: %{{x:.2f}}<extra></extra>",
                )
            )

        fig.update_layout(
            **_LAYOUT,
            title=title,
            xaxis_title=outcome_label,
            yaxis_title="Cumulative Probability",
            showlegend=False,
        )
        return fig

    # ── Convergence ───────────────────────────────────────────────

    @staticmethod
    def convergence_plot(
        outcomes,
        title: str = "Convergence",
        **kwargs,
    ) -> go.Figure:
        """Running mean with ±1 SE band."""

        if isinstance(outcomes, pd.DataFrame):
            outcomes = pd.DataFrame(np.array(outcomes.iloc[:, 0])).values

        running = ConvergenceDiagnostics.running_statistics(outcomes)
        iters = np.array(running["iteration"].values)
        means = np.array(running["cumulative_mean"].values)
        stds = np.array(running["cumulative_std"].values)
        se = stds / np.sqrt(iters)

        fig = go.Figure()

        # SE band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([iters, iters[::-1]]),
                y=np.concatenate([means + se, (means - se)[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(PALETTE[0], 0.15),
                line=dict(width=0),
                name="±1 SE",
                hoverinfo="skip",
            )
        )

        # running mean
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=means,
                mode="lines",
                line=dict(color=PALETTE[0], width=2),
                name="Running mean",
                hovertemplate="n=%{x:,}<br>mean=%{y:.3f}<extra></extra>",
            )
        )

        # final reference
        fig.add_hline(
            y=means[-1],
            line_dash="dot",
            line_color="gray",
            line_width=1,
            annotation_text=f"Final: {means[-1]:,.3f}",
            annotation_position="bottom right",
            annotation_font_size=11,
        )

        fig.update_layout(
            **_LAYOUT,
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Cumulative Mean",
        )
        return fig

    # ── Tornado Chart ─────────────────────────────────────────────

    @staticmethod
    def tornado_chart(
        tornado_data: pd.DataFrame,
        outcome_label: str = "Outcome",
        title: str = "Sensitivity Analysis (10th–90th Percentile)",
        **kwargs,
    ) -> go.Figure:
        """
        Horizontal diverging bar chart centered on the baseline outcome.
        Bars extend left (low) and right (high) from baseline.
        """

        df = tornado_data.sort_values("swing", ascending=True)
        overall_baseline = (
            (np.array(df["low_outcome"].values) + np.array(df["high_outcome"].values))
            / 2
        ).mean()

        fig = go.Figure()

        # low side (leftward from baseline)
        fig.add_trace(
            go.Bar(
                y=df["variable"],
                x=df["low_outcome"] - overall_baseline,
                base=overall_baseline,
                orientation="h",
                marker=dict(color=PALETTE[0], opacity=0.8),
                name="Low (P10)",
                customdata=np.column_stack([df["low_value"], df["low_outcome"]]),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Input: %{customdata[0]:.3f}<br>"
                    "Outcome: %{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )

        # high side (rightward from baseline)
        fig.add_trace(
            go.Bar(
                y=df["variable"],
                x=df["high_outcome"] - overall_baseline,
                base=overall_baseline,
                orientation="h",
                marker=dict(color=PALETTE[1], opacity=0.8),
                name="High (P90)",
                customdata=np.column_stack([df["high_value"], df["high_outcome"]]),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Input: %{customdata[0]:.3f}<br>"
                    "Outcome: %{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )

        # swing annotations
        for _, row in df.iterrows():
            fig.add_annotation(
                x=max(row["high_outcome"], row["low_outcome"]) + row["swing"] * 0.05,
                y=row["variable"],
                text=f"Δ{row['swing']:.2f}",
                showarrow=False,
                font=dict(size=11, color="#444"),
                xanchor="left",
            )

        # baseline reference
        fig.add_vline(
            x=overall_baseline,
            line_dash="dot",
            line_color="gray",
            line_width=1.5,
            annotation_text=f"Baseline: {overall_baseline:.2f}",
            annotation_position="top",
            annotation_font_size=10,
        )

        fig.update_layout(
            **_LAYOUT,
            title=title,
            xaxis_title=outcome_label,
            barmode="overlay",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"
            ),
            height=max(350, 80 * len(df)),
        )
        return fig

    # ── Scenario Comparison ───────────────────────────────────────

    @staticmethod
    def scenario_comparison(
        results: dict,
        outcome_label: str = "Outcome",
        title: str = "Scenario Comparison",
        **kwargs,
    ) -> go.Figure:
        """Overlaid KDE curves across scenarios with mean annotations."""

        fig = go.Figure()

        # shared x-axis range
        all_values = [SimulationPlotter._extract(r) for r in results.values()]
        global_min = min(v.min() for v in all_values)
        global_max = max(v.max() for v in all_values)
        x_range = np.linspace(global_min, global_max, 500)

        for i, (name, result) in enumerate(results.items()):
            values = SimulationPlotter._extract(result)
            if result.mean is None:
                result.summarize()

            color = PALETTE[i % len(PALETTE)]
            kde = gaussian_kde(values)
            y = kde(x_range)

            # KDE curve + fill
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(color, 0.08),
                    line=dict(color=color, width=2),
                    name=f"{name} (μ={result.mean:,.2f})",
                    hovertemplate=f"{name}<br>%{{x:.2f}}: density=%{{y:.4f}}<extra></extra>",
                )
            )

            # mean diamond
            peak_y = kde(np.array([result.mean]))[0]
            fig.add_trace(
                go.Scatter(
                    x=[result.mean],
                    y=[peak_y],
                    mode="markers",
                    marker=dict(color=color, size=7, symbol="diamond"),
                    showlegend=False,
                    hovertemplate=f"{name} mean: {result.mean:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            **_LAYOUT,
            title=title,
            xaxis_title=outcome_label,
            yaxis_title="Density",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                x=1.02,
                font=dict(size=11),
            ),
        )
        return fig

    # ── Side-by-Side Tornado ──────────────────────────────────────

    @staticmethod
    def tornado_comparison(
        tornado_data_list: list,
        labels: list[str],
        outcome_label: str = "Outcome",
        title: str = "Sensitivity Comparison",
        **kwargs,
    ) -> go.Figure:
        """
        Side-by-side tornado subplots for comparing sensitivity across
        regimes or scenarios.

        Args:
            tornado_data_list: list of DataFrames from tornado()
            labels: names for each subplot (e.g. ["Low Debt", "High Debt"])
        """

        n = len(tornado_data_list)
        fig = make_subplots(
            rows=1,
            cols=n,
            subplot_titles=labels,
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        for col_idx, (df_tornado, label) in enumerate(
            zip(tornado_data_list, labels), 1
        ):
            df = df_tornado.sort_values("swing", ascending=True)
            mid = ((df["low_outcome"].values + df["high_outcome"].values) / 2).mean()

            # low bars
            fig.add_trace(
                go.Bar(
                    y=df["variable"],
                    x=df["low_outcome"] - mid,
                    base=mid,
                    orientation="h",
                    marker=dict(color=PALETTE[0], opacity=0.8),
                    name="Low (P10)" if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    hovertemplate="<b>%{y}</b><br>Outcome: %{x:.2f}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

            # high bars
            fig.add_trace(
                go.Bar(
                    y=df["variable"],
                    x=df["high_outcome"] - mid,
                    base=mid,
                    orientation="h",
                    marker=dict(color=PALETTE[1], opacity=0.8),
                    name="High (P90)" if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    hovertemplate="<b>%{y}</b><br>Outcome: %{x:.2f}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

            # swing annotations
            for _, row in df.iterrows():
                fig.add_annotation(
                    x=max(row["high_outcome"], row["low_outcome"])
                    + row["swing"] * 0.05,
                    y=row["variable"],
                    text=f"Δ{row['swing']:.2f}",
                    showarrow=False,
                    font=dict(size=10, color="#444"),
                    xanchor="left",
                    xref=f"x{col_idx}" if col_idx > 1 else "x",
                    yref=f"y{col_idx}" if col_idx > 1 else "y",
                )

            # baseline line
            fig.add_vline(
                x=mid,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                row=1,
                col=col_idx,
            )

            fig.update_xaxes(title_text=outcome_label, row=1, col=col_idx)

        fig.update_layout(
            **_LAYOUT,
            title=title,
            barmode="overlay",
            height=max(400, 90 * max(len(d) for d in tornado_data_list)),
            width=450 * n,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.06, x=0.5, xanchor="center"
            ),
        )
        return fig

    # ── Histogram Comparison ──────────────────────────────────────

    @staticmethod
    def histogram_comparison(
        results: dict,
        outcome_label: str = "Outcome",
        title: str = "Distribution Comparison",
        **kwargs,
    ) -> go.Figure:
        """Overlaid semi-transparent histograms for quick comparison."""

        fig = go.Figure()

        for i, (name, result) in enumerate(results.items()):
            values = SimulationPlotter._extract(result)
            if result.mean is None:
                result.summarize()

            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=60,
                    histnorm="probability density",
                    name=f"{name} (μ={result.mean:,.2f})",
                    marker=dict(
                        color=color, opacity=0.4, line=dict(color=color, width=1)
                    ),
                    hovertemplate=f"{name}<br>%{{x:.2f}}<extra></extra>",
                )
            )

        fig.update_layout(
            **_LAYOUT,
            title=title,
            xaxis_title=outcome_label,
            yaxis_title="Density",
            barmode="overlay",
        )
        return fig
