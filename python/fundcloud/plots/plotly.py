"""Plotly figure builders.

Each public function returns a :class:`plotly.graph_objects.Figure` so the
user can compose these with their own dashboards. Every builder accepts
either a :class:`pandas.Series` (single strategy) or a
:class:`pandas.DataFrame` (one column per strategy, overlayed) — the
exception is :func:`monthly_heatmap`, which needs a single series.

Theming piggy-backs on plotly's builtin templates via
:mod:`fundcloud.plots.themes`. Rich on-figure annotations (stats pills,
VaR/CVaR reference lines, drawdown durations) are available via
``annotations=True`` — off by default so figures compose cleanly into
:class:`fundcloud.reports.Tearsheet` without duplicating its stat cards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fundcloud.metrics import core as _metrics
from fundcloud.metrics import periods as _period_metrics
from fundcloud.plots import _annotations as _ann
from fundcloud.plots._normalize import to_series_list, to_single_series
from fundcloud.plots.themes import _resolve_template

__all__ = [
    "composition",
    "cumulative",
    "drawdown",
    "monthly_heatmap",
    "return_distribution",
    "rolling_sharpe",
    "yearly_returns_bars",
]


_DEFAULT_LAYOUT: dict[str, object] = {
    "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
}

# Diverging colorscale used by the monthly heatmap (theme-invariant: the
# semantics are red=loss / green=gain, not driven by palette).
_DIVERGING_SCALE = [(0, "#C0392B"), (0.5, "white"), (1, "#1F9B64")]


def _style(
    fig: go.Figure,
    *,
    title: str,
    theme: str | None = None,
    y_tick_format: str | None = None,
) -> go.Figure:
    template = _resolve_template(theme)
    fig.update_layout(
        title={"text": title, "x": 0.01},
        margin=_DEFAULT_LAYOUT["margin"],
    )
    if template is None:
        # Preserve the original fundcloud look when no plotly template is
        # selected — white paper, Inter font.
        fig.update_layout(
            plot_bgcolor=_DEFAULT_LAYOUT["plot_bgcolor"],
            paper_bgcolor=_DEFAULT_LAYOUT["paper_bgcolor"],
            font=_DEFAULT_LAYOUT["font"],
        )
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.2)")
        fig.update_yaxes(
            gridcolor="rgba(0,0,0,0.08)",
            zerolinecolor="rgba(0,0,0,0.2)",
            tickformat=y_tick_format,
        )
    else:
        fig.update_layout(template=template)
        if y_tick_format is not None:
            fig.update_yaxes(tickformat=y_tick_format)
    return fig


def _stats_pill(fig: go.Figure, text: str, *, xref: str = "paper", yref: str = "paper") -> None:
    # Force dark text on a near-opaque near-white pill so the numbers stay
    # legible regardless of the active plotly template (dark themes default
    # to light text, which vanishes against our white pill background).
    fig.add_annotation(
        text=text,
        xref=xref,
        yref=yref,
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        font={
            "size": 11,
            "family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            "color": "#1c1c1c",
        },
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="rgba(0,0,0,0.22)",
        borderwidth=1,
        borderpad=6,
    )


# --------------------------------------------------------------------- figures


def cumulative(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    title: str = "Cumulative return (%)",
    theme: str | None = None,
    annotations: bool = False,
) -> go.Figure:
    """Cumulative-return curve (0% at inception, percent-formatted y-axis).

    ``returns`` may be a :class:`pandas.Series` or a :class:`pandas.DataFrame`
    (one line per column). ``benchmark`` is drawn as a dashed reference. Each
    series is ``dropna()``'d before compounding so staggered inception dates
    produce continuous lines from their own first observation rather than
    broken/dashed segments.
    """
    fig = go.Figure()
    series_list = to_series_list(returns)
    for name, series in series_list:
        clean = series.dropna()
        cum = (1.0 + clean).cumprod() - 1.0
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                name=name,
                mode="lines",
                line={"width": 2},
                connectgaps=True,
            )
        )
    if benchmark is not None:
        bench_clean = benchmark.dropna()
        bench_cum = (1.0 + bench_clean).cumprod() - 1.0
        fig.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                name=str(benchmark.name) if benchmark.name is not None else "benchmark",
                line={"color": "#888", "width": 1.5, "dash": "dash"},
                connectgaps=True,
            )
        )
    if annotations:
        _stats_pill(fig, _ann.cumulative_pill(series_list, benchmark=benchmark))
    return _style(fig, title=title, theme=theme, y_tick_format=".0%")


def drawdown(
    returns: pd.Series | pd.DataFrame,
    *,
    title: str = "Drawdown (%)",
    theme: str | None = None,
    annotations: bool = False,
) -> go.Figure:
    """Underwater (drawdown) chart.

    Accepts a :class:`pandas.Series` or a multi-column
    :class:`pandas.DataFrame`; each column is rendered as its own filled
    area with reduced opacity so overlaps stay legible.
    """
    fig = go.Figure()
    series_list = to_series_list(returns)
    single = len(series_list) == 1
    for name, series in series_list:
        clean = series.dropna()
        dd = _metrics.drawdown_series(clean) * 100.0
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                name=name,
                mode="lines",
                fill="tozeroy" if single else None,
                line={"width": 1.2} if single else {"width": 1.5},
                opacity=1.0 if single else 0.9,
                connectgaps=True,
            )
        )
    if annotations:
        _stats_pill(fig, _ann.drawdown_pill(series_list))
    return _style(fig, title=title, theme=theme, y_tick_format=".1f")


def rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 63,
    periods_per_year: int = 252,
    title: str | None = None,
    theme: str | None = None,
    annotations: bool = False,
) -> go.Figure:
    """Rolling annualised Sharpe (``window``-period window).

    Multi-column input overlays one line per column.
    """
    fig = go.Figure()
    series_list = to_series_list(returns)
    for name, series in series_list:
        clean = series.dropna()
        mu = clean.rolling(window).mean()
        sigma = clean.rolling(window).std(ddof=1)
        rs = (mu / sigma) * np.sqrt(periods_per_year)
        fig.add_trace(
            go.Scatter(
                x=rs.index,
                y=rs.values,
                name=f"{name} ({window}-bar)",
                mode="lines",
                line={"width": 1.5},
                connectgaps=True,
            )
        )
    fig.add_hline(y=0, line={"color": "rgba(0,0,0,0.4)", "width": 1, "dash": "dot"})
    if annotations:
        _ann.annotate_full_period_sharpe(
            fig, _metrics.sharpe(returns, periods_per_year=periods_per_year)
        )
    return _style(
        fig,
        title=title or f"Rolling Sharpe ({window} bars)",
        theme=theme,
        y_tick_format=".2f",
    )


def return_distribution(
    returns: pd.Series | pd.DataFrame,
    *,
    bins: int = 60,
    title: str = "Return distribution (%)",
    theme: str | None = None,
    annotations: bool = False,
) -> go.Figure:
    """Histogram of per-period returns (in %).

    Multi-column input overlays translucent histograms.
    """
    fig = go.Figure()
    series_list = to_series_list(returns)
    for name, series in series_list:
        fig.add_trace(
            go.Histogram(
                x=series.values * 100.0,
                nbinsx=bins,
                name=name,
                opacity=0.85 if len(series_list) == 1 else 0.55,
            )
        )
    if len(series_list) > 1:
        fig.update_layout(barmode="overlay")
    if annotations:
        _ann.annotate_var_cvar(fig, series_list)
        _stats_pill(fig, _ann.distribution_pill(series_list))
    return _style(fig, title=title, theme=theme, y_tick_format="d")


def monthly_heatmap(
    returns: pd.Series | pd.DataFrame,
    *,
    title: str = "Monthly returns (%)",
    theme: str | None = None,
    annotations: bool = False,
    colorbar: dict[str, object] | None = None,
    text_values: bool = True,
) -> go.Figure:
    """Year × month heatmap of aggregated returns.

    Requires a :class:`pandas.Series` or a single-column
    :class:`pandas.DataFrame` — overlaying heatmaps is not meaningful. Pass
    one column for multi-asset frames, or call :func:`fundcloud.plots.summary`.

    ``text_values=True`` (default) renders each cell's percent-return
    directly on the tile so readers don't need to colour-match against the
    scale bar. Pass ``False`` on very dense panels (e.g. > 20 years) if
    the numbers crowd out.
    """
    series = to_single_series(returns, caller="monthly_heatmap")
    # dropna first so pre-inception periods (e.g. BTC-USD prior to 2014 when
    # combined with an asset that started earlier) don't pad the pivot with
    # bogus zero-return months that squash the colour scale and compress
    # row heights.
    series = series.dropna().copy()
    if series.empty:
        fig = go.Figure()
        return _style(fig, title=title, theme=theme)
    series.index = pd.DatetimeIndex(series.index)
    monthly = ((1.0 + series).resample("ME").prod() - 1.0) * 100.0
    table = pd.pivot_table(
        monthly.to_frame("ret"),
        index=monthly.index.year,
        columns=monthly.index.month,
        values="ret",
    )
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = [months[m - 1] for m in table.columns]

    cb = {"title": "%", "thickness": 10}
    if colorbar is not None:
        cb = {**cb, **colorbar}

    heatmap_kwargs: dict[str, object] = {
        "z": table.values,
        "x": cols,
        "y": table.index.astype(str),
        "colorscale": _DIVERGING_SCALE,
        "zmid": 0,
        "colorbar": cb,
        "hovertemplate": "%{y} %{x}: %{z:.2f}%<extra></extra>",
    }
    if text_values:
        heatmap_kwargs["texttemplate"] = "%{z:.1f}"
        heatmap_kwargs["textfont"] = {"size": 10, "color": "#1c1c1c"}

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    # type='category' + autorange='reversed' gives newest-at-top while
    # letting plotly auto-scale all 13+ years into the available vertical
    # space. Forcing tickmode='array' + explicit tickvals (as we tried
    # earlier) interacts badly with plotly's row_heights allocation in
    # the composite summary and collapses cells to near-zero height.
    fig.update_yaxes(autorange="reversed", type="category")
    if annotations:
        _ann.annotate_heatmap_margins(fig, table)
    return _style(fig, title=title, theme=theme)


def composition(
    weights: pd.DataFrame,
    *,
    title: str = "Portfolio composition",
    theme: str | None = None,
    annotations: bool = False,
) -> go.Figure:
    """Stacked-area chart of per-asset weight over time."""
    if weights.empty:
        return _style(go.Figure(), title=title, theme=theme, y_tick_format=".0%")
    fig = go.Figure()
    for asset in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[asset],
                mode="lines",
                name=str(asset),
                stackgroup="one",
                line={"width": 0.5},
            )
        )
    resolved_title = title
    if annotations:
        turnover = _ann.turnover(weights)
        resolved_title = f"{title} — avg turnover {turnover:.2%}/period"
    return _style(fig, title=resolved_title, theme=theme, y_tick_format=".0%")


_BENCHMARK_BAR_COLOR = "#F0C36D"
_STRATEGY_BAR_COLOR = "#2F6EE6"
_MEAN_LINE_COLOR = "#C0392B"


def yearly_returns_bars(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    title: str | None = None,
    theme: str | None = None,
) -> go.Figure:
    """Paired grouped-bar chart of end-of-year returns.

    One bar per (year, series) — benchmark first, strategy second — with
    a dashed horizontal reference line at the strategy's mean yearly
    return. Y-axis is percent-formatted.
    """
    series_list = to_series_list(returns)
    resolved_title = title or (
        "EOY Returns vs Benchmark" if benchmark is not None else "EOY Returns"
    )
    fig = go.Figure()
    if benchmark is not None:
        bench_yearly = _period_metrics.yearly_returns(benchmark.dropna())
        fig.add_trace(
            go.Bar(
                x=bench_yearly.index.astype(str),
                y=bench_yearly.values,
                name=str(benchmark.name) if benchmark.name is not None else "Benchmark",
                marker_color=_BENCHMARK_BAR_COLOR,
            )
        )
    mean_ref: float | None = None
    for name, series in series_list:
        yearly = _period_metrics.yearly_returns(series.dropna())
        fig.add_trace(
            go.Bar(
                x=yearly.index.astype(str),
                y=yearly.values,
                name=name,
                marker_color=_STRATEGY_BAR_COLOR,
            )
        )
        if mean_ref is None and not yearly.empty:
            mean_ref = float(yearly.mean())
    fig.update_layout(barmode="group")
    if mean_ref is not None and np.isfinite(mean_ref):
        fig.add_hline(
            y=mean_ref,
            line={"color": _MEAN_LINE_COLOR, "width": 1.5, "dash": "dash"},
            annotation_text=f"mean {mean_ref:.1%}",
            annotation_position="top right",
        )
    fig.add_hline(y=0.0, line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dash"})
    return _style(fig, title=resolved_title, theme=theme, y_tick_format=".0%")
