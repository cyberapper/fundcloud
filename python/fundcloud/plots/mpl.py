"""Matplotlib-backed mirrors of the plotly builders.

Each figure-level public builder has a private ``_build_<name>(ax, ...)``
companion that draws into a supplied :class:`~matplotlib.axes.Axes`. This
split lets :func:`summary` compose panels via
:class:`~matplotlib.gridspec.GridSpec` without spinning up and harvesting
independent :class:`matplotlib.figure.Figure` objects.

Theming is intentionally a plotly-only concern (see
:mod:`fundcloud.plots.themes`). The matplotlib builders keep their
static styling.

Requires ``fundcloud[viz]`` for matplotlib.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fundcloud.metrics import core as _metrics
from fundcloud.plots._annotations import turnover
from fundcloud.plots._normalize import to_series_list, to_single_series

__all__ = [
    "composition",
    "cumulative",
    "drawdown",
    "monthly_heatmap",
    "return_distribution",
    "rolling_sharpe",
    "summary",
]


def _require_mpl() -> tuple[Any, Any]:
    try:
        import sys

        import matplotlib

        # Only switch to Agg before pyplot is first imported.  Calling
        # matplotlib.use() after pyplot is already loaded triggers a
        # RecursionError on Python 3.14 due to re-entrant backend init.
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for PDF / static plot rendering. uv add 'fundcloud[viz]'."
        raise ImportError(msg) from e
    return matplotlib, plt


def _setup_ax(ax: Any, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="600")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#E7E9EE", linestyle="-", linewidth=0.6)


def _make_fig(title: str, *, size: tuple[float, float] = (8, 3)) -> tuple[Any, Any]:
    _, plt = _require_mpl()
    fig, ax = plt.subplots(figsize=size, dpi=120)
    _setup_ax(ax, title)
    return fig, ax


# --------------------------------------------------------------------- axes-level builders


def _build_cumulative(
    ax: Any,
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    title: str = "Cumulative return (%)",
    annotations: bool = False,
) -> None:
    import matplotlib.ticker as _mtick

    _setup_ax(ax, title)
    for name, series in to_series_list(returns):
        cum = (1.0 + series).cumprod() - 1.0
        ax.plot(cum.index, cum.values, label=name, linewidth=1.8)
    if benchmark is not None:
        b = (1.0 + benchmark).cumprod() - 1.0
        ax.plot(
            b.index,
            b.values,
            color="#888",
            linestyle="--",
            linewidth=1.4,
            label=str(benchmark.name) if benchmark.name is not None else "benchmark",
        )
    ax.yaxis.set_major_formatter(_mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if annotations:
        _mpl_stats_pill(ax, _mpl_cumulative_pill(to_series_list(returns), benchmark=benchmark))


def _build_drawdown(
    ax: Any,
    returns: pd.Series | pd.DataFrame,
    *,
    title: str = "Drawdown (%)",
    annotations: bool = False,
) -> None:
    _setup_ax(ax, title)
    series_list = to_series_list(returns)
    single = len(series_list) == 1
    for name, series in series_list:
        dd = _metrics.drawdown_series(series) * 100.0
        if single:
            ax.fill_between(dd.index, dd.values, 0.0, alpha=0.35, color="#C0392B")
            ax.plot(dd.index, dd.values, color="#8E1D13", linewidth=0.8)
        else:
            ax.plot(dd.index, dd.values, linewidth=1.2, label=name, alpha=0.9)
    if not single:
        ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if annotations:
        _mpl_stats_pill(ax, _mpl_drawdown_pill(series_list))


def _build_rolling_sharpe(
    ax: Any,
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 63,
    periods_per_year: int = 252,
    title: str | None = None,
    annotations: bool = False,
) -> None:
    _setup_ax(ax, title or f"Rolling Sharpe ({window} bars)")
    for name, series in to_series_list(returns):
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std(ddof=1)
        rs = (mu / sigma) * np.sqrt(periods_per_year)
        ax.plot(rs.index, rs.values, linewidth=1.4, label=f"{name} ({window}-bar)")
    ax.axhline(0, color="#444", linewidth=0.6, linestyle=":")
    if annotations:
        full = _metrics.sharpe(returns, periods_per_year=periods_per_year)
        value = float(full.mean()) if isinstance(full, pd.Series) else float(full)
        if np.isfinite(value):
            ax.axhline(value, color="#000", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(
                0.99,
                0.02,
                f"full-period Sharpe {value:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#333",
            )
    if len(to_series_list(returns)) > 1:
        ax.legend(loc="upper left", frameon=False, fontsize=9)


def _build_return_distribution(
    ax: Any,
    returns: pd.Series | pd.DataFrame,
    *,
    bins: int = 60,
    title: str = "Return distribution (%)",
    annotations: bool = False,
) -> None:
    _setup_ax(ax, title)
    series_list = to_series_list(returns)
    single = len(series_list) == 1
    for name, series in series_list:
        ax.hist(
            series.values * 100.0,
            bins=bins,
            alpha=0.85 if single else 0.5,
            label=name,
        )
    if not single:
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if annotations and series_list:
        _, first = series_list[0]
        var_pct = float(_metrics.value_at_risk(first)) * 100.0
        cvar_pct = float(_metrics.cvar(first)) * 100.0
        ax.axvline(var_pct, color="#C0392B", linestyle="--", linewidth=1, alpha=0.75)
        ax.axvline(cvar_pct, color="#8E1D13", linestyle=":", linewidth=1, alpha=0.75)
        _mpl_stats_pill(ax, _mpl_distribution_pill(series_list))


def _build_monthly_heatmap(
    ax: Any,
    returns: pd.Series | pd.DataFrame,
    *,
    title: str = "Monthly returns (%)",
    annotations: bool = False,
) -> Any:
    series = to_single_series(returns, caller="monthly_heatmap").copy()
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
    years = [str(y) for y in table.index]

    v = max(abs(np.nanmin(table.values)), abs(np.nanmax(table.values)), 1.0)
    im = ax.imshow(table.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    for row_i in range(table.shape[0]):
        for col_i in range(table.shape[1]):
            val = table.values[row_i, col_i]
            if np.isnan(val):
                continue
            ax.text(col_i, row_i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(title, loc="left", fontsize=11, fontweight="600")
    if annotations:
        annual = table.sum(axis=1, min_count=1)
        for row_i, value in enumerate(annual.values):
            if not np.isfinite(value):
                continue
            ax.text(
                len(cols) - 0.3,
                row_i,
                f"  {value:+.1f}%",
                ha="left",
                va="center",
                fontsize=7,
                color="#333",
            )
    return im


def _build_composition(
    ax: Any,
    weights: pd.DataFrame,
    *,
    title: str = "Portfolio composition",
    annotations: bool = False,
) -> None:
    if annotations and not weights.empty:
        title = f"{title} — avg turnover {turnover(weights):.2%}/period"
    _setup_ax(ax, title)
    if not weights.empty:
        ax.stackplot(weights.index, weights.T.values, labels=list(weights.columns))
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_ylim(0, 1)


# --------------------------------------------------------------------- figure-level public API


def cumulative(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    title: str = "Cumulative return (%)",
    annotations: bool = False,
) -> Any:
    fig, ax = _make_fig(title)
    _build_cumulative(ax, returns, benchmark=benchmark, title=title, annotations=annotations)
    fig.tight_layout()
    return fig


def drawdown(
    returns: pd.Series | pd.DataFrame,
    *,
    title: str = "Drawdown (%)",
    annotations: bool = False,
) -> Any:
    fig, ax = _make_fig(title)
    _build_drawdown(ax, returns, title=title, annotations=annotations)
    fig.tight_layout()
    return fig


def rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 63,
    periods_per_year: int = 252,
    title: str | None = None,
    annotations: bool = False,
) -> Any:
    fig, ax = _make_fig(title or f"Rolling Sharpe ({window} bars)")
    _build_rolling_sharpe(
        ax,
        returns,
        window=window,
        periods_per_year=periods_per_year,
        title=title,
        annotations=annotations,
    )
    fig.tight_layout()
    return fig


def return_distribution(
    returns: pd.Series | pd.DataFrame,
    *,
    bins: int = 60,
    title: str = "Return distribution (%)",
    annotations: bool = False,
) -> Any:
    fig, ax = _make_fig(title)
    _build_return_distribution(ax, returns, bins=bins, title=title, annotations=annotations)
    fig.tight_layout()
    return fig


def monthly_heatmap(
    returns: pd.Series | pd.DataFrame,
    *,
    title: str = "Monthly returns (%)",
    annotations: bool = False,
) -> Any:
    _, plt = _require_mpl()
    series = to_single_series(returns, caller="monthly_heatmap")
    span = pd.DatetimeIndex(series.index)
    years_approx = max(int((span.max() - span.min()).days / 365) + 1, 3)
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.45 * years_approx)), dpi=120)
    im = _build_monthly_heatmap(ax, series, title=title, annotations=annotations)
    fig.colorbar(im, ax=ax, shrink=0.6, label="%")
    fig.tight_layout()
    return fig


def composition(
    weights: pd.DataFrame,
    *,
    title: str = "Portfolio composition",
    annotations: bool = False,
) -> Any:
    fig, ax = _make_fig(title)
    _build_composition(ax, weights, title=title, annotations=annotations)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- aggregator


def summary(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    title: str | None = None,
) -> Any:
    """Matplotlib counterpart of :func:`fundcloud.plots.summary`.

    Returns a single :class:`matplotlib.figure.Figure` composing the same
    canonical panels via :class:`~matplotlib.gridspec.GridSpec`.
    """
    import matplotlib.gridspec as gridspec

    _, plt = _require_mpl()
    include_composition = weights is not None and not weights.empty
    rows = 4 if include_composition else 3
    height = 2.6 * rows + 0.8
    fig = plt.figure(figsize=(12, height), dpi=120, constrained_layout=True)
    gs = gridspec.GridSpec(rows, 2, figure=fig, hspace=0.55, wspace=0.22)

    ax_cum = fig.add_subplot(gs[0, :])
    _build_cumulative(ax_cum, returns, benchmark=benchmark, annotations=True)

    ax_dd = fig.add_subplot(gs[1, 0])
    _build_drawdown(ax_dd, returns, annotations=True)

    ax_rs = fig.add_subplot(gs[1, 1])
    _build_rolling_sharpe(ax_rs, returns, annotations=True)

    ax_dist = fig.add_subplot(gs[2, 0])
    _build_return_distribution(ax_dist, returns, annotations=True)

    ax_heat = fig.add_subplot(gs[2, 1])
    first_name, first_series = to_series_list(returns)[0]
    im = _build_monthly_heatmap(
        ax_heat,
        first_series.rename(first_name),
        annotations=True,
    )
    fig.colorbar(im, ax=ax_heat, shrink=0.7, label="%")

    if include_composition:
        ax_comp = fig.add_subplot(gs[3, :])
        _build_composition(ax_comp, weights, annotations=True)

    fig.suptitle(title or "Strategy summary", x=0.02, ha="left", fontsize=13, fontweight="600")
    return fig


# --------------------------------------------------------------------- internal pill helpers


def _mpl_stats_pill(ax: Any, text: str) -> None:
    # Convert <br> from the shared annotations module to newlines.
    ax.text(
        0.99,
        0.98,
        text.replace("<br>", "\n"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#BBB",
            "alpha": 0.82,
        },
    )


def _mpl_cumulative_pill(
    series_list: list[tuple[str, pd.Series]],
    *,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
) -> str:
    # Reuse the plotly pill formatter — mpl converts <br> to newline.
    from fundcloud.plots._annotations import cumulative_pill

    return cumulative_pill(series_list, benchmark=benchmark, periods_per_year=periods_per_year)


def _mpl_drawdown_pill(series_list: list[tuple[str, pd.Series]]) -> str:
    from fundcloud.plots._annotations import drawdown_pill

    return drawdown_pill(series_list)


def _mpl_distribution_pill(series_list: list[tuple[str, pd.Series]]) -> str:
    from fundcloud.plots._annotations import distribution_pill

    return distribution_pill(series_list)
