"""Figure builders.

Default: plotly. Matplotlib equivalents for PDF embedding live under
:mod:`fundcloud.plots.mpl` (require ``fundcloud[viz]``).

All series-accepting builders (:func:`cumulative`, :func:`drawdown`,
:func:`rolling_sharpe`, :func:`return_distribution`) also accept a
:class:`pandas.DataFrame` — one overlayed trace per column.
:func:`monthly_heatmap` is the exception (one series only — pass a column
or use :func:`summary`).

Theme selection (plotly only) goes through :func:`set_theme`; see
:mod:`fundcloud.plots.themes` for the alias map. :func:`summary` composes
all canonical panels into a single Figure ready to write to HTML / PNG.
"""

from __future__ import annotations

from fundcloud.plots.aggregated import summary
from fundcloud.plots.patterns import (
    plot_asset_patterns,
    plot_pattern_event,
    plot_patterns_overview,
)
from fundcloud.plots.plotly import (
    composition,
    cumulative,
    drawdown,
    monthly_heatmap,
    return_distribution,
    rolling_sharpe,
    yearly_returns_bars,
)
from fundcloud.plots.themes import get_theme, set_theme

__all__ = [
    "composition",
    "cumulative",
    "drawdown",
    "get_theme",
    "monthly_heatmap",
    "plot_asset_patterns",
    "plot_pattern_event",
    "plot_patterns_overview",
    "return_distribution",
    "rolling_sharpe",
    "set_theme",
    "summary",
    "yearly_returns_bars",
]
