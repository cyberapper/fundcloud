"""Benchmark-aware plotly figures used only when ``Tearsheet.benchmark`` is set.

Kept in the reports package rather than ``fundcloud.plots`` because these
figures are report-specific composites; the generic plotting layer stays
single-concern.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fundcloud.metrics import rolling_alpha, rolling_beta

__all__ = ["rolling_alpha_beta_figure"]


def rolling_alpha_beta_figure(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
) -> go.Figure:
    """Rolling alpha (top) and rolling beta (bottom) vs ``benchmark``.

    Displayed as two stacked panels sharing the x-axis so readers can
    spot regime shifts (beta creep, alpha erosion) in one glance.
    """
    alpha = rolling_alpha(returns, benchmark, window=window, periods_per_year=periods_per_year)
    beta = rolling_beta(returns, benchmark, window=window)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Rolling alpha ({window}-bar, annualised)",
            f"Rolling beta ({window}-bar)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=alpha.index, y=alpha.values, name="alpha", line={"width": 1.6}),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0, line={"color": "rgba(0,0,0,0.4)", "width": 1, "dash": "dot"}, row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=beta.index, y=beta.values, name="beta", line={"width": 1.6}),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=1, line={"color": "rgba(0,0,0,0.4)", "width": 1, "dash": "dot"}, row=2, col=1
    )
    fig.update_yaxes(tickformat=".2%", row=1, col=1)
    fig.update_yaxes(tickformat=".2f", row=2, col=1)
    fig.update_layout(
        height=460,
        showlegend=False,
        margin={"l": 60, "r": 40, "t": 50, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Inter, system-ui, sans-serif", "size": 12},
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.2)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.2)")
    return fig
