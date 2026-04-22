"""Tests for :mod:`fundcloud.plots`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fundcloud.plots import plotly as plt_plot


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.DatetimeIndex(pd.date_range("2023-01-02", periods=250, freq="B").values)
    return pd.Series(rng.normal(0.0005, 0.01, 250), index=idx, name="strategy")


def test_cumulative_returns_figure(returns: pd.Series) -> None:
    fig = plt_plot.cumulative(returns)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_cumulative_with_benchmark(returns: pd.Series) -> None:
    bench = returns.rename("benchmark") * 0.5
    fig = plt_plot.cumulative(returns, benchmark=bench)
    assert len(fig.data) == 2


def test_drawdown_returns_nonpositive(returns: pd.Series) -> None:
    fig = plt_plot.drawdown(returns)
    ys = fig.data[0].y
    assert all((y is None) or y <= 1e-9 for y in ys)


def test_rolling_sharpe_has_expected_length(returns: pd.Series) -> None:
    fig = plt_plot.rolling_sharpe(returns, window=30)
    assert len(fig.data[0].y) == len(returns)


def test_monthly_heatmap_builds(returns: pd.Series) -> None:
    fig = plt_plot.monthly_heatmap(returns)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"


def test_return_distribution(returns: pd.Series) -> None:
    fig = plt_plot.return_distribution(returns)
    assert fig.data[0].type == "histogram"


def test_composition_stackplot() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D").values)
    weights = pd.DataFrame(
        {"A": np.linspace(0.5, 0.7, 10), "B": np.linspace(0.5, 0.3, 10)}, index=idx
    )
    fig = plt_plot.composition(weights)
    assert len(fig.data) == 2


def test_matplotlib_builders_import_only_when_called() -> None:
    # Importing the mpl module should always work — mpl is behind [viz] but
    # present in dev deps. Invoke cumulative() to prove the path.
    from fundcloud.plots import mpl as mpl_plot

    r = pd.Series([0.01, 0.0, -0.005], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    fig = mpl_plot.cumulative(r)
    assert hasattr(fig, "savefig")


def test_cumulative_y_axis_is_percentage(returns: pd.Series) -> None:
    """Cumulative trace must be ``(1+r).cumprod() - 1`` with tickformat ``.0%``."""
    fig = plt_plot.cumulative(returns)
    assert fig.layout.yaxis.tickformat == ".0%"
    # First data point is the first bar's return — close to 0, not 1.0.
    assert abs(float(fig.data[0].y[0])) < 0.05


def test_yearly_returns_bars_traces_and_tickformat(returns: pd.Series) -> None:
    bench = returns.rename("bench") * 0.5
    fig = plt_plot.yearly_returns_bars(returns, benchmark=bench)
    assert fig.layout.yaxis.tickformat == ".0%"
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[1].type == "bar"


def test_yearly_returns_bars_no_benchmark(returns: pd.Series) -> None:
    fig = plt_plot.yearly_returns_bars(returns)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
