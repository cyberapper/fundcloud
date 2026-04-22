"""Tests for :func:`fundcloud.plots.summary`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fundcloud.plots.aggregated import summary


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(3)
    idx = pd.DatetimeIndex(pd.date_range("2022-06-01", periods=500, freq="B").values)
    return pd.Series(rng.normal(0.0005, 0.011, 500), index=idx, name="strategy")


@pytest.fixture
def weights(returns: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = len(returns)
    a = 0.6 + 0.05 * rng.normal(size=n).cumsum() / np.sqrt(n)
    a = np.clip(a, 0.2, 0.9)
    return pd.DataFrame({"stocks": a, "bonds": 1.0 - a}, index=returns.index)


def test_summary_returns_figure_with_five_panels(returns: pd.Series) -> None:
    fig = summary(returns)
    assert isinstance(fig, go.Figure)
    # 1 (cumulative) + 1 (drawdown) + 1 (rolling sharpe) + 1 (distribution) + 1 (heatmap)
    # Each panel contributes at least one trace; the cumulative may produce
    # multiple for benchmark overlay but there's no benchmark here.
    assert len(fig.data) >= 5


def test_summary_with_weights_adds_composition_row(
    returns: pd.Series, weights: pd.DataFrame
) -> None:
    without = summary(returns)
    with_weights = summary(returns, weights=weights)
    # Composition row adds two traces (one per asset), reflected in the trace count.
    assert len(with_weights.data) == len(without.data) + 2


def test_summary_with_benchmark_adds_panels(returns: pd.Series) -> None:
    """Benchmark = one overlay trace on cumulative + rolling α and rolling β panels."""
    bench = (returns * 0.6).rename("benchmark")
    without = summary(returns)
    with_bench = summary(returns, benchmark=bench)
    # +1 benchmark trace on cumulative, +1 rolling alpha trace, +1 rolling beta trace.
    assert len(with_bench.data) == len(without.data) + 3

    # The rolling alpha + rolling beta subplot titles must appear.
    titles = {a.text for a in with_bench.layout.annotations if a.text}
    assert "Rolling alpha (annualised)" in titles
    assert "Rolling beta" in titles


def test_summary_theme_applies_plotly_template(returns: pd.Series) -> None:
    fig = summary(returns, theme="dark")
    assert fig.layout.template is not None


def test_summary_accepts_dataframe_input() -> None:
    rng = np.random.default_rng(4)
    idx = pd.DatetimeIndex(pd.date_range("2023-01-02", periods=260, freq="B").values)
    df = pd.DataFrame(
        {
            "alpha": rng.normal(0.0006, 0.010, 260),
            "beta": rng.normal(0.0003, 0.013, 260),
        },
        index=idx,
    )
    fig = summary(df)
    # cumulative alone adds 2 traces (one per column).
    assert len(fig.data) >= 6  # 2 cumulative + 2 dd + 2 rs + 2 dist + 1 heatmap (first col)


def test_summary_custom_title(returns: pd.Series) -> None:
    fig = summary(returns, title="My Strategy 2024")
    assert fig.layout.title.text == "My Strategy 2024"


def test_mpl_summary_returns_matplotlib_figure(
    returns: pd.Series, weights: pd.DataFrame
) -> None:
    from fundcloud.plots import mpl as mpl_plots

    fig = mpl_plots.summary(returns, weights=weights, title="test")
    assert hasattr(fig, "savefig")
    # Canonical layout: 3 rows + composition row = 4 rows, with 6 axes
    # (cumulative spans both cols, composition spans both cols).
    assert len(fig.axes) >= 6  # include colorbar axes


def test_summary_heatmap_renders_all_years() -> None:
    """Regression — the heatmap used to collapse to a single-row strip in the
    composite. We verify the trace still carries every year from the data
    and that the y-axis is a category axis with reversed autorange."""
    rng = np.random.default_rng(11)
    idx = pd.DatetimeIndex(pd.date_range("2014-09-17", periods=2_600, freq="D").values)
    s = pd.Series(rng.normal(0.002, 0.04, 2_600), index=idx, name="BTC-USD")
    fig = summary(s)
    heatmap_trace = next(t for t in fig.data if t.type == "heatmap")
    # At least 7 calendar years in the sample window.
    assert len(heatmap_trace.y) >= 7
    yaxis_id = heatmap_trace.yaxis  # e.g. "y5"
    yaxis_layout = fig.layout[f"{'yaxis' if yaxis_id == 'y' else 'yaxis' + yaxis_id[1:]}"]
    assert yaxis_layout.type == "category"
    assert yaxis_layout.autorange == "reversed"
