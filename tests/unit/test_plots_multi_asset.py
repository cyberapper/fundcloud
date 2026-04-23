"""Tests: plot builders accept pandas DataFrames and overlay one trace per column."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fundcloud.plots import plotly as plt_plot


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.DatetimeIndex(pd.date_range("2023-01-02", periods=200, freq="B").values)
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.0005, 0.010, 200),
            "QQQ": rng.normal(0.0006, 0.013, 200),
            "AGG": rng.normal(0.0001, 0.003, 200),
        },
        index=idx,
    )


def test_cumulative_dataframe_input_renders_one_trace_per_column(returns_df: pd.DataFrame) -> None:
    fig = plt_plot.cumulative(returns_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    names = {trace.name for trace in fig.data}
    assert names == {"SPY", "QQQ", "AGG"}


def test_drawdown_dataframe_input_renders_one_trace_per_column(returns_df: pd.DataFrame) -> None:
    fig = plt_plot.drawdown(returns_df)
    assert len(fig.data) == 3
    # All traces should be non-positive (drawdown is <= 0).
    for trace in fig.data:
        ys = [y for y in trace.y if y is not None]
        assert max(ys) <= 1e-9


def test_rolling_sharpe_dataframe_input_renders_one_trace_per_column(
    returns_df: pd.DataFrame,
) -> None:
    fig = plt_plot.rolling_sharpe(returns_df, window=30)
    assert len(fig.data) == 3
    for trace in fig.data:
        assert len(trace.y) == len(returns_df)


def test_return_distribution_dataframe_input_overlays_histograms(
    returns_df: pd.DataFrame,
) -> None:
    fig = plt_plot.return_distribution(returns_df)
    assert len(fig.data) == 3
    for trace in fig.data:
        assert trace.type == "histogram"
    assert fig.layout.barmode == "overlay"


def test_monthly_heatmap_raises_on_multi_column_dataframe(returns_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="requires a single series"):
        plt_plot.monthly_heatmap(returns_df)


def test_monthly_heatmap_squeezes_single_column_dataframe(returns_df: pd.DataFrame) -> None:
    fig = plt_plot.monthly_heatmap(returns_df[["SPY"]])
    assert fig.data[0].type == "heatmap"


def test_annotations_true_adds_stats_pill(returns_df: pd.DataFrame) -> None:
    fig = plt_plot.cumulative(returns_df, annotations=True)
    # Annotation added by the stats pill helper.
    annotation_texts = [ann.text for ann in fig.layout.annotations]
    assert any("CAGR" in text for text in annotation_texts if text)


def test_annotations_false_leaves_figure_bare(returns_df: pd.DataFrame) -> None:
    fig = plt_plot.cumulative(returns_df, annotations=False)
    annotation_texts = [ann.text for ann in fig.layout.annotations]
    assert all("CAGR" not in (text or "") for text in annotation_texts)
