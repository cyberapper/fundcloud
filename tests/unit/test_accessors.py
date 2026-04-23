"""Tests for the ``.fc`` pandas accessor."""

from __future__ import annotations

import fundcloud  # noqa: F401 — registers the accessor
import numpy as np
import pandas as pd


def test_series_accessor_registered() -> None:
    s = pd.Series([0.01, -0.02, 0.015, 0.005])
    assert hasattr(s, "fc")


def test_dataframe_accessor_registered() -> None:
    df = pd.DataFrame({"A": [0.01, -0.02], "B": [0.005, 0.01]})
    assert hasattr(df, "fc")


def test_series_sharpe_matches_free_function(returns_series: pd.Series) -> None:
    from fundcloud.metrics import core as M

    assert np.isclose(returns_series.fc.sharpe(), M.sharpe(returns_series))


def test_series_all_metrics_callable(returns_series: pd.Series) -> None:
    s = returns_series
    assert np.isfinite(s.fc.sharpe())
    assert np.isfinite(s.fc.sortino())
    assert np.isfinite(s.fc.calmar())
    assert np.isfinite(s.fc.omega())
    assert s.fc.max_drawdown() <= 0
    assert len(s.fc.drawdown_series()) == len(s)
    assert s.fc.cvar() <= s.fc.value_at_risk()


def test_dataframe_summary(returns_panel: pd.DataFrame) -> None:
    stats = returns_panel.fc.summary()
    assert "sharpe" in stats.index
    assert list(stats.columns) == list(returns_panel.columns)


def test_dataframe_to_returns(ohlcv_panel: pd.DataFrame) -> None:
    ret = ohlcv_panel.fc.to_returns()
    assert isinstance(ret, pd.DataFrame)
    assert len(ret) == len(ohlcv_panel) - 1  # first row dropped


def test_render_html_rejects_self_benchmark_when_single_column(tmp_path) -> None:
    """Regression: ``returns.fc.render_html(benchmark='NQ=F')`` used to hit
    a ``ZeroDivisionError`` in ``portfolio_from_frame`` when the returns
    frame had only the benchmark column. Now raises a clear ``ValueError``."""
    import pytest

    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"NQ=F": rng.normal(0, 0.01, 60)}, index=idx)
    with pytest.raises(ValueError, match="only column"):
        returns.fc.render_html(tmp_path / "x.html", benchmark="NQ=F")


def test_portfolio_from_frame_rejects_zero_columns() -> None:
    """Direct callers hitting a 0-column frame get a clear error, not a
    cryptic ``ZeroDivisionError``."""
    import pytest
    from fundcloud.accessors._helpers import portfolio_from_frame

    empty = pd.DataFrame(index=pd.date_range("2024-01-02", periods=10, freq="B"))
    with pytest.raises(ValueError, match="zero columns"):
        portfolio_from_frame(empty)
