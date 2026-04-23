"""Tests for ``fundcloud.metrics.core``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.metrics import core as M


def test_sharpe_scalar_vs_panel(returns_series: pd.Series, returns_panel: pd.DataFrame) -> None:
    s = M.sharpe(returns_series)
    assert isinstance(s, float)
    assert np.isfinite(s)

    p = M.sharpe(returns_panel)
    assert isinstance(p, pd.Series)
    assert list(p.index) == list(returns_panel.columns)


def test_sharpe_zero_vol_gives_nan() -> None:
    # Exact-zero returns → 0/0 → NaN. (For a constant non-zero series, floating
    # point noise in std() gives a huge finite sharpe; that's expected behaviour.)
    s = pd.Series([0.0] * 10, index=pd.date_range("2023-01-01", periods=10))
    assert np.isnan(M.sharpe(s))


def test_sortino_only_penalises_downside() -> None:
    up_only = pd.Series(
        [0.01, 0.02, 0.005, 0.015],
        index=pd.date_range("2023-01-01", periods=4),
    )
    # Zero downside deviation → NaN by convention.
    assert np.isnan(M.sortino(up_only))


def test_drawdown_is_non_positive(returns_series: pd.Series) -> None:
    dd = M.drawdown_series(returns_series)
    assert (dd <= 1e-12).all()


def test_max_drawdown_matches_manual_calculation() -> None:
    r = pd.Series(
        [0.10, 0.10, -0.50, 0.05],
        index=pd.date_range("2023-01-01", periods=4),
    )
    # wealth = [1.1, 1.21, 0.605, 0.635], peak = 1.21, trough = 0.605
    # max_drawdown = 0.605 / 1.21 - 1 = -0.5
    assert np.isclose(M.max_drawdown(r), -0.5)


def test_cvar_rejects_invalid_alpha(returns_series: pd.Series) -> None:
    with pytest.raises(ValueError):
        M.cvar(returns_series, alpha=0.0)
    with pytest.raises(ValueError):
        M.cvar(returns_series, alpha=1.0)


def test_cvar_is_at_most_var(returns_series: pd.Series) -> None:
    var = M.value_at_risk(returns_series, alpha=0.95)
    es = M.cvar(returns_series, alpha=0.95)
    # Expected shortfall (average of worst losses) ≤ VaR (threshold).
    assert es <= var + 1e-12


def test_omega_ratio_positive(returns_series: pd.Series) -> None:
    assert M.omega(returns_series) > 0


def test_returns_stats_is_panel_shape(returns_panel: pd.DataFrame) -> None:
    stats = M.returns_stats(returns_panel)
    expected_metrics = {
        "periods",
        "total_return",
        "cagr",
        "ann_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "ulcer_index",
        "cvar",
        "omega",
    }
    assert expected_metrics.issubset(set(stats.index))
    assert list(stats.columns) == list(returns_panel.columns)
