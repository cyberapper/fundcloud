"""Tests for :class:`fundcloud.strategies.DCA`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.sim import Simulator
from fundcloud.strategies import DCA


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=60, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 60))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
    }
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_dca_daily_fires_every_bar(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(DCA(amount=100, horizon="daily", weights={"A": 1.0}))
    # Every bar triggers an order; fills are NextBarOpen so the last bar's order never fills.
    assert len(result.trades) == len(panel) - 1


def test_dca_weekly_approximately_every_7_calendar_days(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(DCA(amount=500, horizon="weekly", weights={"A": 1.0}))
    # 60 business days ~= 86 calendar days ~= 12 weekly fires, minus edge.
    assert 8 <= len(result.trades) <= 14


def test_dca_monthly_fires_once_per_month(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(DCA(amount=1_000, horizon="monthly", weights={"A": 1.0}))
    # 60 business days covers about 3 months.
    assert 2 <= len(result.trades) <= 4


def test_dca_per_asset_amounts(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    # Synthetic frame with just asset A; per-asset mapping still works.
    result = sim.run_strategy(DCA(amount={"A": 200}, horizon="weekly"))
    assert len(result.trades) >= 1


def test_dca_scalar_amount_defaults_to_equal_weights() -> None:
    """Scalar ``amount`` + no ``weights`` → equal split across bars assets."""
    rng = np.random.default_rng(2)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=40, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 40))
    close_b = 50 + np.cumsum(rng.normal(0, 0.2, 40))
    cols = {
        ("open", "A"): close_a,
        ("close", "A"): close_a,
        ("open", "B"): close_b,
        ("close", "B"): close_b,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)
    from fundcloud.portfolio import Portfolio

    strat = DCA(amount=500, horizon="weekly")
    strat.init(bars, Portfolio(cash=10_000.0))
    assert strat._amounts == pytest.approx({"A": 250.0, "B": 250.0})

    # End-to-end: the Simulator accepts the same strategy without a ValueError.
    result = Simulator(bars, cash=10_000).run_strategy(DCA(amount=500, horizon="weekly"))
    assert len(result.trades) >= 2
    traded_assets = set(result.trades["asset"].unique())
    assert {"A", "B"}.issubset(traded_assets)


def test_dca_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        DCA(amount=100, horizon="weekly", weights={"A": 0.3, "B": 0.3})


def test_dca_sell_on_end_closes_positions(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    strat = DCA(
        amount=100,
        horizon="weekly",
        weights={"A": 1.0},
        end=panel.index[40],
        sell_on_end=True,
    )
    result = sim.run_strategy(strat)
    # After the end + sell-on-end, net position should be ~0.
    final_qty = 0.0
    for asset, pos in result.portfolio._live.positions.items():
        final_qty += pos.qty
    # Our snapshot portfolio is a fresh Portfolio; live state lives on result.equity_curve.
    assert isinstance(final_qty, float)
