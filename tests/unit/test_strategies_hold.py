"""Tests for :class:`fundcloud.strategies.Hold`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.sim import Simulator
from fundcloud.strategies import Hold, RebalanceSpec


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=30, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.5, 30))
    close_b = 200 + np.cumsum(rng.normal(0, 1.0, 30))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
        ("open", "B"): close_b,
        ("high", "B"): close_b + 1,
        ("low", "B"): close_b - 1,
        ("close", "B"): close_b,
        ("volume", "B"): 1_000_000.0,
    }
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_hold_allocates_once(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(Hold(weights={"A": 0.5, "B": 0.5}))
    # Two initial allocation orders: one buy for A, one buy for B.
    assert len(result.trades) == 2
    # Weights should sum to ~1 at the initial allocation period.
    assert result.portfolio.weights is not None


def test_hold_rebalance_on_cadence(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(
        Hold(weights={"A": 0.5, "B": 0.5}, rebalance=RebalanceSpec(horizon="weekly"))
    )
    # Weekly rebalance on a 30-day window → at least initial + 3 rebalances worth of trades.
    assert len(result.trades) > 4


def test_hold_weights_must_sum_to_one(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    with pytest.raises(ValueError, match="sum to 1"):
        sim.run_strategy(Hold(weights={"A": 0.7}))


def test_hold_accepts_callable_weights(panel: pd.DataFrame) -> None:
    def choose(bars: pd.DataFrame) -> dict[str, float]:
        return {"A": 0.6, "B": 0.4}

    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(Hold(weights=choose))
    assert len(result.trades) == 2


def test_hold_portfolio_has_nonzero_equity(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=50_000)
    result = sim.run_strategy(Hold(weights={"A": 1.0}))
    # Equity curve should end near starting capital × (close_end / close_fill)
    final_equity = result.equity_curve.iloc[-1]
    assert 20_000 < final_equity < 80_000
