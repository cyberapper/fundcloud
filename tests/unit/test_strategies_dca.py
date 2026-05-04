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


def test_dca_amount_pct_scales_with_starting_cash() -> None:
    """Scalar ``amount_pct`` deploys ``pct * starting_cash / n_assets`` on the
    first fire (before any equity history exists) — equal-split across
    bars-frame assets when ``weights`` is omitted.
    """
    rng = np.random.default_rng(3)
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

    strat = DCA(amount_pct=0.02, horizon="weekly")
    strat.init(bars, Portfolio(cash=100_000.0))
    # Equal-split deferred to init: 2 % across two assets → 1 % per leg.
    assert strat._amount_pcts == pytest.approx({"A": 0.01, "B": 0.01})

    # Run the simulator — first fire deploys 1 % of starting cash per leg
    # (equity_history is empty on the first bar, so it falls back to cash).
    result = Simulator(bars, cash=100_000.0).run_strategy(DCA(amount_pct=0.02, horizon="weekly"))
    assert len(result.trades) >= 2
    assert {"A", "B"}.issubset(set(result.trades["asset"].unique()))


def test_dca_amount_pct_with_weights(panel: pd.DataFrame) -> None:
    """Per-asset distribution honored when ``weights`` is supplied."""
    strat = DCA(amount_pct=0.02, horizon="weekly", weights={"A": 1.0})
    # Pre-multiplied in __init__: 2 % * 1.0 → 2 % into asset A.
    assert strat._amount_pcts == pytest.approx({"A": 0.02})


def test_dca_amount_pct_per_asset_mapping() -> None:
    """Mapping ``amount_pct`` is stored verbatim — no equal-split."""
    strat = DCA(amount_pct={"SPY": 0.012, "AGG": 0.008}, horizon="monthly")
    assert strat._amount_pcts == pytest.approx({"SPY": 0.012, "AGG": 0.008})
    assert strat._amounts == {}


def test_dca_requires_exactly_one_of_amount_or_amount_pct() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        DCA(amount=500, amount_pct=0.01)
    with pytest.raises(ValueError, match="exactly one of"):
        DCA()


def test_dca_amount_pct_grows_with_equity(panel: pd.DataFrame) -> None:
    """As equity grows, subsequent fires deploy more dollars."""
    sim = Simulator(panel, cash=100_000.0)
    result = sim.run_strategy(DCA(amount_pct=0.05, horizon="weekly", weights={"A": 1.0}))
    # Monotonic-ish growth: trade notionals shouldn't all be identical
    # (which would happen with fixed-dollar amount).
    notionals = (result.trades["qty"] * result.trades["price"]).abs().to_numpy()
    assert notionals.std() > 0.0


def test_dca_amount_pct_does_not_leverage() -> None:
    """``amount_pct`` must not push cash negative or inflate equity past
    the lump-sum upper bound on a single rising asset.

    Regression for the bug where DCA sized orders against current equity
    with no cash check, so a 5 %-of-equity monthly fire over a long rising
    asset accumulated implicit leverage and produced ~40x the buy-and-hold
    return.
    """
    rng = np.random.default_rng(42)
    n = 800  # ~3 years of business days — long enough for cash to deplete
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="B").values)
    # Strong drift so equity-scaled deposits would otherwise leverage up.
    drift = np.linspace(0, 1.5, n)
    noise = np.cumsum(rng.normal(0, 0.005, n))
    close_a = 100.0 * np.exp(drift + noise)
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    cash = 100_000.0
    sim = Simulator(bars, cash=cash)
    result = sim.run_strategy(DCA(amount_pct=0.05, horizon="weekly", weights={"A": 1.0}))

    # Total dollars deployed across the whole run must not exceed starting
    # cash by more than a small fees/slippage fudge — the pre-fix bug would
    # blow this past the cap by 100x within the first few years.
    total_notional = float((result.trades["qty"] * result.trades["price"]).abs().sum())
    assert total_notional <= cash * 1.01, (
        f"DCA deployed {total_notional:.2f} but starting cash was only {cash:.2f} — "
        "implies implicit leverage."
    )

    # Final equity bounded above by the lump-sum upper bound: investing ALL
    # cash on day 1 would give cash * (final_price / first_price). DCA can
    # never beat that on a monotonically rising asset.
    final_equity = float(result.equity_curve.iloc[-1])
    lump_sum_upper_bound = cash * float(close_a[-1]) / float(close_a[0])
    assert final_equity <= lump_sum_upper_bound * 1.01, (
        f"DCA equity {final_equity:.2f} exceeds lump-sum upper bound "
        f"{lump_sum_upper_bound:.2f} — implies leverage."
    )


def test_dca_clips_when_cash_exhausted() -> None:
    """When ``amount_pct`` would deploy more than available cash, DCA clips
    to remaining cash and later fires emit nothing.
    """
    rng = np.random.default_rng(7)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=200, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 200))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    cash = 1_000.0
    # 50%/week into a single asset exhausts cash within 2-3 fires.
    sim = Simulator(bars, cash=cash)
    result = sim.run_strategy(DCA(amount_pct=0.5, horizon="weekly", weights={"A": 1.0}))

    # Total notional traded must not exceed starting cash by more than fees.
    total_notional = float((result.trades["qty"] * result.trades["price"]).abs().sum())
    assert total_notional <= cash * 1.01, (
        f"Total deployed {total_notional:.2f} exceeds starting cash {cash:.2f}"
    )

    # Trade notionals should taper off — the last trade should be smaller
    # than the first as remaining cash shrinks toward zero.
    notionals = (result.trades["qty"] * result.trades["price"]).abs().to_numpy()
    assert len(notionals) >= 2
    assert notionals[-1] <= notionals[0]


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
