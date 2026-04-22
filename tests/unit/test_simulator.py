"""End-to-end Simulator tests across the four entry points."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.sim import (
    FixedBps,
    NoCost,
    SameBarClose,
    SimResult,
    Simulator,
)
from fundcloud.strategies import Hold


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=30, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.5, 30))
    close_b = 50 + np.cumsum(rng.normal(0, 0.3, 30))
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


# -------------------------------------------------------------------- run_strategy


def test_run_strategy_returns_simresult(panel: pd.DataFrame) -> None:
    result = Simulator(panel, cash=100_000).run_strategy(Hold(weights={"A": 1.0}))
    assert isinstance(result, SimResult)
    assert not result.trades.empty
    assert not result.equity_curve.empty


def test_run_strategy_next_bar_open_default(panel: pd.DataFrame) -> None:
    """Default execution is NextBarOpen → first fill at bar 1."""
    result = Simulator(panel, cash=100_000).run_strategy(Hold(weights={"A": 1.0}))
    first_trade_ts = result.trades["ts"].iloc[0]
    assert first_trade_ts == panel.index[1]


def test_run_strategy_same_bar_close(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000, execution=SameBarClose(), costs=NoCost())
    result = sim.run_strategy(Hold(weights={"A": 1.0}))
    first_trade_ts = result.trades["ts"].iloc[0]
    assert first_trade_ts == panel.index[0]


# -------------------------------------------------------------------- run_weights


def test_run_weights_rebalances_to_targets(panel: pd.DataFrame) -> None:
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[panel.index[0]])
    result = Simulator(panel, cash=100_000).run_weights(targets)
    # Two buys at the initial target bar.
    assert len(result.trades) >= 2


# -------------------------------------------------------------------- run_orders


def test_run_orders_executes_explicit_orders(panel: pd.DataFrame) -> None:
    explicit = pd.DataFrame([
        {"ts": panel.index[2], "asset": "A", "side": "buy", "qty": 100.0},
        {"ts": panel.index[10], "asset": "A", "side": "sell", "qty": 50.0},
    ])
    result = Simulator(panel, cash=100_000).run_orders(explicit)
    assert len(result.trades) == 2
    assert result.trades["qty"].iloc[0] > 0
    assert result.trades["qty"].iloc[1] < 0


def test_run_orders_rejects_missing_columns(panel: pd.DataFrame) -> None:
    bad = pd.DataFrame([{"ts": panel.index[0], "asset": "A"}])
    with pytest.raises(KeyError, match="missing"):
        Simulator(panel).run_orders(bad)


# -------------------------------------------------------------------- run_signals


def test_run_signals_emits_buy_on_entry(panel: pd.DataFrame) -> None:
    entries = pd.DataFrame(False, index=panel.index, columns=["A"])
    exits = pd.DataFrame(False, index=panel.index, columns=["A"])
    entries.iloc[5, 0] = True
    exits.iloc[15, 0] = True
    result = Simulator(panel, cash=100_000).run_signals(entries, exits, size=0.5)
    assert len(result.trades) == 2
    assert result.trades["qty"].iloc[0] > 0
    assert result.trades["qty"].iloc[1] < 0


# -------------------------------------------------------------------- costs


def test_costs_subtract_from_cash(panel: pd.DataFrame) -> None:
    sim_no_cost = Simulator(panel, cash=100_000, costs=NoCost())
    sim_with_cost = Simulator(panel, cash=100_000, costs=FixedBps(bps=50))
    result_no = sim_no_cost.run_strategy(Hold(weights={"A": 1.0}))
    result_yes = sim_with_cost.run_strategy(Hold(weights={"A": 1.0}))
    # With 50 bps fee, ending equity should be strictly less.
    assert result_yes.equity_curve.iloc[-1] < result_no.equity_curve.iloc[-1]


# -------------------------------------------------------------------- metrics pass-through


def test_simresult_summary_works(panel: pd.DataFrame) -> None:
    result = Simulator(panel, cash=100_000).run_strategy(Hold(weights={"A": 1.0}))
    stats = result.metrics()
    assert "sharpe" in stats.index


def test_simresult_summary_and_metrics_differ(panel: pd.DataFrame) -> None:
    """``summary()`` is the compact 11-metric view; ``metrics()`` is the
    full ~55-metric bundle. They must not return the same Series."""
    result = Simulator(panel, cash=100_000).run_strategy(Hold(weights={"A": 1.0}))
    summary = result.summary()
    full = result.metrics()
    # Full bundle is meaningfully larger than the compact view.
    assert len(full) >= 2 * len(summary)
    # Rows only present in the full bundle (and not in the compact summary).
    full_only = {"skew", "kurtosis", "tail_ratio", "profit_factor"}
    assert full_only.issubset(set(full.index))
    assert not full_only.intersection(set(summary.index))


def test_simresult_bars_preserved(panel: pd.DataFrame) -> None:
    result = Simulator(panel, cash=100_000).run_strategy(Hold(weights={"A": 1.0}))
    assert len(result.bars) == len(panel)


def test_simresult_pf_is_portfolio_alias(panel: pd.DataFrame) -> None:
    """``result.pf`` is the same object as ``result.portfolio``."""
    result = Simulator(panel, cash=100_000).run_strategy(Hold(weights={"A": 1.0}))
    assert result.pf is result.portfolio
    # And the canonical chain works: result.pf.sharpe()
    assert isinstance(result.pf.sharpe(), float)


# ----------- mixed-frequency panels (e.g. crypto 7-day + equity 5-day) ---


def _mixed_freq_panel() -> pd.DataFrame:
    """BTC (7-day) + EQ (5-day) — simulates real YF data for a crypto+equity mix.

    Starts on a Friday so bar 1 (the next-bar-open fill under the default
    execution model) lands on a Saturday when EQ's open is NaN.
    """
    idx = pd.date_range("2024-01-05", periods=60, freq="D")  # Friday
    rng = np.random.default_rng(0)
    btc = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 60)))
    eq = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 60)))
    is_biz = pd.Series(idx, index=idx).apply(lambda d: d.weekday() < 5)
    eq_nan = np.where(is_biz.values, eq, np.nan)
    rows = {
        ("open", "BTC"): btc,
        ("high", "BTC"): btc * 1.01,
        ("low", "BTC"): btc * 0.99,
        ("close", "BTC"): btc,
        ("volume", "BTC"): 1_000_000.0,
        ("open", "EQ"): eq_nan,
        ("high", "EQ"): eq_nan * 1.01,
        ("low", "EQ"): eq_nan * 0.99,
        ("close", "EQ"): eq_nan,
        ("volume", "EQ"): 1_000_000.0,
    }
    df = pd.DataFrame(rows, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_hold_fills_both_legs_despite_mixed_frequency_nan() -> None:
    """Regression: 50/50 hold must actually place BOTH legs even when the
    fill bar has NaN open for one asset (weekend for a 5-day equity when
    paired with a 7-day crypto)."""
    panel = _mixed_freq_panel()
    result = Simulator(panel, cash=1_000_000).run_strategy(Hold(weights={"BTC": 0.5, "EQ": 0.5}))
    # Exactly two trades — one per leg. Both must have filled.
    assets_filled = set(result.trades["asset"])
    assert assets_filled == {"BTC", "EQ"}, (
        f"hold dropped a leg on a NaN bar; trades filled for {assets_filled}"
    )


def test_mark_to_market_uses_last_price_on_nan_bar() -> None:
    """Regression: an EQ position marked to market on a Saturday (NaN
    close) must use the last known EQ close, not drop to zero value."""
    panel = _mixed_freq_panel()
    result = Simulator(panel, cash=1_000_000).run_strategy(Hold(weights={"BTC": 0.5, "EQ": 0.5}))
    equity = result.equity_curve
    # No single-bar >30% drop: such a jump would indicate the portfolio
    # zeroed out a leg on a NaN-price bar (the old bug).
    pct_change = equity.pct_change().abs()
    max_daily = float(pct_change.max())
    assert max_daily < 0.30, (
        f"mark-to-market produced a {max_daily:.1%} one-bar swing; the "
        "portfolio likely lost a leg to a NaN close."
    )


def test_hold_return_tracks_component_average() -> None:
    """Regression: 50/50 hold total return must land near the arithmetic
    mean of each asset's component return, within a reasonable fee /
    timing tolerance."""
    panel = _mixed_freq_panel()
    result = Simulator(panel, cash=1_000_000).run_strategy(Hold(weights={"BTC": 0.5, "EQ": 0.5}))
    # Per-asset component returns from the first valid fill bar forward.
    btc_closes = panel[("close", "BTC")].dropna()
    eq_closes = panel[("close", "EQ")].dropna()
    btc_ret = float(btc_closes.iloc[-1] / btc_closes.iloc[1] - 1.0)
    eq_ret = float(eq_closes.iloc[-1] / eq_closes.iloc[1] - 1.0)
    expected = 0.5 * (1 + btc_ret) + 0.5 * (1 + eq_ret) - 1
    actual = float(result.equity_curve.iloc[-1] / 1_000_000 - 1)
    # Tolerance: 3% absolute covers tiny fee drag + the <1-day timing gap
    # between the component's own first trading bar and the simulator's
    # next-bar-open fill.
    assert abs(actual - expected) < 0.03, (
        f"50/50 hold returned {actual:+.2%} but components averaged {expected:+.2%}"
    )
