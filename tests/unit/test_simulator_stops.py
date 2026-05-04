"""Intra-bar stop-loss tests for ``Simulator._check_intrabar_exits``.

The simulator carries an ``sl_level`` per :class:`Position`, set when an
entry :class:`Order` carries an ``sl_stop`` fraction. On every bar the
simulator checks the bar's high/low against the stop level *before* the
strategy is asked to decide; a breach synthesises a forced exit at the
stop price (or the bar's open on a gap), with slippage and costs applied
the same as a market exit.

Each test engineers a tiny OHLCV panel with explicit high/low values so
the stop-firing condition is deterministic, then drives the simulator
with a one-shot ``BaseStrategy`` that opens the position with the
desired ``sl_stop`` and never trades again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.sim import (
    FixedBps,
    HalfSpread,
    NextBarOpen,
    NoCost,
    NoSlippage,
    Order,
    SimResult,
    Simulator,
)
from fundcloud.strategies import BaseStrategy, Context

# ----------------------------------------------------------------- helpers


def _bars_from_ohlc(
    ohlc: list[tuple[float, float, float, float]], asset: str = "BTC"
) -> pd.DataFrame:
    """Build a Bars DataFrame from a list of (open, high, low, close) tuples."""
    n = len(ohlc)
    idx = pd.date_range("2024-01-02", periods=n, freq="1D")
    o, h, low, c = zip(*ohlc, strict=True)
    cols = pd.MultiIndex.from_tuples([
        ("open", asset),
        ("high", asset),
        ("low", asset),
        ("close", asset),
        ("volume", asset),
    ])
    return pd.DataFrame(
        {
            ("open", asset): o,
            ("high", asset): h,
            ("low", asset): low,
            ("close", asset): c,
            ("volume", asset): [1.0] * n,
        },
        index=idx,
        columns=cols,
    )


class _OneShotEntry(BaseStrategy):
    """Open a single position on bar 0; never trade again.

    Lets each test isolate the simulator's stop-check behaviour from
    strategy-side logic.
    """

    def __init__(
        self,
        *,
        asset: str,
        side: str,
        qty: float,
        sl_stop: float | None,
    ) -> None:
        self.asset = asset
        self.side = side
        self.qty = qty
        self.sl_stop = sl_stop
        self._fired = False

    def decide(self, ctx: Context) -> list[Order]:
        if self._fired:
            return []
        self._fired = True
        return [
            Order(
                ts=ctx.ts,
                asset=self.asset,
                side=self.side,  # type: ignore[arg-type]
                qty=self.qty,
                sl_stop=self.sl_stop,
            )
        ]


def _run(bars: pd.DataFrame, strat: BaseStrategy, **kwargs: object) -> SimResult:
    return Simulator(
        bars,
        cash=kwargs.pop("cash", 100_000.0),
        costs=kwargs.pop("costs", NoCost()),
        slippage=kwargs.pop("slippage", NoSlippage()),
        execution=kwargs.pop("execution", NextBarOpen()),
    ).run_strategy(strat)


# ----------------------------------------------------------------- Order validation


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_order_rejects_sl_stop_outside_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="sl_stop"):
        Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, sl_stop=bad)


def test_order_accepts_sl_stop_in_unit_interval() -> None:
    order = Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, sl_stop=0.10)
    assert order.sl_stop == 0.10


def test_order_with_qty_propagates_sl_stop() -> None:
    """``Order.with_qty`` must preserve sl_stop when resolving notional → qty."""
    order = Order(
        ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", notional=1000.0, sl_stop=0.10
    )
    resolved = order.with_qty(qty=10.0)
    assert resolved.sl_stop == 0.10


# ----------------------------------------------------------------- long stops


def test_long_sl_fires_when_bar_low_pierces_level() -> None:
    """Bar 2 low pierces the 90 SL → exit at 90."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # bar 0 — strategy emits entry order (fills at bar 1 open)
        (100, 101, 99, 100),  # bar 1 — entry fills at open=100; SL set to 90
        (95, 96, 88, 92),  # bar 2 — low=88 < 90 → STOP fires at 90
        (95, 100, 90, 99),  # bar 3 — should be no-op, position is flat
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat)

    # Entry buy at bar 1 open (100), then forced sell at bar 2 stop (90).
    assert len(result.trades) == 2
    entry, stop = result.trades.iloc[0], result.trades.iloc[1]
    assert entry["qty"] == pytest.approx(10.0)
    assert entry["price"] == pytest.approx(100.0)
    assert entry["reason"] == "signal"
    assert stop["qty"] == pytest.approx(-10.0)
    assert stop["price"] == pytest.approx(90.0)
    assert stop["reason"] == "stop_loss"

    # Position is flat after the stop; subsequent bars don't re-fire.
    assert _run(bars, strat).portfolio  # smoke-rerun


def test_long_sl_does_not_fire_when_low_stays_above_level() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # entry fills; SL=90
        (98, 99, 95, 96),  # low=95 > 90 → no fire
        (96, 98, 94, 95),  # low=94 > 90 → no fire
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat)
    # Only the entry trade, no forced exit.
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["reason"] == "signal"


def test_long_sl_gap_down_fills_at_open_not_stop() -> None:
    """If the bar opens below the stop, the synthesised fill is at the open
    (worse than the stop) — realistic gap behaviour."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # entry fills; SL=90
        (85, 87, 80, 82),  # OPEN=85 already below SL=90 → fill at 85
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat)
    stop_row = result.trades[result.trades["reason"] == "stop_loss"].iloc[0]
    assert stop_row["price"] == pytest.approx(85.0)


# ----------------------------------------------------------------- short stops


class _OpenShortViaSell(BaseStrategy):
    """Open a short by selling from a flat position on bar 0; never trade again."""

    def __init__(self, *, asset: str, qty: float, sl_stop: float) -> None:
        self.asset = asset
        self.qty = qty
        self.sl_stop = sl_stop
        self._fired = False

    def decide(self, ctx: Context) -> list[Order]:
        if self._fired:
            return []
        self._fired = True
        return [
            Order(
                ts=ctx.ts,
                asset=self.asset,
                side="sell",
                qty=self.qty,
                sl_stop=self.sl_stop,
            )
        ]


def test_short_sl_fires_when_bar_high_pierces_level() -> None:
    """Short at 100 with sl_stop=0.10 → SL=110. Bar 2 high=112 → fire at 110."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal (sell)
        (100, 101, 99, 100),  # short fills at open=100; SL=110
        (105, 112, 104, 108),  # high=112 > 110 → STOP fires at 110
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat)
    assert len(result.trades) == 2
    entry, stop = result.trades.iloc[0], result.trades.iloc[1]
    assert entry["qty"] == pytest.approx(-10.0)  # short
    assert stop["qty"] == pytest.approx(10.0)  # cover
    assert stop["price"] == pytest.approx(110.0)
    assert stop["reason"] == "stop_loss"


def test_short_sl_gap_up_fills_at_open_not_stop() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # short fills; SL=110
        (115, 120, 114, 118),  # OPEN=115 already above SL=110 → fill at 115
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat)
    stop_row = result.trades[result.trades["reason"] == "stop_loss"].iloc[0]
    assert stop_row["price"] == pytest.approx(115.0)


# ----------------------------------------------------------------- accumulation + clearing


def test_sl_level_anchored_to_latest_fill_price_on_accumulation() -> None:
    """Each new entry resets the SL based on its own fill price (not avg_cost).

    Anchoring the stop to the latest fill makes the stop tighten as the
    position accumulates in an uptrend — the conservative choice for
    risk management (an avg-cost-anchored stop would drift further
    below current price as the position adds at higher levels).
    """
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")

    o1 = Order(ts=ts, asset="X", side="buy", qty=100.0, sl_stop=0.10)
    o2 = Order(ts=ts, asset="X", side="buy", qty=100.0, sl_stop=0.10)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o1, qty=100.0, price=100.0))
    pos = p.position("X")
    assert pos.avg_cost == pytest.approx(100.0)
    assert pos.sl_level == pytest.approx(90.0)  # 100 * 0.9

    # Add 100 @ 110 → avg_cost still recomputes to 105 (volume-weighted),
    # but the stop level moves to 99 (= 110 * 0.9), tighter than the
    # original 90.
    p.apply(_T(o2, qty=100.0, price=110.0))
    pos = p.position("X")
    assert pos.avg_cost == pytest.approx(105.0)
    assert pos.sl_level == pytest.approx(99.0)


def test_sl_level_cleared_after_position_closes() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # entry fills; SL=90
        (95, 96, 88, 92),  # SL fires at 90
        (95, 100, 89, 99),  # would fire again if not cleared (low<90)
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat)
    # Only one stop fill — the second bar's low is below 90 but the position
    # is already flat, so nothing fires.
    stop_rows = result.trades[result.trades["reason"] == "stop_loss"]
    assert len(stop_rows) == 1


# ----------------------------------------------------------------- slippage


def test_stop_fill_uses_slippage_model() -> None:
    """``HalfSpread`` shifts a sell-side stop fill *down* by half the spread.

    Note: the entry buy is also slipped (up by 10 bps), so ``avg_cost``
    becomes ``100 * 1.001 = 100.1`` and the stop level is
    ``100.1 * 0.9 = 90.09``. The stop fill is then ``90.09 * 0.999``.
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),
        (95, 96, 88, 92),  # bar's low pierces 90.09 → stop fires
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10)
    result = _run(bars, strat, slippage=HalfSpread(spread_bps=20.0))
    stop_row = result.trades[result.trades["reason"] == "stop_loss"].iloc[0]
    expected_price = 100.0 * 1.001 * 0.9 * 0.999
    assert stop_row["price"] == pytest.approx(expected_price, rel=1e-6)
    assert stop_row["slippage_bps"] == pytest.approx(10.0)


# ----------------------------------------------------------------- run_orders bracket support


def test_run_orders_honours_sl_stop_column() -> None:
    """``run_orders`` now wires ``sl_stop`` through to the dispatcher.

    Routes via the Python fallback today; once the Rust kernel grows
    bracket support the same path takes the Rust fast lane. The
    behaviour assertion is identical either way.
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # buy fills here at open=100; SL set to 90
        (95, 96, 88, 92),  # bar low=88 ≤ 90 → SL fires at 90
    ])
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 1.0, "sl_stop": 0.10},
    ])
    result = Simulator(bars, cash=10_000.0, costs=FixedBps(5)).run_orders(orders)
    sl_rows = result.trades[result.trades["reason"] == "stop_loss"]
    assert len(sl_rows) == 1
    assert sl_rows.iloc[0]["price"] == pytest.approx(90.0)


def test_run_orders_honours_tp_stop_column() -> None:
    """Same as above for take-profit."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # buy fills at 100; TP set to 110
        (108, 115, 107, 112),  # bar high=115 ≥ 110 → TP fires at 110
    ])
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 1.0, "tp_stop": 0.10},
    ])
    result = Simulator(bars, cash=10_000.0, costs=FixedBps(5)).run_orders(orders)
    tp_rows = result.trades[result.trades["reason"] == "take_profit"]
    assert len(tp_rows) == 1
    assert tp_rows.iloc[0]["price"] == pytest.approx(110.0)


def test_run_orders_accepts_orders_without_sl_stop_column() -> None:
    """Backward compat: existing orders DataFrames (no sl_stop column) work."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
    ])
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 1.0},
    ])
    result = Simulator(bars, cash=10_000.0, costs=FixedBps(5)).run_orders(orders)
    assert not result.trades.empty


# ----------------------------------------------------------------- final equity


def test_long_sl_caps_drawdown_versus_no_stop() -> None:
    """Same buy-and-hold path; the version with sl_stop must end with at
    least as much equity in the bad-tail bar as the unstopped one."""
    # Path: buy at 100, then close drops to 50.
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),
        (95, 96, 89, 90),  # SL at 90 fires here (low=89 pierces)
        (88, 90, 85, 87),  # close=87
        (50, 90, 45, 50),  # disaster bar
    ])
    stopped = _run(bars, _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10))
    unstopped = _run(bars, _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=None))
    assert stopped.equity_curve.iloc[-1] > unstopped.equity_curve.iloc[-1]


def test_iron_law_no_stop_means_no_trade_reason_column_diversity() -> None:
    """When no Order carries sl_stop, every Trade.reason is 'signal'."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),
        (95, 96, 88, 92),
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=None)
    result = _run(bars, strat)
    assert (result.trades["reason"] == "signal").all()


# ----------------------------------------------------------------- defensive


def test_sl_check_handles_nan_high_low() -> None:
    """A bar with NaN high/low (mixed-frequency panel) doesn't crash the stop check."""
    asset = "BTC"
    idx = pd.date_range("2024-01-02", periods=4, freq="1D")
    cols = pd.MultiIndex.from_tuples([
        ("open", asset),
        ("high", asset),
        ("low", asset),
        ("close", asset),
        ("volume", asset),
    ])
    df = pd.DataFrame(
        {
            ("open", asset): [100, 100, np.nan, 95],
            ("high", asset): [102, 101, np.nan, 98],
            ("low", asset): [99, 99, np.nan, 92],
            ("close", asset): [100, 100, np.nan, 96],
            ("volume", asset): [1.0, 1.0, 0.0, 1.0],
        },
        index=idx,
        columns=cols,
    )
    strat = _OneShotEntry(asset=asset, side="buy", qty=10.0, sl_stop=0.10)
    # Should run without raising. The NaN bar is forward-filled by the
    # simulator's price ffill, but high/low are NOT among the ffilled
    # fields if the test bypasses that — the stop check just skips bars
    # where any of open/high/low is non-finite.
    result = _run(df, strat)
    assert result is not None
