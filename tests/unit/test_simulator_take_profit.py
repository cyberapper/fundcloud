"""Intra-bar take-profit + bracket-order tests for ``Simulator``.

Mirror of :mod:`tests.unit.test_simulator_stops`. Each test engineers a
tiny OHLCV panel with explicit high/low values so the trigger condition
is deterministic, then drives the simulator with a one-shot
:class:`BaseStrategy` that opens the position with the bracket
parameters under test and never trades again.

What's exercised:

* Long / short take-profit firing on bar high / bar low respectively.
* Take-profit gap behaviour — favourable open is honoured.
* Bracket orders (both ``sl_stop`` and ``tp_stop`` set on the same
  entry).
* SL-wins arbitration when both could fire on the same bar.
* TP level recomputed on accumulating entries (latest-fill anchor).
* TP level cleared on close.
* Slippage and cost models applied to TP fills.
* Multi-asset positions with independent brackets.
* NaN-bar safety.
* ``Order.tp_stop`` validation.
* ``Order.with_qty`` propagation of ``tp_stop``.
* Iron-law: when no brackets are set, every trade has ``reason="signal"``.
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
    """Build a Bars DataFrame from a list of ``(open, high, low, close)`` tuples."""
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
    """Open a single position on bar 0 with optional brackets; never trade again."""

    def __init__(
        self,
        *,
        asset: str,
        side: str,
        qty: float,
        sl_stop: float | None = None,
        tp_stop: float | None = None,
    ) -> None:
        self.asset = asset
        self.side = side
        self.qty = qty
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
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
                tp_stop=self.tp_stop,
            )
        ]


class _OpenShortViaSell(BaseStrategy):
    """Open a short by selling from a flat position on bar 0; brackets optional."""

    def __init__(
        self,
        *,
        asset: str,
        qty: float,
        sl_stop: float | None = None,
        tp_stop: float | None = None,
    ) -> None:
        self.asset = asset
        self.qty = qty
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
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
                tp_stop=self.tp_stop,
            )
        ]


def _run(bars: pd.DataFrame, strat: BaseStrategy, **kwargs: object) -> SimResult:
    cash = kwargs.pop("cash", 100_000.0)
    costs = kwargs.pop("costs", NoCost())
    slippage = kwargs.pop("slippage", NoSlippage())
    execution = kwargs.pop("execution", NextBarOpen())
    if kwargs:
        # Surface typos like ``slippge=`` instead of swallowing them silently.
        raise TypeError(f"_run got unexpected keyword arguments: {sorted(kwargs)}")
    return Simulator(
        bars, cash=cash, costs=costs, slippage=slippage, execution=execution
    ).run_strategy(strat)


# ----------------------------------------------------------------- Order validation


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_order_rejects_non_positive_tp_stop(bad: float) -> None:
    with pytest.raises(ValueError, match="tp_stop"):
        Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, tp_stop=bad)


def test_order_accepts_tp_stop_in_unit_interval() -> None:
    order = Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, tp_stop=0.10)
    assert order.tp_stop == 0.10


def test_order_accepts_tp_stop_above_one() -> None:
    """No upper bound — TP at +200% is valid (just unusual)."""
    order = Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, tp_stop=2.0)
    assert order.tp_stop == 2.0


def test_order_accepts_bracket() -> None:
    """sl_stop and tp_stop can be set on the same entry order."""
    order = Order(
        ts=pd.Timestamp("2024-01-02"),
        asset="X",
        side="buy",
        qty=1.0,
        sl_stop=0.10,
        tp_stop=0.20,
    )
    assert order.sl_stop == 0.10
    assert order.tp_stop == 0.20


def test_order_with_qty_propagates_tp_stop() -> None:
    """``Order.with_qty`` must preserve tp_stop when resolving notional → qty."""
    order = Order(
        ts=pd.Timestamp("2024-01-02"),
        asset="X",
        side="buy",
        notional=1000.0,
        sl_stop=0.05,
        tp_stop=0.15,
    )
    resolved = order.with_qty(qty=10.0)
    assert resolved.sl_stop == 0.05
    assert resolved.tp_stop == 0.15


# ----------------------------------------------------------------- long take-profit


def test_long_tp_fires_when_bar_high_pierces_level() -> None:
    """Bar 2 high pierces TP=110 → exit at 110."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # entry fills at open=100; TP=110
        (108, 115, 107, 112),  # high=115 ≥ 110 → TAKE PROFIT at 110
        (95, 100, 90, 99),  # post-exit, no-op
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)

    assert len(result.trades) == 2
    entry, tp = result.trades.iloc[0], result.trades.iloc[1]
    assert entry["qty"] == pytest.approx(10.0)
    assert entry["price"] == pytest.approx(100.0)
    assert entry["reason"] == "signal"
    assert tp["qty"] == pytest.approx(-10.0)
    assert tp["price"] == pytest.approx(110.0)
    assert tp["reason"] == "take_profit"


def test_long_tp_does_not_fire_when_high_below_level() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry at 100; TP=110
        (101, 105, 100, 103),  # high=105 < 110 → no fire
        (102, 109, 101, 108),  # high=109 < 110 → no fire
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["reason"] == "signal"


def test_long_tp_gap_up_fills_at_open_not_level() -> None:
    """If the bar opens ABOVE the TP, the synthesised fill is at the open
    (favourable to the trader, > TP) — realistic gap behaviour."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; TP=110
        (115, 118, 114, 117),  # OPEN=115 already above TP=110 → fill at 115
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)
    tp_row = result.trades[result.trades["reason"] == "take_profit"].iloc[0]
    assert tp_row["price"] == pytest.approx(115.0)


# ----------------------------------------------------------------- short take-profit


def test_short_tp_fires_when_bar_low_pierces_level() -> None:
    """Short at 100 with tp_stop=0.10 → TP=90. Bar 2 low=88 → fire at 90."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short fills at 100; TP=90
        (95, 96, 88, 92),  # low=88 ≤ 90 → TAKE PROFIT at 90
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)
    assert len(result.trades) == 2
    entry, tp = result.trades.iloc[0], result.trades.iloc[1]
    assert entry["qty"] == pytest.approx(-10.0)  # short
    assert tp["qty"] == pytest.approx(10.0)  # cover
    assert tp["price"] == pytest.approx(90.0)
    assert tp["reason"] == "take_profit"


def test_short_tp_does_not_fire_when_low_above_level() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short; TP=90
        (95, 99, 92, 96),  # low=92 > 90 → no fire
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)
    assert len(result.trades) == 1


def test_short_tp_gap_down_fills_at_open_not_level() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short; TP=90
        (85, 87, 80, 82),  # OPEN=85 already below TP=90 → fill at 85
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)
    tp_row = result.trades[result.trades["reason"] == "take_profit"].iloc[0]
    assert tp_row["price"] == pytest.approx(85.0)


# ----------------------------------------------------------------- bracket orders


def test_bracket_order_sets_both_levels_on_position() -> None:
    """Order with both sl_stop and tp_stop populates both Position fields after fill.

    Uses ``Portfolio.apply`` directly because ``Simulator.snapshot()``
    resets ``_live.positions`` post-run.
    """
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    o = Order(ts=ts, asset="BTC", side="buy", qty=10.0, sl_stop=0.10, tp_stop=0.20)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o, qty=10.0, price=100.0))
    pos = p.position("BTC")
    assert pos.qty == pytest.approx(10.0)
    assert pos.sl_level == pytest.approx(90.0)  # 100 * 0.9
    assert pos.tp_level == pytest.approx(120.0)  # 100 * 1.2


def test_sl_only_fires_when_tp_set_too_but_not_breached() -> None:
    """Bracket where only the SL trips — TP exists but is far away."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; SL=90, TP=130 (far)
        (95, 96, 88, 92),  # low=88 ≤ 90 → SL fires; high=96 < 130
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10, tp_stop=0.30)
    result = _run(bars, strat)
    sl_row = result.trades[result.trades["reason"] == "stop_loss"]
    tp_row = result.trades[result.trades["reason"] == "take_profit"]
    assert len(sl_row) == 1
    assert len(tp_row) == 0
    assert sl_row.iloc[0]["price"] == pytest.approx(90.0)


def test_tp_only_fires_when_sl_set_too_but_not_breached() -> None:
    """Bracket where only the TP trips — SL exists but is far away."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; SL=70, TP=110
        (108, 115, 107, 112),  # high=115 ≥ 110 → TP fires; low=107 > 70
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.30, tp_stop=0.10)
    result = _run(bars, strat)
    sl_row = result.trades[result.trades["reason"] == "stop_loss"]
    tp_row = result.trades[result.trades["reason"] == "take_profit"]
    assert len(sl_row) == 0
    assert len(tp_row) == 1
    assert tp_row.iloc[0]["price"] == pytest.approx(110.0)


def test_sl_wins_when_both_could_fire_same_bar() -> None:
    """Wide-range bar pierces both SL and TP → SL fires (conservative default)."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; SL=90, TP=110
        (100, 115, 88, 105),  # high=115 ≥ 110 AND low=88 ≤ 90 → SL wins
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10, tp_stop=0.10)
    result = _run(bars, strat)
    forced = result.trades[result.trades["reason"].isin(["stop_loss", "take_profit"])]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "stop_loss"
    assert forced.iloc[0]["price"] == pytest.approx(90.0)


def test_sl_wins_when_both_could_fire_short_bracket() -> None:
    """Mirror: short with both stops, wide-range bar → SL wins."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short fills; SL=110, TP=90
        (100, 115, 88, 100),  # high=115 ≥ 110 AND low=88 ≤ 90 → SL wins
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, sl_stop=0.10, tp_stop=0.10)
    result = _run(bars, strat)
    forced = result.trades[result.trades["reason"].isin(["stop_loss", "take_profit"])]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "stop_loss"
    assert forced.iloc[0]["price"] == pytest.approx(110.0)


# ----------------------------------------------------------------- accumulation + clearing


def test_tp_level_anchored_to_latest_fill_on_accumulation() -> None:
    """Each new entry resets the TP based on its own fill price (latest-anchor)."""
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")

    o1 = Order(ts=ts, asset="X", side="buy", qty=100.0, tp_stop=0.10)
    o2 = Order(ts=ts, asset="X", side="buy", qty=100.0, tp_stop=0.10)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o1, qty=100.0, price=100.0))
    pos = p.position("X")
    assert pos.tp_level == pytest.approx(110.0)  # 100 * 1.10

    # Second buy at 110 → TP moves to 121 (= 110 * 1.10)
    p.apply(_T(o2, qty=100.0, price=110.0))
    pos = p.position("X")
    assert pos.tp_level == pytest.approx(121.0)


def test_tp_level_cleared_after_position_closes() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; TP=110
        (108, 115, 107, 112),  # TP fires at 110
        (113, 118, 110, 116),  # would fire again if not cleared
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat)
    tp_rows = result.trades[result.trades["reason"] == "take_profit"]
    assert len(tp_rows) == 1


def test_bracket_levels_cleared_together_on_close() -> None:
    """qty → 0 clears BOTH sl_level and tp_level (regardless of which fired).

    Uses ``Portfolio.apply`` directly: open with bracket, then close.
    """
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    o_open = Order(ts=ts, asset="BTC", side="buy", qty=10.0, sl_stop=0.10, tp_stop=0.10)
    o_close = Order(ts=ts, asset="BTC", side="sell", qty=10.0)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o_open, qty=10.0, price=100.0))
    pos = p.position("BTC")
    assert pos.sl_level == pytest.approx(90.0)
    assert pos.tp_level == pytest.approx(110.0)

    p.apply(_T(o_close, qty=-10.0, price=110.0))
    pos = p.position("BTC")
    assert pos.qty == pytest.approx(0.0)
    assert pos.sl_level is None
    assert pos.tp_level is None


# ----------------------------------------------------------------- slippage + costs


def test_tp_uses_slippage_model_for_long() -> None:
    """HalfSpread shifts a sell-side TP fill *down* by half the spread.

    Entry buy slipped up to 100.1 (half-spread); TP=110.11
    (= 100.1 * 1.10); fill = 110.11 * 0.999 = 110.000.
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry slipped to 100.1
        (108, 115, 107, 112),  # high=115 ≥ TP=110.11
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat, slippage=HalfSpread(spread_bps=20.0))
    tp_row = result.trades[result.trades["reason"] == "take_profit"].iloc[0]
    expected_price = 100.0 * 1.001 * 1.10 * 0.999
    assert tp_row["price"] == pytest.approx(expected_price, rel=1e-6)
    assert tp_row["slippage_bps"] == pytest.approx(10.0)


def test_tp_uses_cost_model() -> None:
    """FixedBps fee is charged on the synthesised TP trade."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry
        (108, 115, 107, 112),  # TP fires at 110
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    result = _run(bars, strat, costs=FixedBps(bps=30.0))
    tp_row = result.trades[result.trades["reason"] == "take_profit"].iloc[0]
    expected_fee = abs(110.0 * -10.0) * 30.0 * 1e-4  # 1100 * 0.003 = 3.30
    assert tp_row["fee"] == pytest.approx(expected_fee)


# ----------------------------------------------------------------- multi-asset


def test_multi_asset_brackets_are_independent() -> None:
    """Two assets each with their own bracket; each fires only on its own asset."""
    n = 4
    idx = pd.date_range("2024-01-02", periods=n, freq="1D")
    cols = pd.MultiIndex.from_tuples(
        [("open", a) for a in ("BTC", "ETH")]
        + [("high", a) for a in ("BTC", "ETH")]
        + [("low", a) for a in ("BTC", "ETH")]
        + [("close", a) for a in ("BTC", "ETH")]
        + [("volume", a) for a in ("BTC", "ETH")]
    )
    data = {
        # BTC: TP fires bar 2 (high=115)
        ("open", "BTC"): [100, 100, 108, 95],
        ("high", "BTC"): [102, 101, 115, 100],
        ("low", "BTC"): [99, 99, 107, 90],
        ("close", "BTC"): [100, 100, 112, 99],
        ("volume", "BTC"): [1.0, 1.0, 1.0, 1.0],
        # ETH: SL fires bar 3 (low=88)
        ("open", "ETH"): [50, 50, 49, 47],
        ("high", "ETH"): [52, 51, 50, 49],
        ("low", "ETH"): [49, 49, 48, 44],  # bar 3 low=44 ≤ 45 (50*0.9) → SL
        ("close", "ETH"): [50, 50, 49, 46],
        ("volume", "ETH"): [1.0, 1.0, 1.0, 1.0],
    }
    bars = pd.DataFrame(data, index=idx, columns=cols)

    class _OpenBoth(BaseStrategy):
        def __init__(self) -> None:
            self._fired = False

        def decide(self, ctx: Context) -> list[Order]:
            if self._fired:
                return []
            self._fired = True
            return [
                Order(ts=ctx.ts, asset="BTC", side="buy", qty=5.0, tp_stop=0.10),
                Order(ts=ctx.ts, asset="ETH", side="buy", qty=10.0, sl_stop=0.10),
            ]

    result = _run(bars, _OpenBoth(), cash=200_000.0)
    btc_trades = result.trades[result.trades["asset"] == "BTC"]
    eth_trades = result.trades[result.trades["asset"] == "ETH"]
    # BTC: signal entry + TP exit
    assert (btc_trades["reason"] == "signal").sum() == 1
    assert (btc_trades["reason"] == "take_profit").sum() == 1
    # ETH: signal entry + SL exit
    assert (eth_trades["reason"] == "signal").sum() == 1
    assert (eth_trades["reason"] == "stop_loss").sum() == 1


# ----------------------------------------------------------------- defensive


def test_tp_check_handles_nan_high_low() -> None:
    """A bar with NaN high/low (mixed-frequency panel) doesn't crash the check."""
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
    strat = _OneShotEntry(asset=asset, side="buy", qty=10.0, tp_stop=0.10)
    result = _run(df, strat)
    assert result is not None  # didn't crash


def test_iron_law_no_brackets_means_no_extra_reasons() -> None:
    """A strategy without sl_stop / tp_stop produces only signal-reason trades."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),
        (95, 96, 88, 92),  # would fire SL if any was set
        (108, 115, 107, 112),  # would fire TP
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0)
    result = _run(bars, strat)
    assert (result.trades["reason"] == "signal").all()


def test_strategy_sees_flat_position_on_bar_after_tp_fire() -> None:
    """The TP synthesised exit is applied before decide() runs, so the
    strategy sees a flat position on the bar after the fire."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; TP=110
        (108, 115, 107, 112),  # TP fires here
        (113, 118, 110, 116),  # decide() runs with flat position
    ])

    seen_qty: list[float] = []

    class _Watcher(_OneShotEntry):
        def decide(self, ctx: Context) -> list[Order]:
            pos = ctx.portfolio._live.positions.get(self.asset)
            seen_qty.append(pos.qty if pos else 0.0)
            return super().decide(ctx)

    strat = _Watcher(asset="BTC", side="buy", qty=10.0, tp_stop=0.10)
    _run(bars, strat)
    # bar 0: flat (about to enter), bar 1: still flat (entry queued for fill at 1's
    # open), bar 2: long (entry filled), bar 3: flat (TP exited).
    # Validate the FULL timeline — not just the final bar — so the test fails
    # if the position never opened (which would also satisfy "final qty == 0").
    assert any(q > 0 for q in seen_qty[:-1]), (
        f"expected at least one bar with a long position before TP fire; got {seen_qty}"
    )
    assert seen_qty[-1] == pytest.approx(0.0)
