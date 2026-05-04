"""Intra-bar trailing-stop tests for ``Simulator``.

Mirror of :mod:`tests.unit.test_simulator_stops`, but for the
``tsl_stop`` field on :class:`~fundcloud.sim.Order`. The trailing
stop's anchor ratchets in the favourable direction with each bar
(long: ``max(anchor, bar.high)``; short: ``min(anchor, bar.low)``),
then the trigger is checked against the unfavourable side
(``bar.low`` for long, ``bar.high`` for short). A fired trailing stop
is tagged ``Trade.reason="trailing_stop"``.

What's exercised:

* Long / short TSL fires on bar.low / bar.high pierce.
* Anchor ratchets up on a long; never moves backward.
* Same for short (ratchets down).
* TSL gap behaviour.
* Coexistence with fixed ``sl_stop`` — tighter level wins
  (``max(sl, tsl)`` for long, ``min`` for short); reason flips
  between ``"stop_loss"`` and ``"trailing_stop"``.
* Coexistence with ``tp_stop`` — stops still beat take-profit.
* TSL re-anchored to latest fill on accumulating entries.
* TSL cleared on close.
* Multi-asset positions with independent trails.
* NaN-bar safety.
* ``Order.tsl_stop`` validation.
* ``Order.with_qty`` propagation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.sim import (
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
        tsl_stop: float | None = None,
    ) -> None:
        self.asset = asset
        self.side = side
        self.qty = qty
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
        self.tsl_stop = tsl_stop
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
                tsl_stop=self.tsl_stop,
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
def test_order_rejects_tsl_stop_outside_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="tsl_stop"):
        Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, tsl_stop=bad)


def test_order_accepts_tsl_stop_in_unit_interval() -> None:
    o = Order(ts=pd.Timestamp("2024-01-02"), asset="X", side="buy", qty=1.0, tsl_stop=0.05)
    assert o.tsl_stop == 0.05


def test_order_with_qty_propagates_tsl_stop() -> None:
    o = Order(
        ts=pd.Timestamp("2024-01-02"),
        asset="X",
        side="buy",
        notional=1000.0,
        tsl_stop=0.07,
    )
    resolved = o.with_qty(qty=10.0)
    assert resolved.tsl_stop == 0.07


def test_order_accepts_full_bracket_with_tsl() -> None:
    o = Order(
        ts=pd.Timestamp("2024-01-02"),
        asset="X",
        side="buy",
        qty=1.0,
        sl_stop=0.10,
        tp_stop=0.20,
        tsl_stop=0.05,
    )
    assert o.sl_stop == 0.10
    assert o.tp_stop == 0.20
    assert o.tsl_stop == 0.05


# ----------------------------------------------------------------- long trailing stop


def test_long_tsl_fires_when_bar_low_pierces_ratcheted_level() -> None:
    """Anchor ratchets up to bar 2's high; bar 3 low pierces the new level → fire."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # entry fills at 100; anchor=100, tsl=90
        (102, 110, 101, 108),  # high=110 → anchor=110, tsl=99
        (105, 107, 95, 96),  # low=95 ≤ 99 → TSL fires at 99
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)

    assert len(result.trades) == 2
    entry, tsl = result.trades.iloc[0], result.trades.iloc[1]
    assert entry["price"] == pytest.approx(100.0)
    assert tsl["qty"] == pytest.approx(-10.0)
    assert tsl["price"] == pytest.approx(99.0)
    assert tsl["reason"] == "trailing_stop"


def test_long_tsl_does_not_fire_when_low_stays_above_level() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; anchor=100, tsl=90
        (98, 99, 95, 96),  # low=95 > 90 → no fire (anchor unchanged)
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)
    assert len(result.trades) == 1


def test_long_tsl_anchor_only_ratchets_up_never_down() -> None:
    """Once the anchor reaches a high, a subsequent lower-high bar must not move it.

    Bar 2 (post-trigger) ratchets anchor up to 120 (level=108).
    Bar 3 opens at 110 (above level), high=105 < 120, low=100. Post-open
    anchor unchanged (110 < 120 → no change), level stays 108. low=100
    ≤ 108 → TSL fires at 108.
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; anchor=100
        (102, 120, 101, 108),  # post-trigger ratchet → anchor=120, tsl=108
        (110, 115, 100, 108),  # opens above 108, low=100 ≤ 108 → TSL fires at 108
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)
    tsl_rows = result.trades[result.trades["reason"] == "trailing_stop"]
    assert len(tsl_rows) == 1
    assert tsl_rows.iloc[0]["price"] == pytest.approx(108.0)


def test_long_tsl_gap_down_fills_at_open() -> None:
    """If the bar opens below the trail level, the synthesised fill is at the open.

    Bar 2 ratchets the anchor up to 120 without firing (low=110 > new
    tsl=108). Bar 3 then opens at 90 — a gap-down vs the 108 level the
    trader had in force at bar start — so the fill is at 90.
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; anchor=100, tsl=90
        (110, 120, 110, 115),  # anchor → 120; tsl=108. low=110 > 108 → no fire
        (90, 92, 85, 88),  # OPEN=90 already below tsl=108 → fill at 90
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)
    tsl_row = result.trades[result.trades["reason"] == "trailing_stop"].iloc[0]
    assert tsl_row["price"] == pytest.approx(90.0)


# ----------------------------------------------------------------- short trailing stop


class _OpenShortViaSell(BaseStrategy):
    """Open a short with optional brackets on bar 0; never trade again."""

    def __init__(
        self,
        *,
        asset: str,
        qty: float,
        sl_stop: float | None = None,
        tp_stop: float | None = None,
        tsl_stop: float | None = None,
    ) -> None:
        self.asset = asset
        self.qty = qty
        self.sl_stop = sl_stop
        self.tp_stop = tp_stop
        self.tsl_stop = tsl_stop
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
                tsl_stop=self.tsl_stop,
            )
        ]


def test_short_tsl_fires_when_bar_high_pierces_ratcheted_level() -> None:
    """Short at 100; anchor ratchets DOWN to bar 2's low; bar 3 high pierces new level."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),  # entry signal
        (100, 101, 99, 100),  # short fills at 100; anchor=100, tsl=110
        (95, 96, 90, 92),  # low=90 → anchor=90, tsl=99
        (97, 102, 96, 100),  # high=102 ≥ 99 → TSL fires at 99
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)
    assert len(result.trades) == 2
    entry, tsl = result.trades.iloc[0], result.trades.iloc[1]
    assert entry["qty"] == pytest.approx(-10.0)
    assert tsl["qty"] == pytest.approx(10.0)  # cover
    assert tsl["price"] == pytest.approx(99.0)
    assert tsl["reason"] == "trailing_stop"


def test_short_tsl_anchor_only_ratchets_down() -> None:
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short; anchor=100, tsl=110
        (90, 95, 80, 88),  # low=80 → anchor=80, tsl=88
        (88, 92, 85, 90),  # low=85 > 80 → anchor STAYS 80, tsl STAYS 88
        (87, 90, 85, 89),  # high=90 ≥ 88 → TSL fires at 88
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)
    tsl_rows = result.trades[result.trades["reason"] == "trailing_stop"]
    assert len(tsl_rows) == 1
    assert tsl_rows.iloc[0]["price"] == pytest.approx(88.0)


def test_short_tsl_gap_up_fills_at_open() -> None:
    """Mirror of the long gap test for shorts.

    Bar 2 ratchets the anchor down to 80 without firing (high=86 < new
    tsl=88). Bar 3 opens at 95 — a gap-up vs the 88 level — so the fill
    is at 95.
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short; anchor=100, tsl=110
        (85, 86, 80, 85),  # anchor → 80; tsl=88. high=86 < 88 → no fire
        (95, 98, 92, 96),  # OPEN=95 already above tsl=88 → fill at 95
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, tsl_stop=0.10)
    result = _run(bars, strat)
    tsl_row = result.trades[result.trades["reason"] == "trailing_stop"].iloc[0]
    assert tsl_row["price"] == pytest.approx(95.0)


# ----------------------------------------------------------------- coexistence with fixed SL


def test_long_tsl_wins_over_fixed_sl_when_tighter() -> None:
    """tsl_stop=0.05 ratcheted to bar 2's high gives a tighter level than sl_stop=0.10.

    Bar 2 (post-trigger) ratchets anchor up to 120, level becomes 114.
    Bar 3 must open *above* the trail level to avoid a gap fire — open=115,
    low=110. low ≤ 114 but > sl=90 → TSL wins (tighter fill).
    """
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; sl_level=90, anchor=100, tsl=95
        (102, 120, 101, 108),  # post-trigger ratchet → anchor=120; tsl=114
        (115, 116, 110, 113),  # opens above 114, low=110 ≤ 114 (tsl) but > 90 (sl)
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10, tsl_stop=0.05)
    result = _run(bars, strat)
    forced = result.trades[result.trades["reason"].isin(["stop_loss", "trailing_stop"])]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "trailing_stop"
    assert forced.iloc[0]["price"] == pytest.approx(114.0)


def test_long_fixed_sl_wins_when_tsl_is_looser() -> None:
    """Before the trail anchor moves, fixed SL=0.10 (level=90) is tighter than tsl=0.20 (level=80)."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; sl_level=90, anchor=100, tsl=80
        (98, 100, 88, 92),  # low=88 ≤ 90 → fixed SL fires at 90
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, sl_stop=0.10, tsl_stop=0.20)
    result = _run(bars, strat)
    forced = result.trades[result.trades["reason"].isin(["stop_loss", "trailing_stop"])]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "stop_loss"
    assert forced.iloc[0]["price"] == pytest.approx(90.0)


def test_short_fixed_sl_wins_when_tsl_is_looser() -> None:
    """Short: tsl=0.20 (level=120) is looser than sl=0.10 (level=110); fixed SL fires first."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # short; sl_level=110, anchor=100, tsl=120
        (105, 112, 104, 109),  # high=112 ≥ 110 → fixed SL fires at 110
    ])
    strat = _OpenShortViaSell(asset="BTC", qty=10.0, sl_stop=0.10, tsl_stop=0.20)
    result = _run(bars, strat)
    forced = result.trades[result.trades["reason"].isin(["stop_loss", "trailing_stop"])]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "stop_loss"


# ----------------------------------------------------------------- coexistence with TP


def test_stops_beat_tp_with_tsl_present() -> None:
    """Wide-range bar pierces both TP and the ratcheted TSL → TSL (a stop) wins."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry; tsl=95, tp_level=110
        (100, 115, 90, 100),  # anchor → 115, tsl=109. low=90 ≤ 109 + high=115 ≥ 110
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tp_stop=0.10, tsl_stop=0.05)
    result = _run(bars, strat)
    forced = result.trades[
        result.trades["reason"].isin(["stop_loss", "take_profit", "trailing_stop"])
    ]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "trailing_stop"


# ----------------------------------------------------------------- accumulation + clearing


def test_tsl_anchor_retained_on_accumulating_entry() -> None:
    """Accumulating entries retain the original trail anchor.

    The first entry sets ``tsl_pct`` and ``tsl_anchor`` from its fill
    price. A subsequent buy at a *higher* price (which would tighten
    the trail under a re-anchor convention) leaves both fields alone —
    the high-water mark continues ratcheting from the original entry.
    """
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    o1 = Order(ts=ts, asset="X", side="buy", qty=100.0, tsl_stop=0.10)
    o2 = Order(ts=ts, asset="X", side="buy", qty=100.0, tsl_stop=0.10)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o1, qty=100.0, price=100.0))
    pos = p.position("X")
    assert pos.tsl_pct == pytest.approx(0.10)
    assert pos.tsl_anchor == pytest.approx(100.0)

    p.apply(_T(o2, qty=100.0, price=120.0))
    pos = p.position("X")
    assert pos.tsl_pct == pytest.approx(0.10)
    assert pos.tsl_anchor == pytest.approx(100.0)  # NOT re-anchored


def test_tsl_anchor_retained_when_second_order_omits_tsl_stop() -> None:
    """A subsequent add without tsl_stop also leaves the trail in place."""
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    o1 = Order(ts=ts, asset="X", side="buy", qty=100.0, tsl_stop=0.10)
    o2 = Order(ts=ts, asset="X", side="buy", qty=100.0)  # no tsl_stop

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o1, qty=100.0, price=100.0))
    p.apply(_T(o2, qty=100.0, price=120.0))
    pos = p.position("X")
    assert pos.tsl_pct == pytest.approx(0.10)
    assert pos.tsl_anchor == pytest.approx(100.0)


def test_tsl_initialised_on_first_entry_when_first_order_omits_tsl_stop() -> None:
    """If the first entry omits tsl_stop, the trail is inactive until a
    later add that carries it. The first add to set tsl_stop becomes
    the anchor."""
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    o1 = Order(ts=ts, asset="X", side="buy", qty=100.0)
    o2 = Order(ts=ts, asset="X", side="buy", qty=100.0, tsl_stop=0.10)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o1, qty=100.0, price=100.0))
    pos = p.position("X")
    assert pos.tsl_pct is None
    assert pos.tsl_anchor is None

    p.apply(_T(o2, qty=100.0, price=120.0))
    pos = p.position("X")
    assert pos.tsl_pct == pytest.approx(0.10)
    assert pos.tsl_anchor == pytest.approx(120.0)


def test_tsl_cleared_on_close() -> None:
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    o_open = Order(ts=ts, asset="X", side="buy", qty=10.0, tsl_stop=0.10)
    o_close = Order(ts=ts, asset="X", side="sell", qty=10.0)

    class _T:
        def __init__(self, order: Order, qty: float, price: float) -> None:
            self.order = order
            self.asset = order.asset
            self.qty = qty
            self.price = price
            self.fee = 0.0

    p.apply(_T(o_open, qty=10.0, price=100.0))
    pos = p.position("X")
    assert pos.tsl_pct == pytest.approx(0.10)
    p.apply(_T(o_close, qty=-10.0, price=110.0))
    pos = p.position("X")
    assert pos.qty == pytest.approx(0.0)
    assert pos.tsl_pct is None
    assert pos.tsl_anchor is None


# ----------------------------------------------------------------- multi-asset / NaN


def test_multi_asset_tsl_is_independent() -> None:
    """Two assets with different tsl_stop values trail independently."""
    n = 5
    idx = pd.date_range("2024-01-02", periods=n, freq="1D")
    cols = pd.MultiIndex.from_tuples(
        [("open", a) for a in ("BTC", "ETH")]
        + [("high", a) for a in ("BTC", "ETH")]
        + [("low", a) for a in ("BTC", "ETH")]
        + [("close", a) for a in ("BTC", "ETH")]
        + [("volume", a) for a in ("BTC", "ETH")]
    )
    data = {
        # BTC: anchor ratchets to 120 on bar 2; bar 3 low=105 ≤ 108 → TSL
        ("open", "BTC"): [100, 100, 102, 105, 100],
        ("high", "BTC"): [102, 101, 120, 110, 105],
        ("low", "BTC"): [99, 99, 100, 105, 95],
        ("close", "BTC"): [100, 100, 108, 107, 100],
        ("volume", "BTC"): [1.0] * n,
        # ETH: stays in a tight range; no fire
        ("open", "ETH"): [50, 50, 51, 50, 50],
        ("high", "ETH"): [51, 51, 52, 51, 51],
        ("low", "ETH"): [49, 49, 50, 49, 49],
        ("close", "ETH"): [50, 50, 51, 50, 50],
        ("volume", "ETH"): [1.0] * n,
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
                Order(ts=ctx.ts, asset="BTC", side="buy", qty=5.0, tsl_stop=0.10),
                Order(ts=ctx.ts, asset="ETH", side="buy", qty=10.0, tsl_stop=0.10),
            ]

    result = _run(bars, _OpenBoth(), cash=200_000.0)
    btc_tsl = result.trades[
        (result.trades["asset"] == "BTC") & (result.trades["reason"] == "trailing_stop")
    ]
    eth_tsl = result.trades[
        (result.trades["asset"] == "ETH") & (result.trades["reason"] == "trailing_stop")
    ]
    assert len(btc_tsl) == 1
    assert len(eth_tsl) == 0  # ETH never trailed enough to fire


def test_tsl_check_handles_nan_high_low() -> None:
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
    strat = _OneShotEntry(asset=asset, side="buy", qty=10.0, tsl_stop=0.10)
    result = _run(df, strat)
    assert result is not None  # didn't crash


# ----------------------------------------------------------------- slippage


def test_tsl_uses_slippage_model() -> None:
    """HalfSpread shifts the TSL fill on the appropriate side."""
    bars = _bars_from_ohlc([
        (100, 102, 99, 100),
        (100, 101, 99, 100),  # entry slipped to 100.1 → anchor=100.1, tsl=95.095
        (
            101,
            110,
            100,
            105,
        ),  # high=110 → anchor=110*1.001 → no slippage on the high; anchor=110.11
        (108, 109, 100, 102),  # low=100 ≤ tsl ≈ 99 (after slip-adjusted entry) → fire
    ])
    strat = _OneShotEntry(asset="BTC", side="buy", qty=10.0, tsl_stop=0.05)
    result = _run(bars, strat, slippage=HalfSpread(spread_bps=20.0))
    tsl_row = result.trades[result.trades["reason"] == "trailing_stop"]
    assert len(tsl_row) == 1
    # Sanity: slippage_bps recorded
    assert tsl_row.iloc[0]["slippage_bps"] == pytest.approx(10.0)
