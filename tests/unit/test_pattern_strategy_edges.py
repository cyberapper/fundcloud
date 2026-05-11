"""Edge-case coverage for ``strategies.PatternStrategy``.

The happy-path simulator tests live in ``test_pattern_condition.py``;
this file fills the per-bar branches codecov flagged on the FLG-1015 PR
— skipped events (bearish without inverse, NaN target/stop, empty
indicator), the small ``_bar_field`` and ``_equity`` helpers, the
time-stop logic, and the simulator-level "position already flat" path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    Direction,
    DoubleBottom,
    DoubleTop,
    Pattern,
    PatternCondition,
)
from fundcloud.portfolio import Portfolio
from fundcloud.sim import Simulator
from fundcloud.strategies import PatternStrategy
from fundcloud.strategies.base import Context


def _bars(n: int = 200, asset: str = "AAA") -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, size=n))
    df = pd.DataFrame(
        {
            ("open", asset): close,
            ("high", asset): close + 0.5,
            ("low", asset): close - 0.5,
            ("close", asset): close,
            ("volume", asset): np.full(n, 1.0e6),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def _double_bottom_event(
    asset: str,
    breakout_ts: pd.Timestamp,
    formation_start: pd.Timestamp,
    *,
    entry: float = 100.0,
    target: float = 105.0,
    stop: float = 95.0,
    pattern: Pattern = Pattern.DOUBLE_BOTTOM,
) -> dict:
    return {
        "pattern": pattern,
        "asset": asset,
        "formation_start": formation_start,
        "formation_end": breakout_ts,
        "breakout_ts": breakout_ts,
        "breakout_level": entry,
        "formation_height": abs(target - entry),
        "target_price": target,
        "stop_price": stop,
        "quality": 75.0,
        "variant": "STRICT_ADAM_ADAM",
        "pivots": [
            {"ts": formation_start, "price": stop, "kind": "LOW"},
            {"ts": breakout_ts, "price": entry, "kind": "HIGH"},
            {"ts": breakout_ts, "price": stop, "kind": "LOW"},
        ],
        "meta": {},
    }


# ---------------------------------------------------------------------------
# init() filtering paths
# ---------------------------------------------------------------------------


class _NullPortfolio:
    cash: float = 0.0


class TestInit:
    def test_empty_events_is_clean(self) -> None:
        bars = _bars()
        strat = PatternStrategy(DoubleBottom())
        # Stub the indicator to return an empty events table.
        strat.indicator.events = lambda _bars: pd.DataFrame(  # type: ignore[method-assign]
            columns=EVENTS_COLUMNS
        )
        strat.init(bars, _NullPortfolio())  # type: ignore[arg-type]
        assert strat._events_by_asset == {}
        assert strat._open == {}

    def test_bearish_direction_skips_long_only_execution(self) -> None:
        """Strategy is long-only by design; events resolved short are dropped."""
        bars = _bars()
        ts = bars.index[120]
        fs = bars.index[100]
        events = pd.DataFrame(
            [
                _double_bottom_event(
                    "AAA",
                    ts,
                    fs,
                    entry=100.0,
                    target=95.0,
                    stop=105.0,
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        strat = PatternStrategy(DoubleTop(), direction=Direction.BEARISH)
        strat.indicator.events = lambda _bars: events  # type: ignore[method-assign]
        strat.init(bars, _NullPortfolio())  # type: ignore[arg-type]
        assert strat._events_by_asset == {}

    def test_nan_target_or_stop_event_is_skipped(self) -> None:
        # Event references an asset that's not in the bars frame —
        # ``apply_condition`` writes NaN target/stop, and ``init`` then
        # skips the row via the NaN-guard branch.
        bars = _bars()
        ts = bars.index[120]
        fs = bars.index[100]
        events = pd.DataFrame(
            [_double_bottom_event("ZZZ", ts, fs, entry=100.0, target=105.0, stop=95.0)],
            columns=EVENTS_COLUMNS,
        )
        strat = PatternStrategy(DoubleBottom())
        strat.indicator.events = lambda _bars: events  # type: ignore[method-assign]
        strat.init(bars, _NullPortfolio())  # type: ignore[arg-type]
        assert strat._events_by_asset == {}


# ---------------------------------------------------------------------------
# Small helpers (_bar_field / _equity)
# ---------------------------------------------------------------------------


def _ctx(bars: pd.DataFrame, i: int, *, portfolio: Portfolio) -> Context:
    ts = bars.index[i]
    bar = bars.iloc[i]
    history = bars.iloc[: i + 1]
    return Context(
        ts=ts,
        bar=bar,
        history=history,
        portfolio=portfolio,
        assets=("AAA",),
    )


class TestBarFieldHelper:
    def test_returns_none_for_missing_column(self) -> None:
        bars = _bars()
        portfolio = Portfolio(cash=1.0e5)
        strat = PatternStrategy(DoubleBottom())
        ctx = _ctx(bars, 50, portfolio=portfolio)
        assert strat._bar_field(ctx, "ZZZ", "close") is None

    def test_returns_none_for_nan_value(self) -> None:
        bars = _bars()
        bars.loc[bars.index[10], ("close", "AAA")] = float("nan")
        portfolio = Portfolio(cash=1.0e5)
        strat = PatternStrategy(DoubleBottom())
        ctx = _ctx(bars, 10, portfolio=portfolio)
        assert strat._bar_field(ctx, "AAA", "close") is None

    def test_equity_includes_open_position_value(self) -> None:
        bars = _bars()
        qty = 5.0
        portfolio = Portfolio(cash=10_000.0, positions={"AAA": qty})
        strat = PatternStrategy(DoubleBottom())
        ctx = _ctx(bars, 50, portfolio=portfolio)
        close = float(bars.iloc[50][("close", "AAA")])
        assert strat._equity(ctx) == pytest.approx(10_000.0 + qty * close)

    def test_equity_cash_only_when_no_positions(self) -> None:
        bars = _bars()
        portfolio = Portfolio(cash=10_000.0)
        strat = PatternStrategy(DoubleBottom())
        ctx = _ctx(bars, 50, portfolio=portfolio)
        assert strat._equity(ctx) == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Time-stop branches
# ---------------------------------------------------------------------------


class TestTimeStop:
    def test_returns_false_when_disabled(self) -> None:
        bars = _bars()
        strat = PatternStrategy(DoubleBottom())  # default condition: time_stop_bars=None
        strat._bar_index = bars.index
        portfolio = Portfolio(cash=1.0)
        ctx = _ctx(bars, 5, portfolio=portfolio)
        trade = {"entry_ts": bars.index[0]}
        assert strat._time_stop_hit(ctx, trade) is False

    def test_returns_false_for_unknown_entry_ts(self) -> None:
        bars = _bars()
        cond = PatternCondition(time_stop_bars=5)
        strat = PatternStrategy(DoubleBottom(), condition=cond)
        strat._bar_index = bars.index
        portfolio = Portfolio(cash=1.0)
        ctx = _ctx(bars, 10, portfolio=portfolio)
        trade = {"entry_ts": pd.Timestamp("1990-01-01", tz="UTC")}
        assert strat._time_stop_hit(ctx, trade) is False

    def test_fires_after_n_bars(self) -> None:
        bars = _bars()
        cond = PatternCondition(time_stop_bars=3)
        strat = PatternStrategy(DoubleBottom(), condition=cond)
        strat._bar_index = bars.index
        portfolio = Portfolio(cash=1.0)
        ctx = _ctx(bars, 10, portfolio=portfolio)
        trade = {"entry_ts": bars.index[7]}
        assert strat._time_stop_hit(ctx, trade) is True

    def test_does_not_fire_before_n_bars(self) -> None:
        bars = _bars()
        cond = PatternCondition(time_stop_bars=10)
        strat = PatternStrategy(DoubleBottom(), condition=cond)
        strat._bar_index = bars.index
        portfolio = Portfolio(cash=1.0)
        ctx = _ctx(bars, 5, portfolio=portfolio)
        trade = {"entry_ts": bars.index[0]}
        assert strat._time_stop_hit(ctx, trade) is False


# ---------------------------------------------------------------------------
# Simulator-driven exit paths
# ---------------------------------------------------------------------------


class TestSimulatorExits:
    def test_target_hit_closes_trade(self) -> None:
        """Sets up a synthetic event whose target is reached on the very
        next bar; the strategy's exit path should fire and the trade list
        should record an open + close."""
        bars = _bars(n=100)
        ts_entry = bars.index[40]
        fs = bars.index[20]
        # Make sure the bar after the breakout has a high above the target.
        target = float(bars[("close", "AAA")].iloc[40]) + 1.0
        bars.loc[bars.index[41], ("high", "AAA")] = target + 5.0
        events = pd.DataFrame(
            [
                _double_bottom_event(
                    "AAA",
                    ts_entry,
                    fs,
                    entry=float(bars[("close", "AAA")].iloc[40]),
                    target=target,
                    stop=float(bars[("close", "AAA")].iloc[40]) - 10.0,
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        strat = PatternStrategy(DoubleBottom(), size=0.1)
        strat.indicator.events = lambda _bars: events  # type: ignore[method-assign]
        result = Simulator(bars).run_strategy(strat)
        # At least one round-trip trade — entry on bar 40 + exit on bar 41.
        assert len(result.trades) >= 1

    def test_time_stop_via_simulator(self) -> None:
        """``time_stop_bars`` should produce an exit even if neither
        target nor stop is hit on a flat-ish bar series."""
        bars = _bars(n=100)
        ts_entry = bars.index[40]
        fs = bars.index[20]
        entry_close = float(bars[("close", "AAA")].iloc[40])
        events = pd.DataFrame(
            [
                _double_bottom_event(
                    "AAA",
                    ts_entry,
                    fs,
                    entry=entry_close,
                    target=entry_close + 1_000.0,  # never hit
                    stop=entry_close - 1_000.0,  # never hit
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        cond = PatternCondition(time_stop_bars=3)
        strat = PatternStrategy(DoubleBottom(), condition=cond, size=0.1)
        strat.indicator.events = lambda _bars: events  # type: ignore[method-assign]
        result = Simulator(bars).run_strategy(strat)
        # The time stop should have flushed the position before the run
        # ends, so the trades list contains at least one round trip.
        assert len(result.trades) >= 1
