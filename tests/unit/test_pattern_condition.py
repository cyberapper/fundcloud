"""Tests for ``apply_condition`` (target/stop filling) and the
``PatternStrategy`` end-to-end backtest path.

Each test uses a small synthetic events table — the geometry is hand-
crafted so the expected target / stop levels are computable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    Direction,
    Pattern,
    PatternCondition,
    StopMethod,
    TargetMethod,
    apply_condition,
)


def _bars(n: int = 200, asset: str = "AAA") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, size=n))
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame(
        {
            ("open", asset): close,
            ("high", asset): high,
            ("low", asset): low,
            ("close", asset): close,
            ("volume", asset): np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def _double_bottom_event(
    asset: str,
    breakout_ts: pd.Timestamp,
    formation_start: pd.Timestamp,
    *,
    entry: float,
    pivot_lows: tuple[float, float],
    pivot_high: float,
) -> dict:
    """Construct a synthetic Double Bottom event row with explicit pivots."""
    return {
        "pattern": Pattern.DOUBLE_BOTTOM,
        "asset": asset,
        "direction": Direction.BULLISH,
        "formation_start": formation_start,
        "formation_end": breakout_ts,
        "breakout_ts": breakout_ts,
        "entry_price": entry,
        "breakout_price": entry,
        "target_price": float("nan"),
        "stop_price": float("nan"),
        "quality": 75.0,
        "variant": "STRICT_ADAM_ADAM",
        "pivots": [
            {"ts": formation_start, "price": pivot_lows[0], "kind": "LOW"},
            {"ts": breakout_ts, "price": pivot_high, "kind": "HIGH"},
            {"ts": breakout_ts, "price": pivot_lows[1], "kind": "LOW"},
        ],
        "meta": {},
    }


def test_apply_condition_measured_move_target_for_bullish_double_bottom() -> None:
    """Bullish event with two troughs at 95 and entry (neckline) at 100:
    pattern_height = 100 - 95 = 5. MEASURED_MOVE target = 100 + 5 = 105.
    BELOW_PIVOT stop = min(troughs) = 95.
    """
    bars = _bars()
    ts = bars.index[100]
    fs = bars.index[80]
    events = pd.DataFrame(
        [
            _double_bottom_event(
                "AAA",
                ts,
                fs,
                entry=100.0,
                pivot_lows=(95.0, 95.0),
                pivot_high=100.0,
            )
        ],
        columns=EVENTS_COLUMNS,
    )

    out = apply_condition(events, PatternCondition(), bars)

    assert out.loc[0, "target_price"] == pytest.approx(105.0)
    assert out.loc[0, "stop_price"] == pytest.approx(95.0)


def test_apply_condition_fib_1_618_target_extends_further() -> None:
    """FIB target = entry + 1.618 * pattern_height = 100 + 1.618*5 = 108.09."""
    bars = _bars()
    ts = bars.index[100]
    fs = bars.index[80]
    events = pd.DataFrame(
        [
            _double_bottom_event(
                "AAA", ts, fs, entry=100.0, pivot_lows=(95.0, 95.0), pivot_high=100.0
            )
        ],
        columns=EVENTS_COLUMNS,
    )
    cond = PatternCondition().override(target_method=TargetMethod.FIB_1_618)

    out = apply_condition(events, cond, bars)

    assert out.loc[0, "target_price"] == pytest.approx(100.0 + 1.618 * 5.0)


def test_apply_condition_fixed_pct_stop() -> None:
    """FIXED_PCT 5% stop on a 100 entry = 95."""
    bars = _bars()
    ts = bars.index[100]
    fs = bars.index[80]
    events = pd.DataFrame(
        [
            _double_bottom_event(
                "AAA", ts, fs, entry=100.0, pivot_lows=(95.0, 95.0), pivot_high=100.0
            )
        ],
        columns=EVENTS_COLUMNS,
    )
    cond = PatternCondition().override(stop_method=StopMethod.FIXED_PCT)

    out = apply_condition(events, cond, bars)

    assert out.loc[0, "stop_price"] == pytest.approx(95.0)


def test_apply_condition_atr_multiple_stop() -> None:
    """ATR-multiple stop is finite and below entry for bullish events."""
    bars = _bars()
    ts = bars.index[100]
    fs = bars.index[80]
    events = pd.DataFrame(
        [
            _double_bottom_event(
                "AAA", ts, fs, entry=100.0, pivot_lows=(95.0, 95.0), pivot_high=100.0
            )
        ],
        columns=EVENTS_COLUMNS,
    )
    cond = PatternCondition().override(stop_method=StopMethod.ATR_MULTIPLE, atr_multiple=2.0)

    out = apply_condition(events, cond, bars)

    stop = out.loc[0, "stop_price"]
    assert np.isfinite(stop)
    assert stop < 100.0  # below entry for bullish
    # 2× ATR on this synthetic series is small (~2.0) → stop near 98
    assert 95.0 < stop < 99.5


def test_apply_condition_empty_events_returns_empty() -> None:
    bars = _bars()
    empty = pd.DataFrame(columns=EVENTS_COLUMNS)
    out = apply_condition(empty, PatternCondition(), bars)
    assert out.empty
    assert list(out.columns) == list(EVENTS_COLUMNS)


def _double_top_event(
    asset: str,
    breakout_ts: pd.Timestamp,
    formation_start: pd.Timestamp,
    *,
    entry: float,
    pivot_highs: tuple[float, float],
    pivot_low: float,
) -> dict:
    """Construct a synthetic Double Top event row with explicit pivots."""
    return {
        "pattern": Pattern.DOUBLE_TOP,
        "asset": asset,
        "direction": Direction.BEARISH,
        "formation_start": formation_start,
        "formation_end": breakout_ts,
        "breakout_ts": breakout_ts,
        "entry_price": entry,
        "breakout_price": entry,
        "target_price": float("nan"),
        "stop_price": float("nan"),
        "quality": 75.0,
        "variant": "STRICT_ADAM_ADAM",
        "pivots": [
            {"ts": formation_start, "price": pivot_highs[0], "kind": "HIGH"},
            {"ts": breakout_ts, "price": pivot_low, "kind": "LOW"},
            {"ts": breakout_ts, "price": pivot_highs[1], "kind": "HIGH"},
        ],
        "meta": {},
    }


def test_pattern_strategy_runs_end_to_end_with_synthetic_events() -> None:
    """PatternStrategy.init + decide should run without error and produce
    a SimResult with at least one trade placed.
    """
    from fundcloud.features.patterns import DoubleBottom
    from fundcloud.sim import Simulator
    from fundcloud.strategies import PatternStrategy

    bars = _bars(n=600, asset="ZZZ")
    # Stub the indicator so the strategy is fed a deterministic event
    # rather than relying on the random walk to produce one.
    fs = bars.index[100]
    ts = bars.index[200]
    stub_events = pd.DataFrame(
        [
            _double_bottom_event(
                "ZZZ",
                ts,
                fs,
                entry=float(bars[("close", "ZZZ")].iloc[200]),
                pivot_lows=(95.0, 95.0),
                pivot_high=100.0,
            )
        ],
        columns=EVENTS_COLUMNS,
    )
    indicator = DoubleBottom(min_quality=0.0)
    indicator.events = lambda _bars: stub_events  # type: ignore[method-assign]

    strat = PatternStrategy(indicator, size=0.1)
    result = Simulator(bars).run_strategy(strat)

    assert len(result.equity_curve) == len(bars)
    assert np.isfinite(result.equity_curve.iloc[-1])
    assert len(result.trades) >= 1


def test_pattern_strategy_inverse_flips_bearish_to_long() -> None:
    """With ``inverse=True``, a bearish events panel produces long entries
    (each event's sign flips from -1 to +1)."""
    from fundcloud.features.patterns import DoubleTop
    from fundcloud.strategies import PatternStrategy

    bars = _bars(n=600, asset="WWW")
    fs = bars.index[100]
    ts = bars.index[200]
    stub_events = pd.DataFrame(
        [
            _double_top_event(
                "WWW",
                ts,
                fs,
                entry=float(bars[("close", "WWW")].iloc[200]),
                pivot_highs=(105.0, 105.0),
                pivot_low=100.0,
            )
        ],
        columns=EVENTS_COLUMNS,
    )
    indicator = DoubleTop(min_quality=0.0)
    indicator.events = lambda _bars: stub_events  # type: ignore[method-assign]

    strat = PatternStrategy(indicator, inverse=True, size=0.1)
    strat.init(bars, _NullPortfolio())

    assert strat._events_by_asset, "inverse flip should retain bearish events as longs"
    for recs in strat._events_by_asset.values():
        assert recs, "expected at least one cached entry"
        assert all(r["sign"] == 1 for r in recs)


class _NullPortfolio:
    """Test-only stand-in for ``Portfolio``; satisfies the type hint."""

    cash: float = 0.0


def test_pattern_strategy_rejects_invalid_size() -> None:
    from fundcloud.features.patterns import DoubleBottom
    from fundcloud.strategies import PatternStrategy

    with pytest.raises(ValueError, match=r"size must be in"):
        PatternStrategy(DoubleBottom(), size=0.0)
    with pytest.raises(ValueError, match=r"size must be in"):
        PatternStrategy(DoubleBottom(), size=2.0)
