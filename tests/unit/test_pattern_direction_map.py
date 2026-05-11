"""Tests for ``fundcloud.metrics.pattern_direction``.

Covers the per-pattern mean-forward-return computation, the
``min_samples`` fallback, the ``null_threshold`` indecision gate, and
the strategy-time integration (events frame → direction map →
``PatternStrategy`` consumes it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    Direction,
    Pattern,
)
from fundcloud.metrics import pattern_direction as pd_


def _bars(asset: str, n: int, *, drift_per_bar: float, seed: int) -> pd.DataFrame:
    """Synthetic OHLCV with a known per-bar drift."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift_per_bar, 0.005, size=n)))
    high = close * (1.0 + rng.uniform(0.001, 0.005, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.005, size=n))
    open_ = close.copy()
    volume = np.full(n, 1_000_000.0)
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            ("open", asset): open_,
            ("high", asset): high,
            ("low", asset): low,
            ("close", asset): close,
            ("volume", asset): volume,
        },
        index=index,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def _event(asset: str, ts: pd.Timestamp, *, pattern: Pattern, entry: float) -> dict:
    return {
        "pattern": pattern,
        "asset": asset,
        "formation_start": ts - pd.Timedelta(days=5),
        "formation_end": ts,
        "breakout_ts": ts,
        "breakout_level": entry,
        "formation_height": 1.0,
        "target_price": float("nan"),
        "stop_price": float("nan"),
        "quality": 50.0,
        "variant": None,
        "pivots": [],
        "meta": {},
    }


def _events_for_pattern(
    bars: pd.DataFrame,
    asset: str,
    *,
    pattern: Pattern,
    positions: list[int],
) -> pd.DataFrame:
    rows = []
    close = bars[("close", asset)]
    for pos in positions:
        ts = bars.index[pos]
        rows.append(_event(asset, ts, pattern=pattern, entry=float(close.iloc[pos])))
    return pd.DataFrame(rows, columns=EVENTS_COLUMNS)


def test_empty_events_returns_empty_map():
    bars = _bars("AAA", n=200, drift_per_bar=0.001, seed=1)
    out = pd_.direction_map_from_outcomes(pd.DataFrame(columns=EVENTS_COLUMNS), bars)
    assert out == {}


def test_invalid_horizon_raises():
    bars = _bars("AAA", n=50, drift_per_bar=0.001, seed=1)
    with pytest.raises(ValueError, match="horizon"):
        pd_.direction_map_from_outcomes(pd.DataFrame(columns=EVENTS_COLUMNS), bars, horizon=0)


def test_invalid_min_samples_raises():
    bars = _bars("AAA", n=50, drift_per_bar=0.001, seed=1)
    with pytest.raises(ValueError, match="min_samples"):
        pd_.direction_map_from_outcomes(pd.DataFrame(columns=EVENTS_COLUMNS), bars, min_samples=0)


def test_invalid_null_threshold_raises():
    bars = _bars("AAA", n=50, drift_per_bar=0.001, seed=1)
    with pytest.raises(ValueError, match="null_threshold"):
        pd_.direction_map_from_outcomes(
            pd.DataFrame(columns=EVENTS_COLUMNS), bars, null_threshold=-0.01
        )


def test_uptrend_yields_bullish_when_above_min_samples():
    """50 events scattered through a strong uptrend → mean forward return
    > 0 → BULLISH."""
    bars = _bars("AAA", n=400, drift_per_bar=0.005, seed=42)
    positions = list(range(50, 350, 6))  # 50 events
    events = _events_for_pattern(bars, "AAA", pattern=Pattern.DOUBLE_TOP, positions=positions)
    assert len(events) >= 30  # sanity: above min_samples default
    out = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    assert out["double_top"] is Direction.BULLISH


def test_downtrend_yields_bearish_when_above_min_samples():
    """Same fixture, negative drift → BEARISH."""
    bars = _bars("BBB", n=400, drift_per_bar=-0.005, seed=42)
    positions = list(range(50, 350, 6))
    events = _events_for_pattern(bars, "BBB", pattern=Pattern.DOUBLE_TOP, positions=positions)
    out = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    assert out["double_top"] is Direction.BEARISH


def test_below_min_samples_falls_back_to_default():
    """Five events isn't enough to trust the mean — should return the
    user-supplied default regardless of which way the synthetic series
    drifts."""
    bars = _bars("CCC", n=400, drift_per_bar=0.005, seed=11)
    events = _events_for_pattern(
        bars, "CCC", pattern=Pattern.DOUBLE_TOP, positions=[50, 100, 150, 200, 250]
    )
    # min_samples=30 (default), so 5 events trigger the fallback.
    out_default_long = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    assert out_default_long["double_top"] is Direction.BULLISH  # default = BULLISH
    out_default_short = pd_.direction_map_from_outcomes(
        events, bars, horizon=20, default=Direction.BEARISH
    )
    assert out_default_short["double_top"] is Direction.BEARISH


def test_null_threshold_gates_low_magnitude_means():
    """A tiny but positive mean (under null_threshold) should fall back
    to default rather than committing to BULLISH on noise."""
    bars = _bars("DDD", n=400, drift_per_bar=0.0001, seed=5)
    positions = list(range(50, 350, 6))
    events = _events_for_pattern(bars, "DDD", pattern=Pattern.DOUBLE_TOP, positions=positions)
    # null_threshold=1.0 (100% mean return) is impossible to clear → all
    # patterns route to the default.
    out = pd_.direction_map_from_outcomes(
        events, bars, horizon=20, default=Direction.BEARISH, null_threshold=1.0
    )
    assert out["double_top"] is Direction.BEARISH


def test_multiple_patterns_classified_independently():
    """Two patterns on the same bars frame should each get their own
    classification based on their own event timestamps."""
    bars = _bars("EEE", n=400, drift_per_bar=0.005, seed=42)
    early_positions = list(range(50, 200, 4))  # 38 events in early uptrend
    late_positions = list(range(220, 380, 4))  # 40 events in later uptrend
    rows = _events_for_pattern(
        bars, "EEE", pattern=Pattern.DOUBLE_TOP, positions=early_positions
    ).to_dict("records") + _events_for_pattern(
        bars, "EEE", pattern=Pattern.HEAD_AND_SHOULDERS, positions=late_positions
    ).to_dict("records")
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)
    out = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    # Both patterns saw uptrending forward returns — both → BULLISH.
    assert out["double_top"] is Direction.BULLISH
    assert out["head_and_shoulders"] is Direction.BULLISH


def test_events_off_grid_are_silently_dropped():
    """Events with breakout_ts outside the bars index don't crash and
    don't contribute to the count."""
    bars = _bars("FFF", n=400, drift_per_bar=0.005, seed=42)
    valid_positions = list(range(50, 350, 6))
    rows = _events_for_pattern(
        bars, "FFF", pattern=Pattern.DOUBLE_TOP, positions=valid_positions
    ).to_dict("records")
    # Add one event with an off-grid breakout timestamp.
    bogus = _event(
        "FFF", pd.Timestamp("1999-01-01", tz="UTC"), pattern=Pattern.DOUBLE_TOP, entry=100.0
    )
    rows.append(bogus)
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)
    out = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    assert "double_top" in out
    means = pd_.mean_forward_returns(events, bars, horizon=20)
    assert means["double_top"]["count"] == len(valid_positions)


def test_strategy_consumes_direction_map():
    """End-to-end: events → direction_map → PatternStrategy treats the
    bearish-mapped pattern as short and skips it (long-only execution)."""
    from fundcloud.features.patterns import DoubleTop
    from fundcloud.strategies import PatternStrategy

    bars = _bars("GGG", n=400, drift_per_bar=-0.005, seed=42)
    positions = list(range(50, 350, 6))
    events = _events_for_pattern(bars, "GGG", pattern=Pattern.DOUBLE_TOP, positions=positions)
    dmap = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    assert dmap["double_top"] is Direction.BEARISH

    indicator = DoubleTop(min_quality=0.0)
    indicator.events = lambda _bars: events  # type: ignore[method-assign]
    strat = PatternStrategy(indicator, direction_map=dmap)

    class _NullPortfolio:
        cash: float = 0.0

    strat.init(bars, _NullPortfolio())  # type: ignore[arg-type]
    # Long-only execution: BEARISH-mapped events are skipped.
    assert strat._events_by_asset == {}


def test_horizon_off_lookahead_drops_terminal_events():
    """An event whose horizon exceeds the bars series should be dropped
    from the count — same convention as feature_quality.evaluate."""
    bars = _bars("HHH", n=100, drift_per_bar=0.001, seed=3)
    # Event 5 bars from end with horizon=20 → no lookahead → drop.
    events = _events_for_pattern(
        bars,
        "HHH",
        pattern=Pattern.DOUBLE_TOP,
        positions=[len(bars) - 5],
    )
    means = pd_.mean_forward_returns(events, bars, horizon=20)
    assert "double_top" not in means
