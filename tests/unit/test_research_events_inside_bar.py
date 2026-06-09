"""Synthetic-fixture tests for ``fundcloud.research.events._inside_bar``.

Hand-built single-asset OHLCV frames that deterministically trigger a neutral
inside-bar contraction, a strict-mode equal-extreme bar that does NOT fire, a
frame with no contraction, an empty frame, and a longer noisy series fed to the
mandatory prefix-invariance proof. Inside bars are bar-local (no pivots), so a
passing ``assert_prefix_invariant`` is direct evidence of online causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._inside_bar import detect_inside_bar
from fundcloud.research.events.schema import OBSERVATION_COLUMNS


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a single-asset OHLCV frame from ``(open, high, low, close)`` rows."""
    arr = np.asarray(rows, dtype=float)
    index = pd.date_range("2021-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": np.full(len(rows), 1_000.0),
        },
        index=index,
    )


def test_inside_bar_fires_neutral() -> None:
    """A bar wholly inside its mother bar emits one neutral ``ev_inside_bar`` row."""
    # atr_n=2 -> atr[1] is the seed, so bar t=2 has a finite atr_at_confirm.
    # Bar 0 is narrow so the mother bar (index 1) is NOT inside it; only bar 2 fires.
    # Mother bar (index 1) high=101 low=99; inside bar (index 2) high=100.5 low=99.5.
    bars = _bars(
        [
            (100.0, 100.4, 99.6, 100.0),  # 0  narrow -> bar 1 is not inside it
            (100.0, 101.0, 99.0, 100.0),  # 1  mother bar: range [99, 101]
            (100.0, 100.5, 99.5, 100.0),  # 2  inside -> fires
        ]
    )

    out = detect_inside_bar(bars, asset="AAA", atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_inside_bar"
    assert row["confirmed_ts"] == bars.index[2]
    assert row["formation_end_ts"] == bars.index[2]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # Zone is the mother-bar range; no directional stop.
    assert row["zone_lo"] == 99.0
    assert row["zone_hi"] == 101.0
    assert pd.isna(row["stop_ref_price"])
    assert pd.isna(row["quality"])


def test_inside_bar_records_next_bar_execution() -> None:
    """An inside bar with a following bar records the next bar's open as the fill."""
    bars = _bars(
        [
            (100.0, 100.4, 99.6, 100.0),  # 0  narrow -> bar 1 is not inside it
            (100.0, 101.0, 99.0, 100.0),  # 1  mother bar
            (100.0, 100.5, 99.5, 100.0),  # 2  inside -> fires
            (100.3, 103.0, 97.0, 101.0),  # 3  next bar -> fill open 100.3
        ]
    )

    out = detect_inside_bar(bars, asset="BBB", atr_n=2)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["confirmed_ts"] == bars.index[2]
    assert row["execution_ts"] == bars.index[3]
    assert row["entry_ref_price"] == 100.3


def test_higher_high_does_not_fire() -> None:
    """A bar that breaks above the mother-bar high is not an inside bar."""
    bars = _bars(
        [
            (100.0, 100.4, 99.6, 100.0),  # 0  narrow -> bar 1 is not inside it
            (100.0, 101.0, 99.0, 100.0),  # 1  mother bar: high 101
            (100.0, 101.5, 99.5, 100.5),  # 2  higher high (101.5 > 101) -> no fire
        ]
    )

    out = detect_inside_bar(bars, asset="CCC", atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_strict_rejects_equal_high() -> None:
    """In strict mode a bar that equals the mother-bar high does not fire."""
    # Inside bar shares the mother bar's high (101 == 101): non-strict fires,
    # strict does not.
    bars = _bars(
        [
            (100.0, 100.4, 99.6, 100.0),  # 0  narrow -> bar 1 is not inside it
            (100.0, 101.0, 99.0, 100.0),  # 1  mother bar: high 101
            (100.0, 101.0, 99.5, 100.0),  # 2  equal high (101 == 101), low inside
        ]
    )

    assert len(detect_inside_bar(bars, asset="DDD", atr_n=2, strict=False)) == 1
    assert detect_inside_bar(bars, asset="DDD", atr_n=2, strict=True).empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_inside_bar(empty, asset="EEE")
    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_prefix_invariance_no_future_leak() -> None:
    """Prove online causality: prefix runs reproduce the full run's confirmed events."""
    rng = np.random.default_rng(11)
    n = 60
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, size=n))
    high = close + rng.uniform(0.2, 0.6, size=n)
    low = close - rng.uniform(0.2, 0.6, size=n)
    open_ = close - rng.normal(0.0, 0.1, size=n)

    # Inject a few clean inside bars: shrink bar t entirely inside its mother bar.
    for t in (20, 35, 50):
        span = high[t - 1] - low[t - 1]
        mid = (high[t - 1] + low[t - 1]) / 2.0
        high[t] = mid + span * 0.2
        low[t] = mid - span * 0.2
        close[t] = mid
        open_[t] = mid

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(detect_inside_bar, bars, atr_n=5, strict=False, asset="FFF")
