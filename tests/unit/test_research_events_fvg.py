"""Tests for the three-candle fair-value-gap detector (``ev_gap_imb_3c``).

Covers a guaranteed bullish trigger, a guaranteed bearish trigger, a flat
non-triggering frame, and the mandatory prefix-invariance proof that the
detector reads no future bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._fvg import EVENT_ID, detect_fvg
from fundcloud.research.events.schema import OBSERVATION_COLUMNS


def _index(n: int) -> pd.DatetimeIndex:
    """``n`` consecutive daily tz-aware UTC stamps."""
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame from ``(open, high, low, close)`` tuples (volume=1)."""
    arr = np.asarray(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": np.ones(len(rows)),
        },
        index=_index(len(rows)),
    )


def _flat(n: int, *, level: float = 100.0) -> list[tuple[float, float, float, float]]:
    """``n`` calm, fully-overlapping bars used as ATR warmup / filler."""
    return [(level, level + 0.5, level - 0.5, level) for _ in range(n)]


def test_bullish_gap_fires() -> None:
    # atr_n=3 -> first eligible centre is t=3. Build warmup, then a wide, decisive
    # up-candle at t-1 whose neighbours leave a void: low[t] > high[t-2].
    rows = _flat(3, level=100.0)
    rows.append((100.0, 100.5, 99.5, 100.0))  # t-2 (index 3): high = 100.5
    rows.append((100.5, 110.0, 100.0, 109.5))  # t-1 (index 4): wide bullish body
    rows.append((110.0, 111.0, 101.0, 110.5))  # t   (index 5): low = 101.0 > 100.5
    bars = _bars(rows)

    out = detect_fvg(bars, asset="TEST", atr_n=3)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    ev = out.iloc[0]
    assert ev["event_id"] == EVENT_ID
    assert ev["direction"] == "bullish"
    assert ev["confirmed_ts"] == bars.index[5]
    assert ev["formation_end_ts"] == bars.index[5]
    assert ev["zone_lo"] == 100.5  # high[t-2]
    assert ev["zone_hi"] == 101.0  # low[t]
    # t is the last bar -> no execution bar.
    assert pd.isna(ev["execution_ts"])
    assert pd.isna(ev["entry_ref_price"])


def test_bearish_gap_fires() -> None:
    # Mirror image: wide bearish middle bar, high[t] < low[t-2].
    rows = _flat(3, level=100.0)
    rows.append((100.0, 100.5, 99.5, 100.0))  # t-2 (index 3): low = 99.5
    rows.append((99.5, 100.0, 90.0, 90.5))  # t-1 (index 4): wide bearish body
    rows.append((90.0, 99.0, 89.0, 90.0))  # t   (index 5): high = 99.0 < 99.5
    bars = _bars(rows)

    out = detect_fvg(bars, asset="TEST", atr_n=3)

    assert len(out) == 1
    ev = out.iloc[0]
    assert ev["direction"] == "bearish"
    assert ev["confirmed_ts"] == bars.index[5]
    assert ev["zone_lo"] == 99.0  # high[t]
    assert ev["zone_hi"] == 99.5  # low[t-2]


def test_execution_fields_when_next_bar_exists() -> None:
    # Same bullish setup but with a trailing bar so execution_ts/entry resolve.
    rows = _flat(3, level=100.0)
    rows.append((100.0, 100.5, 99.5, 100.0))
    rows.append((100.5, 110.0, 100.0, 109.5))
    rows.append((110.0, 111.0, 101.0, 110.5))  # t (index 5): the gap
    rows.append((112.0, 113.0, 111.0, 112.5))  # t+1 (index 6): execution bar
    bars = _bars(rows)

    out = detect_fvg(bars, asset="TEST", atr_n=3)

    assert len(out) == 1
    ev = out.iloc[0]
    assert ev["direction"] == "bullish"
    assert ev["confirmed_ts"] == bars.index[5]
    assert ev["execution_ts"] == bars.index[6]
    assert ev["entry_ref_price"] == 112.0  # open[t+1]


def test_no_gap_yields_empty() -> None:
    # Calm, fully-overlapping bars: no non-overlapping outer bars, no impulse.
    bars = _bars(_flat(20, level=100.0))

    out = detect_fvg(bars, asset="TEST", atr_n=3)

    assert out.empty
    assert list(out.columns) == list(OBSERVATION_COLUMNS)


def test_empty_input_yields_empty_frame() -> None:
    empty = pd.DataFrame(
        {c: [] for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([], tz="UTC"),
    )

    out = detect_fvg(empty, asset="TEST")

    assert out.empty
    assert list(out.columns) == list(OBSERVATION_COLUMNS)


def test_prefix_invariance_no_future_leak() -> None:
    # A longer mixed frame with calm warmup plus two embedded gaps (one bullish,
    # one bearish). assert_prefix_invariant proves truncating the series never
    # changes the events confirmed at/<= each cutoff.
    rows = _flat(14, level=100.0)
    # Bullish gap centred at index 15.
    rows.append((100.0, 100.5, 99.5, 100.0))  # 14: t-2
    rows.append((100.5, 110.0, 100.0, 109.5))  # 15: wide up-candle
    rows.append((110.0, 111.0, 101.0, 110.5))  # 16: low > high[14]
    # Calm filler.
    rows.extend(_flat(5, level=110.0))  # 17..21
    # Bearish gap centred near index 23.
    rows.append((110.0, 110.5, 109.5, 110.0))  # 22: t-2
    rows.append((109.5, 110.0, 100.0, 100.5))  # 23: wide down-candle
    rows.append((100.0, 109.0, 99.0, 100.0))  # 24: high < low[22]
    rows.extend(_flat(5, level=100.0))  # 25..29
    bars = _bars(rows)

    # Sanity: at least one of each direction is present in the full run.
    full = detect_fvg(bars, asset="TEST", atr_n=14)
    assert set(full["direction"]) >= {"bullish", "bearish"}

    assert_prefix_invariant(detect_fvg, bars, asset="TEST", atr_n=14)
