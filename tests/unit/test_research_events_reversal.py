"""Synthetic-fixture tests for ``fundcloud.research.events._reversal``.

Hand-built single-asset OHLCV frames that deterministically trigger one bullish
and one bearish outside-bar key reversal, non-triggering frames (an inside bar
and an outside bar that closes back inside the prior range), an empty-input
frame, and a longer noisy frame fed to the mandatory prefix-invariance proof.
Because the event is bar-local (it reads only bars ``t-1`` and ``t``), a passing
``assert_prefix_invariant`` is real evidence of online causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._reversal import detect_key_reversal
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


def test_bullish_key_reversal_fires() -> None:
    """An engulfing bar closing strongly above the prior high emits ev_keyrev_up."""
    # Bar 2 is an outside bar: high 102 > prev 101, low 98 < prev 99, close 101.5
    # > prev high 101, clv = (101.5 - 98) / 4 = 0.875 >= clv_min -> bullish.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.2),  # 1  prior bar
            (100.5, 102.0, 98.0, 101.5),  # 2  outside + close > 101 + high clv
            (101.5, 101.8, 101.0, 101.4),  # 3  next bar (execution)
        ]
    )

    out = detect_key_reversal(bars, asset="AAA", clv_min=0.7, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_keyrev_up"
    assert row["confirmed_ts"] == bars.index[2]
    assert row["formation_end_ts"] == bars.index[2]
    assert row["execution_ts"] == bars.index[3]
    assert row["entry_ref_price"] == 101.5
    assert row["stop_ref_price"] == 98.0
    assert pd.isna(row["zone_lo"])
    assert pd.isna(row["zone_hi"])
    assert np.isfinite(row["atr_at_confirm"])


def test_bearish_key_reversal_fires() -> None:
    """An engulfing bar closing strongly below the prior low emits ev_keyrev_dn."""
    # Bar 2 is an outside bar: high 102 > prev 101, low 98 < prev 99, close 98.5
    # < prev low 99, clv = (98.5 - 98) / 4 = 0.125 <= 1 - clv_min -> bearish.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.2),  # 1  prior bar
            (100.5, 102.0, 98.0, 98.5),  # 2  outside + close < 99 + low clv
            (98.5, 99.0, 98.2, 98.7),  # 3  next bar (execution)
        ]
    )

    out = detect_key_reversal(bars, asset="BBB", clv_min=0.7, atr_n=2)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_keyrev_dn"
    assert row["confirmed_ts"] == bars.index[2]
    assert row["execution_ts"] == bars.index[3]
    assert row["entry_ref_price"] == 98.5
    assert row["stop_ref_price"] == 102.0


def test_inside_bar_does_not_fire() -> None:
    """An inside bar (not engulfing the prior range) emits nothing."""
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.2),  # 1  prior bar
            (100.2, 100.8, 99.5, 100.6),  # 2  inside bar: high 100.8 < 101, low 99.5 > 99
            (100.6, 100.9, 100.0, 100.7),  # 3
        ]
    )

    out = detect_key_reversal(bars, asset="CCC", clv_min=0.7, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_outside_bar_closing_inside_does_not_fire() -> None:
    """An outside bar that closes back inside the prior range emits nothing."""
    # Bar 2 engulfs (high 102 > 101, low 98 < 99) but closes at 100.0 — neither
    # above prev high 101 nor below prev low 99, so no branch fires.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.2),  # 1  prior bar
            (100.5, 102.0, 98.0, 100.0),  # 2  outside but close inside prior range
            (100.0, 100.3, 99.5, 100.1),  # 3
        ]
    )

    out = detect_key_reversal(bars, asset="DDD", clv_min=0.7, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_key_reversal(empty, asset="EEE")
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

    # Inject a few clean key reversals so the proof has events to compare. Each
    # is built purely from bars t-1 and t (engulf + decisive close), so they are
    # legitimately confirmable online.
    for t, side in ((20, "bull"), (35, "bear"), (50, "bull")):
        if side == "bull":
            high[t] = high[t - 1] + 2.0
            low[t] = low[t - 1] - 2.0
            close[t] = high[t] - 0.1  # close near the top, above prior high
            open_[t] = low[t] + 0.2
        else:
            high[t] = high[t - 1] + 2.0
            low[t] = low[t - 1] - 2.0
            close[t] = low[t] + 0.1  # close near the bottom, below prior low
            open_[t] = high[t] - 0.2

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(detect_key_reversal, bars, clv_min=0.7, atr_n=5, asset="FFF")
