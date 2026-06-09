"""Synthetic-fixture tests for ``fundcloud.research.events._opening_gap``.

Hand-built single-asset OHLCV frames that deterministically trigger one
bullish (gap-up continuation) and one bearish (gap-down continuation) opening
gap, frames that must NOT fire (a sub-threshold gap and a gap-up that closes
below its open), a flat non-triggering frame, an empty-input frame, and a
longer noisy frame with injected gaps fed to the mandatory prefix-invariance
proof. The gap test is bar-local (``open[t]`` vs ``close[t-1]``, both closed at
bar ``t``), so a passing ``assert_prefix_invariant`` is real evidence of online
causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._opening_gap import detect_opening_gap
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


def test_gap_up_continuation_fires() -> None:
    """A gap-up that closes above its own open emits a single bullish event."""
    # Bars 0-3 build an ATR of ~1 around price 100. Bar 4 opens at 102
    # (> close[3]=100 + 0.5*atr ~ 100.5) and closes 103 > open 102 -> gap-up.
    bars = _bars(
        [
            (100.0, 101.0, 99.0, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.0),  # 1
            (100.0, 101.0, 99.0, 100.0),  # 2
            (100.0, 101.0, 99.0, 100.0),  # 3  close 100, prior ATR ~ 1
            (102.0, 103.5, 101.5, 103.0),  # 4  gap up: open 102 > 100.5, close 103 > 102
        ]
    )

    out = detect_opening_gap(bars, asset="AAA", k=0.5, atr_n=3)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_opengap_up"
    assert row["confirmed_ts"] == bars.index[4]
    assert row["formation_end_ts"] == bars.index[4]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # Gap void runs from prior close (100) up to today's open (102).
    assert row["zone_lo"] == 100.0
    assert row["zone_hi"] == 102.0
    assert row["stop_ref_price"] == 101.5


def test_gap_down_continuation_fires() -> None:
    """A gap-down that closes below its own open emits a single bearish event."""
    # Bar 4 opens at 98 (< close[3]=100 - 0.5*atr ~ 99.5) and closes 97 < open 98.
    bars = _bars(
        [
            (100.0, 101.0, 99.0, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.0),  # 1
            (100.0, 101.0, 99.0, 100.0),  # 2
            (100.0, 101.0, 99.0, 100.0),  # 3  close 100, prior ATR ~ 1
            (98.0, 98.5, 96.5, 97.0),  # 4  gap down: open 98 < 99.5, close 97 < 98
        ]
    )

    out = detect_opening_gap(bars, asset="BBB", k=0.5, atr_n=3)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_opengap_dn"
    assert row["confirmed_ts"] == bars.index[4]
    # Gap void runs from today's open (98) up to prior close (100).
    assert row["zone_lo"] == 98.0
    assert row["zone_hi"] == 100.0
    assert row["stop_ref_price"] == 98.5


def test_small_gap_does_not_fire() -> None:
    """A gap smaller than ``k * atr`` is below threshold and emits nothing."""
    # Bar 4 opens at 100.3, only 0.3 above prior close — under 0.5*atr (~0.5).
    bars = _bars(
        [
            (100.0, 101.0, 99.0, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.0),  # 1
            (100.0, 101.0, 99.0, 100.0),  # 2
            (100.0, 101.0, 99.0, 100.0),  # 3
            (100.3, 101.0, 100.0, 100.8),  # 4  tiny gap, sub-threshold
        ]
    )

    out = detect_opening_gap(bars, asset="CCC", k=0.5, atr_n=3)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_gap_up_closing_below_open_does_not_fire() -> None:
    """A large gap-up that fails the continuation test (close < open) emits nothing."""
    # Bar 4 gaps up to open 102 but closes 101 < open -> no continuation.
    bars = _bars(
        [
            (100.0, 101.0, 99.0, 100.0),  # 0
            (100.0, 101.0, 99.0, 100.0),  # 1
            (100.0, 101.0, 99.0, 100.0),  # 2
            (100.0, 101.0, 99.0, 100.0),  # 3
            (102.0, 102.5, 100.5, 101.0),  # 4  gapped open but close 101 < open 102
        ]
    )

    out = detect_opening_gap(bars, asset="DDD", k=0.5, atr_n=3)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_opening_gap(empty, asset="EEE")
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

    # Inject a few clean opening-gap continuations so the proof has events to
    # compare. Each opens well past the prior close and closes through its open.
    for t, side in ((20, "up"), (35, "dn"), (50, "up")):
        if side == "up":
            open_[t] = close[t - 1] + 4.0
            close[t] = open_[t] + 1.0
            high[t] = close[t] + 0.3
            low[t] = open_[t] - 0.3
        else:
            open_[t] = close[t - 1] - 4.0
            close[t] = open_[t] - 1.0
            low[t] = close[t] - 0.3
            high[t] = open_[t] + 0.3

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(detect_opening_gap, bars, k=0.5, atr_n=5, asset="FFF")
