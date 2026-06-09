"""Synthetic-fixture tests for ``fundcloud.research.events._sr_touch``.

Hand-built single-asset OHLCV frames that deterministically trigger one
bullish (support) and one bearish (resistance) touch-and-hold bounce, a
flat non-triggering frame, a touch-but-break frame that must NOT fire, and a
longer frame fed to the mandatory prefix-invariance proof. Every fixture keeps
the pivot that builds the touched level neighbour-locked strictly before the
touch bar, so a passing ``assert_prefix_invariant`` is real evidence of online
causality, not an artefact of placement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._sr_touch import detect_sr_touch_bounce
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


def test_bullish_support_bounce_fires() -> None:
    """A held pivot-low touch emits a single neighbour-locked bullish event."""
    # Index 1 is a pivot low (low=98) confirmed at index 2 with pivot_k=1.
    # Bar 4 dips toward 98 within eps*atr (but does NOT pierce below) and closes
    # above -> support touch-and-hold bounce.
    bars = _bars(
        [
            (100.0, 101.0, 99.5, 100.5),  # 0
            (100.5, 100.8, 98.0, 100.0),  # 1  pivot low @ 98
            (100.0, 101.0, 99.8, 100.7),  # 2  pivot confirmed here
            (100.7, 101.5, 100.0, 101.0),  # 3
            (101.0, 101.2, 98.05, 100.8),  # 4  touch: low 98.05 within band, close > 98
        ]
    )

    out = detect_sr_touch_bounce(bars, asset="AAA", pivot_k=1, eps=0.10, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_sr_bounce_up"
    assert row["confirmed_ts"] == bars.index[4]
    assert row["formation_end_ts"] == bars.index[4]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # Band straddles the touched level: zone_lo < L < zone_hi.
    assert row["zone_lo"] < 98.0 < row["zone_hi"]
    assert row["stop_ref_price"] == 98.05


def test_bearish_resistance_bounce_fires() -> None:
    """A held pivot-high touch emits a single neighbour-locked bearish event."""
    # Index 1 is a pivot high (high=102) confirmed at index 2 with pivot_k=1.
    # Bar 4 pushes toward 102 within eps*atr (but does NOT pierce above) and closes
    # below -> resistance touch-and-hold bounce.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),  # 0
            (100.0, 102.0, 99.8, 100.2),  # 1  pivot high @ 102
            (100.2, 100.6, 99.9, 100.3),  # 2  pivot confirmed here
            (100.3, 100.8, 99.7, 100.5),  # 3
            (100.5, 101.95, 100.4, 101.0),  # 4  touch: high 101.95 within band, close < 102
        ]
    )

    out = detect_sr_touch_bounce(bars, asset="BBB", pivot_k=1, eps=0.10, atr_n=2)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_sr_bounce_dn"
    assert row["confirmed_ts"] == bars.index[4]
    assert row["zone_lo"] < 102.0 < row["zone_hi"]
    assert row["stop_ref_price"] == 101.95


def test_touch_but_close_below_does_not_fire() -> None:
    """A bar that reaches the support band but closes BELOW it is a break, not a hold."""
    # Same support setup as the bullish case, but bar 4 closes below 98 -> no hold.
    bars = _bars(
        [
            (100.0, 101.0, 99.5, 100.5),  # 0
            (100.5, 100.8, 98.0, 100.0),  # 1  pivot low @ 98
            (100.0, 101.0, 99.8, 100.7),  # 2  pivot confirmed here
            (100.7, 101.5, 100.0, 101.0),  # 3
            (101.0, 101.2, 97.5, 97.6),  # 4  touch band but close 97.6 < 98 -> break
        ]
    )

    out = detect_sr_touch_bounce(bars, asset="CCC", pivot_k=1, eps=0.10, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_no_touch_yields_empty_frame() -> None:
    """A frame whose bars never reach any level's band emits nothing (right shape)."""
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.6, 99.6, 100.1),
            (100.1, 100.7, 99.7, 100.2),
            (100.2, 100.8, 99.8, 100.3),
            (100.3, 100.9, 99.9, 100.4),
        ]
    )

    out = detect_sr_touch_bounce(bars, asset="DDD", pivot_k=1, eps=0.10, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_sr_touch_bounce(empty, asset="EEE")
    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_prefix_invariance_no_future_leak() -> None:
    """Prove online causality: prefix runs reproduce the full run's confirmed events."""
    rng = np.random.default_rng(7)
    n = 60
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, size=n))
    high = close + rng.uniform(0.2, 0.6, size=n)
    low = close - rng.uniform(0.2, 0.6, size=n)
    open_ = close - rng.normal(0.0, 0.1, size=n)

    # Inject a few clean touch-and-hold bounces so the proof has events to compare.
    for t, side in ((20, "bull"), (35, "bear"), (50, "bull")):
        if side == "bull":
            # Dip toward the recent low without piercing far below; hold on close.
            support = min(low[t - 5 : t])
            low[t] = support + 0.05
            close[t] = support + 1.0
            high[t] = close[t] + 0.3
        else:
            # Push toward the recent high without piercing far above; reject on close.
            resistance = max(high[t - 5 : t])
            high[t] = resistance - 0.05
            close[t] = resistance - 1.0
            low[t] = close[t] - 0.3

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(
        detect_sr_touch_bounce, bars, pivot_k=2, eps=0.10, atr_n=5, asset="FFF"
    )
