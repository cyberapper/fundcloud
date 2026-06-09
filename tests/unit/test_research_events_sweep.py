"""Synthetic-fixture tests for ``fundcloud.research.events._sweep``.

Hand-built single-asset OHLCV frames that deterministically trigger one
bullish (support) and one bearish (resistance) sweep failure, a flat
non-triggering frame, and a longer frame fed to the mandatory
prefix-invariance proof. Every fixture keeps the pivot that builds the swept
level neighbour-locked strictly before the sweep bar, so a passing
``assert_prefix_invariant`` is real evidence of online causality, not an
artefact of placement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._sweep import detect_sweep_fail
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


def test_bullish_support_sweep_fires() -> None:
    """A reclaimed pivot-low sweep emits a single neighbour-locked bullish event."""
    # Index 1 is a pivot low (low=98) confirmed at index 2 with pivot_k=1.
    # Bar 4 wicks below 98 by > eps*atr and closes back above -> support sweep.
    bars = _bars(
        [
            (100.0, 101.0, 99.5, 100.5),  # 0
            (100.5, 100.8, 98.0, 100.0),  # 1  pivot low @ 98
            (100.0, 101.0, 99.8, 100.7),  # 2  pivot confirmed here
            (100.7, 101.5, 100.0, 101.0),  # 3
            (101.0, 101.2, 96.0, 100.8),  # 4  sweep: low 96 < 98, close 100.8 > 98
        ]
    )

    out = detect_sweep_fail(bars, asset="AAA", pivot_k=1, eps=0.10, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_sweep_up"
    assert row["confirmed_ts"] == bars.index[4]
    assert row["formation_end_ts"] == bars.index[4]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # zone_hi is the swept level; zone_lo is one margin below it.
    assert row["zone_hi"] == 98.0
    assert row["zone_lo"] < 98.0
    assert row["stop_ref_price"] == 96.0


def test_bearish_resistance_sweep_fires() -> None:
    """A rejected pivot-high sweep emits a single neighbour-locked bearish event."""
    # Index 1 is a pivot high (high=102) confirmed at index 2 with pivot_k=1.
    # Bar 4 wicks above 102 by > eps*atr and closes back below -> resistance sweep.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),  # 0
            (100.0, 102.0, 99.8, 100.2),  # 1  pivot high @ 102
            (100.2, 100.6, 99.9, 100.3),  # 2  pivot confirmed here
            (100.3, 100.8, 99.7, 100.5),  # 3
            (100.5, 105.0, 100.4, 101.0),  # 4  sweep: high 105 > 102, close 101 < 102
        ]
    )

    out = detect_sweep_fail(bars, asset="BBB", pivot_k=1, eps=0.10, atr_n=2)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_sweep_dn"
    assert row["confirmed_ts"] == bars.index[4]
    assert row["zone_lo"] == 102.0
    assert row["zone_hi"] > 102.0
    assert row["stop_ref_price"] == 105.0


def test_no_sweep_yields_empty_frame() -> None:
    """A frame with no level pierced-and-reclaimed emits nothing (but right shape)."""
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.6, 99.6, 100.1),
            (100.1, 100.7, 99.7, 100.2),
            (100.2, 100.8, 99.8, 100.3),
            (100.3, 100.9, 99.9, 100.4),
        ]
    )

    out = detect_sweep_fail(bars, asset="CCC", pivot_k=1, eps=0.10, atr_n=2)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_sweep_fail(empty, asset="DDD")
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

    # Inject a few clean sweep failures so the proof has events to compare.
    for t, side in ((20, "bull"), (35, "bear"), (50, "bull")):
        if side == "bull":
            low[t] = min(low[t - 5 : t]) - 3.0
            close[t] = max(close[t - 5 : t]) + 0.5
            high[t] = close[t] + 0.3
        else:
            high[t] = max(high[t - 5 : t]) + 3.0
            close[t] = min(close[t - 5 : t]) - 0.5
            low[t] = close[t] - 0.3

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(
        detect_sweep_fail, bars, pivot_k=2, eps=0.10, atr_n=5, asset="EEE"
    )
