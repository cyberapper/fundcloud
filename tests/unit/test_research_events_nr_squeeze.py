"""Synthetic-fixture tests for ``fundcloud.research.events._nr_squeeze``.

Hand-built single-asset OHLCV frames that deterministically trigger one
narrowest-range (NRn) contraction, a strictly-expanding frame that never
contracts (no fire), an empty frame, and a longer noisy series fed to the
mandatory prefix-invariance proof. ``ev_nr_squeeze`` is a *neutral* single-id
detector (no ``_up`` / ``_dn`` branch) reading only a backward window, so a
passing ``assert_prefix_invariant`` is real evidence of online causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._nr_squeeze import detect_nr_squeeze
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


def _row_for_range(rng: float) -> tuple[float, float, float, float]:
    """A bar of span ``rng`` anchored at low=100 with open/close inside the range."""
    low = 100.0
    high = low + rng
    return (low + rng * 0.25, high, low, low + rng * 0.5)


def test_nrn_contraction_fires_once_on_last_bar() -> None:
    """The narrowest-of-7 bar emits a single neutral contraction event.

    Ranges ``[5, 1, 4, 3, 2.5, 2, 1.5, 0.9]``: the early narrow bar (range 1.0)
    keeps index 6 (range 1.5) from being narrowest-of-7, so only the final bar
    (range 0.9 <= prior-window min 1.0) fires.
    """
    ranges = [5.0, 1.0, 4.0, 3.0, 2.5, 2.0, 1.5, 0.9]
    bars = _bars([_row_for_range(r) for r in ranges])

    out = detect_nr_squeeze(bars, asset="AAA", n=7, atr_n=3)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_nr_squeeze"
    assert row["confirmed_ts"] == bars.index[7]
    assert row["formation_end_ts"] == bars.index[7]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # Neutral detector carries no zone, stop or quality.
    assert pd.isna(row["zone_lo"])
    assert pd.isna(row["zone_hi"])
    assert pd.isna(row["stop_ref_price"])
    assert pd.isna(row["quality"])
    # atr_at_confirm is the recorded volatility unit (finite at the firing bar).
    assert np.isfinite(row["atr_at_confirm"])


def test_no_contraction_yields_empty_frame() -> None:
    """A strictly-expanding frame never contracts -> emits nothing (right shape)."""
    ranges = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    bars = _bars([_row_for_range(r) for r in ranges])

    out = detect_nr_squeeze(bars, asset="CCC", n=7, atr_n=3)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_nr_squeeze(empty, asset="DDD")
    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_prefix_invariance_no_future_leak() -> None:
    """Prove online causality: prefix runs reproduce the full run's confirmed events."""
    rng = np.random.default_rng(11)
    n = 60
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, size=n))
    span = rng.uniform(0.4, 1.2, size=n)
    high = close + span / 2.0
    low = close - span / 2.0
    open_ = close - rng.normal(0.0, 0.1, size=n)

    # Inject a few clean contraction bars (very narrow span) so the proof has
    # events to compare. Each is far narrower than its trailing window.
    for t in (18, 33, 49):
        mid = close[t]
        high[t] = mid + 0.02
        low[t] = mid - 0.02

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(detect_nr_squeeze, bars, n=7, atr_n=5, asset="EEE")
