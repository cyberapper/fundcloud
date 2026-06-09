"""Synthetic-fixture tests for ``fundcloud.research.events._donchian``.

Hand-built single-asset OHLCV frames that deterministically trigger one upward
and one downward N-day Donchian channel breakout, a frame that pokes the channel
intrabar but closes back inside (no break), an empty-input frame, and a longer
noisy series fed to the mandatory prefix-invariance proof. Because the channel is
a *trailing* extreme over ``[t-N, t)`` (not a centred pivot), it reads no future
bars — a passing ``assert_prefix_invariant`` is real evidence of online causality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._donchian import detect_donchian
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


def _flat_then(breakout: tuple[float, float, float, float]) -> pd.DataFrame:
    """22 flat bars (high 100.5 / low 99.5 / close 100) then one ``breakout`` bar."""
    flat = [(100.0, 100.5, 99.5, 100.0)] * 22
    return _bars([*flat, breakout])


def test_up_break_fires() -> None:
    """A close above the trailing 20-bar high emits a single bullish breakout."""
    # Bar 22 closes at 103 > prior-20-bar high (100.5); trailing window excludes t.
    bars = _flat_then((100.0, 103.5, 100.0, 103.0))

    out = detect_donchian(bars, asset="AAA", N=20, buf=0.0, atr_n=5)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_donchian_up"
    assert row["confirmed_ts"] == bars.index[22]
    assert row["formation_end_ts"] == bars.index[22]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # Stop sits at the opposite (lower) channel boundary; zones unused.
    assert row["stop_ref_price"] == 99.5
    assert pd.isna(row["zone_lo"])
    assert pd.isna(row["zone_hi"])
    assert pd.isna(row["quality"])


def test_down_break_fires() -> None:
    """A close below the trailing 20-bar low emits a single bearish breakout."""
    # Bar 22 closes at 97 < prior-20-bar low (99.5); trailing window excludes t.
    bars = _flat_then((100.0, 100.0, 96.5, 97.0))

    out = detect_donchian(bars, asset="BBB", N=20, buf=0.0, atr_n=5)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_donchian_dn"
    assert row["confirmed_ts"] == bars.index[22]
    # Stop sits at the opposite (upper) channel boundary.
    assert row["stop_ref_price"] == 100.5


def test_intrabar_poke_no_close_break_yields_empty_frame() -> None:
    """A bar that pierces the channel intrabar but closes inside emits nothing."""
    # High pokes to 104 (above 100.5) but the close (100.2) stays inside the channel.
    bars = _flat_then((100.0, 104.0, 99.0, 100.2))

    out = detect_donchian(bars, asset="CCC", N=20, buf=0.0, atr_n=5)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_donchian(empty, asset="DDD")
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

    # Inject a few clean breakouts so the proof has events to compare. Each pushes
    # the close decisively past the trailing 10-bar extreme.
    for t, side in ((25, "up"), (40, "dn"), (52, "up")):
        if side == "up":
            close[t] = max(high[t - 10 : t]) + 3.0
            high[t] = close[t] + 0.3
            low[t] = close[t] - 0.5
        else:
            close[t] = min(low[t - 10 : t]) - 3.0
            low[t] = close[t] - 0.3
            high[t] = close[t] + 0.5

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a leak would surface as a prefix/full mismatch.
    assert_prefix_invariant(detect_donchian, bars, N=10, buf=0.0, atr_n=5, asset="EEE")
