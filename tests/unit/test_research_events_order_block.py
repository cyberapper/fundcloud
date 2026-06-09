"""Synthetic-fixture tests for ``fundcloud.research.events._order_block``.

The prefix-invariance proof LEADS the file: an order block is the trickiest
detector to keep online-causal, because the naive form scans *forward* from the
opposing candle and dates the event there, leaking future bars. The impulse-bar-
driven loop dates the event at the impulse bar ``c`` with every read ``<= c``, so
``assert_prefix_invariant`` passing on a noisy series with injected impulses is
real evidence of causality. The remaining fixtures pin the bullish/bearish firing
geometry, two distinct no-fire branches, and the empty-input shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._order_block import detect_order_block
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


def test_prefix_invariance_no_future_leak() -> None:
    """Prove online causality: prefix runs reproduce the full run's confirmed events.

    A leak would arise from an opposing-candle-driven loop (scanning forward from
    ``j``). The impulse-bar-driven loop must keep every read ``<= c``.
    """
    rng = np.random.default_rng(11)
    n = 60
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, size=n))
    high = close + rng.uniform(0.2, 0.6, size=n)
    low = close - rng.uniform(0.2, 0.6, size=n)
    open_ = close - rng.normal(0.0, 0.1, size=n)

    # Inject clean order blocks: an opposing candle at j = c - 1, then a decisive
    # displacement at c that clears the opposing candle's extreme.
    for c, side in ((20, "bull"), (35, "bear"), (50, "bull")):
        j = c - 1
        if side == "bull":
            open_[j] = close[j] + 1.5  # bearish candle at j
            high[j] = open_[j] + 0.3
            low[j] = close[j] - 0.3
            open_[c] = close[c - 1]  # bullish impulse at c
            close[c] = open_[c] + 6.0
            high[c] = close[c] + 0.3
            low[c] = open_[c] - 0.3
        else:
            open_[j] = close[j] - 1.5  # bullish candle at j
            low[j] = open_[j] - 0.3
            high[j] = close[j] + 0.3
            open_[c] = close[c - 1]  # bearish impulse at c
            close[c] = open_[c] - 6.0
            low[c] = close[c] - 0.3
            high[c] = open_[c] + 0.3

    rows = list(zip(open_, high, low, close, strict=True))
    bars = _bars([(o, h, low_, c) for o, h, low_, c in rows])

    # Must not raise: a forward-scanning leak surfaces as a prefix/full mismatch.
    assert_prefix_invariant(
        detect_order_block, bars, m=5, r=1, z_body=1.0, atr_n=5, asset="EEE"
    )


def test_bullish_order_block_fires() -> None:
    """A demand block: the last bearish candle before a bullish displacement."""
    # Index 3 is the last bearish candle (open 101, close 99) before the impulse.
    # Index 6 is a bullish impulse: body 5.4 > atr[5] and close 105 clears high[2:4].
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.2),  # 0
            (100.2, 100.6, 99.8, 100.3),  # 1
            (100.3, 100.7, 99.9, 100.4),  # 2
            (101.0, 101.2, 98.8, 99.0),  # 3  bearish j: body [99, 101]
            (99.0, 99.6, 98.7, 99.3),  # 4  small bar
            (99.3, 99.9, 99.0, 99.6),  # 5  small bar
            (99.6, 105.5, 99.5, 105.0),  # 6  bullish impulse, close 105
        ]
    )

    out = detect_order_block(bars, asset="AAA", m=5, r=1, z_body=1.0, atr_n=5)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["asset"] == "AAA"
    assert row["event_id"] == "ev_ob_up"
    assert row["confirmed_ts"] == bars.index[6]
    assert row["formation_end_ts"] == bars.index[6]
    # Last bar -> no next bar to execute on.
    assert pd.isna(row["execution_ts"])
    assert pd.isna(row["entry_ref_price"])
    # Zone is the opposing candle's body; stop sits under its low.
    assert row["zone_lo"] == 99.0
    assert row["zone_hi"] == 101.0
    assert row["stop_ref_price"] == bars["low"].iloc[3]


def test_bearish_order_block_fires() -> None:
    """A supply block: the last bullish candle before a bearish displacement."""
    # Index 3 is the last bullish candle (open 99, close 101) before the impulse.
    # Index 6 is a bearish impulse: body 5.4 > atr[5] and close 95 clears low[2:4].
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.2),  # 0
            (100.2, 100.6, 99.8, 100.3),  # 1
            (100.3, 100.7, 99.9, 100.4),  # 2
            (99.0, 101.2, 98.8, 101.0),  # 3  bullish j: body [99, 101]
            (101.0, 101.3, 100.4, 100.7),  # 4  small bar
            (100.7, 101.0, 100.1, 100.4),  # 5  small bar
            (100.4, 100.5, 94.5, 95.0),  # 6  bearish impulse, close 95
        ]
    )

    out = detect_order_block(bars, asset="BBB", m=5, r=1, z_body=1.0, atr_n=5)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_ob_dn"
    assert row["confirmed_ts"] == bars.index[6]
    assert row["zone_lo"] == 99.0
    assert row["zone_hi"] == 101.0
    assert row["stop_ref_price"] == bars["high"].iloc[3]


def test_no_clearance_yields_empty_frame() -> None:
    """An impulse that fails to clear the opposing candle's extreme emits nothing."""
    # Index 2 carries a high of 110 inside the clearance window [j-r, j], so the
    # bullish impulse's close (105) never clears it -> no event.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.2),  # 0
            (100.2, 100.6, 99.8, 100.3),  # 1
            (100.3, 110.0, 99.9, 100.4),  # 2  tall high blocks clearance
            (101.0, 101.2, 98.8, 99.0),  # 3  bearish j
            (99.0, 99.6, 98.7, 99.3),  # 4
            (99.3, 99.9, 99.0, 99.6),  # 5
            (99.6, 105.5, 99.5, 105.0),  # 6  impulse, close 105 < 110
        ]
    )

    out = detect_order_block(bars, asset="CCC", m=5, r=1, z_body=1.0, atr_n=5)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_no_opposing_candle_yields_empty_frame() -> None:
    """A displacement with no opposing candle in the look-back window emits nothing."""
    # Every bar before the bullish impulse is itself bullish, so no bearish
    # opposing candle exists in [c - m, c - 1] -> no demand block.
    bars = _bars(
        [
            (100.0, 100.5, 99.5, 100.2),  # 0  bullish
            (100.2, 100.6, 99.8, 100.4),  # 1  bullish
            (100.4, 100.9, 100.0, 100.7),  # 2  bullish
            (100.7, 101.2, 100.3, 101.0),  # 3  bullish
            (101.0, 101.6, 100.7, 101.4),  # 4  bullish
            (101.4, 102.0, 101.1, 101.8),  # 5  bullish
            (101.8, 108.0, 101.5, 107.5),  # 6  bullish impulse, no opposing j
        ]
    )

    out = detect_order_block(bars, asset="DDD", m=5, r=1, z_body=1.0, atr_n=5)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty


def test_empty_input_yields_empty_frame() -> None:
    """Empty bars -> empty observation frame with the canonical columns."""
    empty = _bars([(1.0, 1.0, 1.0, 1.0)]).iloc[:0]
    out = detect_order_block(empty, asset="EEE")
    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert out.empty
