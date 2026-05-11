"""Tests for ``fundcloud.metrics.feature_quality.per_pattern``.

Mirrors the existing ``per_asset`` coverage but stratifies by pattern
name. Closes the loop on the TA-Lib-style bulk workflow:
``scan_all_patterns`` → ``per_pattern`` ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    Pattern,
)
from fundcloud.metrics import feature_quality as fq


def _bars(n: int = 60) -> pd.DataFrame:
    """Linearly drifting OHLCV — gives forward-return paths something to chew on."""
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    high = close + 0.5
    low = close - 0.5
    open_ = close.copy()
    volume = np.full(n, 1_000_000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    cols = pd.MultiIndex.from_product(
        [["open", "high", "low", "close", "volume"], ["TST"]],
        names=["field", "asset"],
    )
    return pd.DataFrame(
        np.column_stack([open_, high, low, close, volume]),
        index=idx,
        columns=cols,
    )


def _events(bars: pd.DataFrame, *, breakouts: list[tuple[Pattern, int]]) -> pd.DataFrame:
    """Hand-rolled events frame so we don't depend on detector internals."""
    rows: list[dict[str, object]] = []
    idx = bars.index
    close = bars[("close", "TST")].values
    for pattern, pos in breakouts:
        rows.append({
            "pattern": pattern,
            "asset": "TST",
            "formation_start": idx[max(pos - 5, 0)],
            "formation_end": idx[pos],
            "breakout_ts": idx[pos],
            "breakout_level": float(close[pos]),
            "formation_height": 5.0,
            "target_price": float("nan"),
            "stop_price": float("nan"),
            "quality": 75.0,
            "variant": None,
            "pivots": [],
            "meta": {},
        })
    return pd.DataFrame(rows, columns=list(EVENTS_COLUMNS))


def test_per_pattern_empty_events_returns_empty_frame():
    bars = _bars()
    out = fq.per_pattern(pd.DataFrame(columns=list(EVENTS_COLUMNS)), bars)
    assert out.empty
    assert "n_events" in out.columns
    assert out.index.name is None or out.index.name == "pattern"


def test_per_pattern_invalid_horizon_raises():
    bars = _bars()
    with pytest.raises(ValueError, match="horizon"):
        fq.per_pattern(pd.DataFrame(columns=list(EVENTS_COLUMNS)), bars, horizon=0)


def test_per_pattern_invalid_trade_direction_raises():
    bars = _bars()
    with pytest.raises(ValueError, match="trade_direction"):
        fq.per_pattern(
            pd.DataFrame(columns=list(EVENTS_COLUMNS)),
            bars,
            trade_direction="diagonal",
        )


def test_per_pattern_groups_rows_by_pattern_name():
    bars = _bars(n=80)
    events = _events(
        bars,
        breakouts=[
            (Pattern.DOUBLE_TOP, 20),
            (Pattern.DOUBLE_TOP, 35),
            (Pattern.HEAD_AND_SHOULDERS, 50),
        ],
    )
    out = fq.per_pattern(events, bars, horizon=10)

    assert out.index.name == "pattern"
    assert sorted(out.index.tolist()) == [
        Pattern.DOUBLE_TOP.value,
        Pattern.HEAD_AND_SHOULDERS.value,
    ]
    assert int(out.loc[Pattern.DOUBLE_TOP.value, "n_events"]) == 2
    assert int(out.loc[Pattern.HEAD_AND_SHOULDERS.value, "n_events"]) == 1


def test_per_pattern_columns_match_per_asset():
    """Schema parity with the sister stratifier — easier downstream code."""
    bars = _bars()
    events = _events(
        bars,
        breakouts=[(Pattern.DOUBLE_TOP, 20)],
    )
    pat = fq.per_pattern(events, bars, horizon=10)
    asset = fq.per_asset(events, bars, horizon=10)
    assert list(pat.columns) == list(asset.columns)
