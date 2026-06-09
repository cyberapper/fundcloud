"""Tests for the observation schema, the params hash, and the reuse projection.

The end-to-end case is the load-bearing one: it synthesises a two-symbol
``(field, symbol)`` panel, scans it with :func:`detect_fvg`, projects the
observations with :func:`to_events_frame`, and feeds the result straight to
:func:`fundcloud.metrics.feature_quality.evaluate` — proving the reuse mapping
(``confirmed_ts``→``breakout_ts`` etc.) actually drives the existing engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.metrics.feature_quality import evaluate
from fundcloud.research.events.detectors import detect_fvg, scan_panel
from fundcloud.research.events.schema import (
    OBSERVATION_COLUMNS,
    build_observations,
    params_hash,
    to_events_frame,
)


def _row() -> dict[str, object]:
    """A single observation dict with every schema key populated."""
    ts = pd.Timestamp("2020-01-05", tz="UTC")
    return {
        "event_id": "ev_gap_imb_3c",
        "asset": "AAA",
        "timeframe": "1D",
        "formation_end_ts": ts,
        "confirmed_ts": ts,
        "execution_ts": ts + pd.Timedelta(days=1),
        "direction": "bullish",
        "params": {"body_min": 0.6},
        "logic_version": 1,
        "params_hash": "deadbeef0000",
        "entry_ref_price": 101.0,
        "stop_ref_price": float("nan"),
        "zone_lo": 100.0,
        "zone_hi": 101.0,
        "quality": float("nan"),
        "atr_at_confirm": 1.5,
    }


def test_build_observations_empty() -> None:
    out = build_observations([])

    assert out.empty
    assert list(out.columns) == list(OBSERVATION_COLUMNS)


def test_build_observations_columns_and_ts_dtypes() -> None:
    out = build_observations([_row(), _row()])

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 2
    for col in ("formation_end_ts", "confirmed_ts", "execution_ts"):
        assert isinstance(out[col].dtype, pd.DatetimeTZDtype)
        assert str(out[col].dtype) == "datetime64[ns, UTC]"


def test_params_hash_stable_and_order_independent() -> None:
    a = params_hash("ev", {"x": 1, "y": 2}, 1)
    b = params_hash("ev", {"y": 2, "x": 1}, 1)
    assert a == b

    # Stable across calls.
    assert a == params_hash("ev", {"x": 1, "y": 2}, 1)
    # Sensitive to logic_version.
    assert a != params_hash("ev", {"x": 1, "y": 2}, 2)


# --- end-to-end: scan_panel -> to_events_frame -> evaluate ------------------


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


def _asset_rows() -> np.ndarray:
    """An OHLC sequence with a bullish FVG plus enough forward bars to score."""
    rows: list[tuple[float, float, float, float]] = [
        (100.0, 100.5, 99.5, 100.0) for _ in range(3)
    ]  # ATR warmup (atr_n=3)
    rows.append((100.0, 100.5, 99.5, 100.0))  # t-2
    rows.append((100.5, 110.0, 100.0, 109.5))  # t-1: wide up-candle
    rows.append((110.0, 111.0, 101.0, 110.5))  # t: low > high[t-2] -> bullish gap
    # Forward path so execution + horizons resolve.
    rows.extend((112.0 + i, 113.0 + i, 111.0 + i, 112.5 + i) for i in range(12))
    return np.asarray(rows, dtype=float)


def _panel() -> pd.DataFrame:
    """A ``(field, symbol)`` panel for two symbols sharing one OHLC fixture."""
    arr = _asset_rows()
    idx = _index(len(arr))
    data: dict[tuple[str, str], pd.Series] = {}
    for sym in ("AAA", "BBB"):
        data[("open", sym)] = pd.Series(arr[:, 0], index=idx)
        data[("high", sym)] = pd.Series(arr[:, 1], index=idx)
        data[("low", sym)] = pd.Series(arr[:, 2], index=idx)
        data[("close", sym)] = pd.Series(arr[:, 3], index=idx)
        data[("volume", sym)] = pd.Series(np.ones(len(arr)), index=idx)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)


def test_end_to_end_scan_to_evaluate() -> None:
    panel = _panel()

    obs = scan_panel(panel, detect_fvg, atr_n=3)
    assert not obs.empty
    assert set(obs["asset"]) == {"AAA", "BBB"}
    assert list(obs.columns) == list(OBSERVATION_COLUMNS)

    events = to_events_frame(obs)
    assert list(events.columns) == [
        "asset",
        "breakout_ts",
        "long_entry",
        "short_entry",
        "stop_price",
        "quality",
        "pattern",
    ]

    # atr_window matches the detector's atr_n=3 so the engine's stop-ATR is
    # defined at the breakout (the fixture's gap fires at pos 5, inside a
    # 14-bar ATR warmup which would otherwise NaN out the stop and drop it).
    result = evaluate(events, panel, horizons=(5, 10), atr_window=3)
    assert not result.empty
    assert (result["n_events"] > 0).any()
