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
from fundcloud.research.events.explore import forward_paths
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
        "event_id": "ev_gap_up",
        "asset": "AAA",
        "timeframe": "1D",
        "formation_end_ts": ts,
        "confirmed_ts": ts,
        "execution_ts": ts + pd.Timedelta(days=1),
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


def test_to_events_frame_routes_by_event_id_suffix() -> None:
    # _up suffix -> long_entry populated, short_entry NaN.
    up = build_observations([_row()])  # event_id "ev_gap_up"
    events_up = to_events_frame(up)
    assert events_up["long_entry"].iloc[0] == 101.0
    assert pd.isna(events_up["short_entry"].iloc[0])

    # _dn suffix -> short_entry populated, long_entry NaN.
    dn_row = _row()
    dn_row["event_id"] = "ev_gap_dn"
    dn = build_observations([dn_row])
    events_dn = to_events_frame(dn)
    assert events_dn["short_entry"].iloc[0] == 101.0
    assert pd.isna(events_dn["long_entry"].iloc[0])


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


def _panel_late_listed(pad: int = 5) -> pd.DataFrame:
    """Panel where ``BBB`` lists ``pad`` bars after the panel start (leading NaN).

    ``AAA`` is valid across the whole index; ``BBB`` is NaN for the first ``pad``
    bars then carries the FVG fixture. Mirrors a real late-IPO symbol in a panel
    that starts earlier.
    """
    arr = _asset_rows()
    n = len(arr)
    idx = _index(pad + n)
    flat = np.array([(100.0, 100.5, 99.5, 100.0)] * (pad + n), dtype=float)
    bbb = np.full((pad + n, 4), np.nan)
    bbb[pad:] = arr
    data: dict[tuple[str, str], pd.Series] = {}
    for sym, rows in (("AAA", flat), ("BBB", bbb)):
        data[("open", sym)] = pd.Series(rows[:, 0], index=idx)
        data[("high", sym)] = pd.Series(rows[:, 1], index=idx)
        data[("low", sym)] = pd.Series(rows[:, 2], index=idx)
        data[("close", sym)] = pd.Series(rows[:, 3], index=idx)
        data[("volume", sym)] = pd.Series(np.ones(pad + n), index=idx)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)


def test_late_listed_symbol_is_not_silently_dropped() -> None:
    # Regression: a symbol that lists after the panel start has leading-NaN bars.
    # Feeding those to the detector poisons Wilder's ATR seed (all-NaN ATR), so
    # every event's atr_at_confirm came back NaN and forward_paths dropped the
    # whole symbol. scan_panel must dropna per symbol before detecting.
    panel = _panel_late_listed(pad=5)

    obs = scan_panel(panel, detect_fvg, atr_n=3)
    bbb = obs[obs["asset"] == "BBB"]

    assert not bbb.empty  # the late-listed symbol produces events
    assert bbb["atr_at_confirm"].notna().all()  # ATR is real, not NaN-poisoned
    # and the events survive into the forward-path layer (not silently dropped).
    paths = forward_paths(obs, panel, horizons=(5,))
    assert (paths["asset"] == "BBB").any()
