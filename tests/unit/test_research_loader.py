"""Unit tests for fundcloud.research.loader (no live ClickHouse)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.research.loader import (
    EPOCH_FLOOR,
    _build_bars_query,
    _coerce_utc,
    _q_table,
    _to_panel,
    load_bars,
)


def test_build_bars_query_has_dedup_collapse_shape() -> None:
    sql = _build_bars_query("default.md_ohlcv_data", with_symbols=False)
    assert "argMax(open, (timestamp, updated_at))" in sql
    assert "argMax(close, (timestamp, updated_at))" in sql
    assert "toDate(timestamp) AS date" in sql
    assert "GROUP BY symbol, date" in sql
    assert "`default`.`md_ohlcv_data`" in sql
    # bound, injection-safe params for the source filters
    assert "{adapter:String}" in sql
    assert "{timeframe:String}" in sql
    assert "toDateTime64({start:String}, 3, 'UTC')" in sql
    # no symbol filter when not requested
    assert "symbol IN" not in sql


def test_build_bars_query_symbol_filter_toggle() -> None:
    sql = _build_bars_query("md_ohlcv_data", with_symbols=True)
    assert "symbol IN {symbols:Array(String)}" in sql


def test_q_table_quoting() -> None:
    assert _q_table("default.md_ohlcv_data") == "`default`.`md_ohlcv_data`"
    assert _q_table("bars") == "`bars`"


def test_coerce_utc_naive_and_aware() -> None:
    naive = _coerce_utc("2024-01-02")
    assert naive.tzinfo is not None
    assert str(naive.tz) == "UTC"
    aware = _coerce_utc(pd.Timestamp("2024-01-02 12:00", tz="US/Eastern"))
    assert str(aware.tz) == "UTC"
    assert aware.hour == 17  # 12:00 ET -> 17:00 UTC


def test_to_panel_pivots_long_to_canonical_wide() -> None:
    raw = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "BBB"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
        "open": [10.0, 11.0, 20.0, 21.0],
        "high": [10.5, 11.5, 20.5, 21.5],
        "low": [9.5, 10.5, 19.5, 20.5],
        "close": [10.2, 11.2, 20.2, 21.2],
        "volume": [100.0, 110.0, 200.0, 210.0],
    })
    panel = _to_panel(raw)

    assert isinstance(panel.columns, pd.MultiIndex)
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.tz is not None
    # one row per date, sorted
    assert list(panel.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True))
    # canonical OHLCV field ordering (open first)
    fields = list(dict.fromkeys(panel.columns.get_level_values(0)))
    assert fields == ["open", "high", "low", "close", "volume"]
    # values land in the right cells
    closes = panel.xs("close", axis=1, level=0)
    assert closes.loc[pd.Timestamp("2024-01-03", tz="UTC"), "BBB"] == 21.2


def test_to_panel_empty_returns_empty() -> None:
    assert _to_panel(pd.DataFrame()).empty


def test_load_bars_rejects_start_before_epoch_floor() -> None:
    with pytest.raises(ValueError, match="before"):
        load_bars(host="dummy", start="1999-01-01")
    # the floor itself is the 2000 boundary
    assert pd.Timestamp("2000-01-01", tz="UTC") == EPOCH_FLOOR


def test_load_bars_empty_symbols_short_circuits() -> None:
    # empty symbol list must not attempt a connection
    out = load_bars(host="dummy", symbols=[], start="2005-01-01")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_to_panel_handles_single_symbol() -> None:
    raw = pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
        "close": [1.2, 2.2], "volume": [np.nan, 5.0],
    })
    panel = _to_panel(raw)
    assert isinstance(panel.columns, pd.MultiIndex)
    assert sorted(set(panel.columns.get_level_values(1))) == ["AAA"]
