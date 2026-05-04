"""ClickHouse backend — Docker-based integration tests.

Spins up an ephemeral ``clickhouse/clickhouse-server`` container via
testcontainers (see ``conftest.py``), seeds it with deterministic synthetic
OHLCV + ML feature columns across two prefixes × two codes × two
timeframes, and exercises every public method on the backend against the
real driver. Skipped by default; run with::

    uv run pytest tests/integration -m docker -q
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.docker


# ----------------------------------------------------------------- read


def test_read_multi_asset_returns_multiindex(clickhouse_multi_asset: object) -> None:
    bars = clickhouse_multi_asset.read()  # type: ignore[attr-defined]
    assert isinstance(bars, pd.DataFrame)
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert bars.index.is_monotonic_increasing
    assert isinstance(bars.columns, pd.MultiIndex)
    fields = list(dict.fromkeys(bars.columns.get_level_values(0)))
    symbols = sorted(set(bars.columns.get_level_values(1)))
    # OHLCV first, then features
    assert fields[:5] == ["open", "high", "low", "close", "volume"]
    assert "rsi_14" in fields
    assert "sentiment" in fields
    # Composite asset key with the configured ":" separator
    assert symbols == ["HKEX:0001", "HKEX:0002", "TSE:6758", "TSE:7203"]


def test_read_with_key_filters_to_single_symbol(clickhouse_multi_asset: object) -> None:
    bars = clickhouse_multi_asset.read(key="HKEX:0001")  # type: ignore[attr-defined]
    assert isinstance(bars.columns, pd.MultiIndex)
    assert sorted(set(bars.columns.get_level_values(1))) == ["HKEX:0001"]


def test_read_window_bounds_are_inclusive(clickhouse_multi_asset: object) -> None:
    bars = clickhouse_multi_asset.read(  # type: ignore[attr-defined]
        start="2024-01-10", end="2024-01-20"
    )
    assert bars.index.min() >= pd.Timestamp("2024-01-10")
    assert bars.index.max() <= pd.Timestamp("2024-01-20")


def test_read_columns_filter_picks_canonical_field(clickhouse_multi_asset: object) -> None:
    bars = clickhouse_multi_asset.read(columns=["close"])  # type: ignore[attr-defined]
    fields = set(bars.columns.get_level_values(0))
    assert fields == {"close"}


def test_read_feature_cols_explicit_list(clickhouse_kwargs: dict[str, object]) -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(
        table="bars",
        asset_cols=["prefix", "code"],
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1d",
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        feature_cols=["rsi_14"],
        **clickhouse_kwargs,
    )
    bars = ch.read()
    fields = list(dict.fromkeys(bars.columns.get_level_values(0)))
    assert "rsi_14" in fields
    assert "sentiment" not in fields


def test_read_feature_cols_none_drops_extras(clickhouse_kwargs: dict[str, object]) -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(
        table="bars",
        asset_cols=["prefix", "code"],
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1d",
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        feature_cols=None,
        **clickhouse_kwargs,
    )
    bars = ch.read()
    fields = set(bars.columns.get_level_values(0))
    assert fields <= {"open", "high", "low", "close", "volume"}
    assert "rsi_14" not in fields


def test_timeframe_filter_distinguishes_intervals(clickhouse_kwargs: dict[str, object]) -> None:
    from fundcloud.data import ClickHouse

    daily = ClickHouse(
        table="bars",
        asset_cols=["prefix", "code"],
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1d",
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        **clickhouse_kwargs,
    )
    hourly = ClickHouse(
        table="bars",
        asset_cols=["prefix", "code"],
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1h",
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        **clickhouse_kwargs,
    )
    daily_close = daily.read(columns=["close"])[("close", "HKEX:0001")]
    hourly_close = hourly.read(columns=["close"])[("close", "HKEX:0001")]
    # Daily and hourly bars come from independent random series — they must differ
    assert not daily_close.equals(hourly_close)


def test_read_single_asset_mode_returns_flat_columns(
    clickhouse_single_asset: object,
) -> None:
    bars = clickhouse_single_asset.read()  # type: ignore[attr-defined]
    assert not isinstance(bars.columns, pd.MultiIndex)
    assert list(bars.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert "rsi_14" in bars.columns


# ----------------------------------------------------------------- discovery


def test_keys_returns_composite_asset_strings(clickhouse_multi_asset: object) -> None:
    keys = clickhouse_multi_asset.keys()  # type: ignore[attr-defined]
    assert keys == ["HKEX:0001", "HKEX:0002", "TSE:6758", "TSE:7203"]


def test_assets_reports_period_coverage(clickhouse_multi_asset: object) -> None:
    df = clickhouse_multi_asset.assets()  # type: ignore[attr-defined]
    assert list(df.columns) == ["asset", "start", "end", "n_rows"]
    assert sorted(df["asset"]) == ["HKEX:0001", "HKEX:0002", "TSE:6758", "TSE:7203"]
    assert (df["n_rows"] == 60).all()  # 60 days seeded per (prefix, code, tf=1d)
    assert df["start"].le(df["end"]).all()


def test_last_index_matches_max_seeded_timestamp(clickhouse_multi_asset: object) -> None:
    last = clickhouse_multi_asset.last_index()  # type: ignore[attr-defined]
    assert isinstance(last, pd.Timestamp)
    expected = pd.Timestamp("2024-01-02") + pd.Timedelta(days=59)
    assert last == expected


def test_last_index_per_key(clickhouse_multi_asset: object) -> None:
    last = clickhouse_multi_asset.last_index("HKEX:0001")  # type: ignore[attr-defined]
    expected = pd.Timestamp("2024-01-02") + pd.Timedelta(days=59)
    assert last == expected


def test_exists_per_key(clickhouse_multi_asset: object) -> None:
    assert clickhouse_multi_asset.exists("HKEX:0001") is True  # type: ignore[attr-defined]
    assert clickhouse_multi_asset.exists("XXXX:9999") is False  # type: ignore[attr-defined]


# ----------------------------------------------------------------- read-only


def test_write_blocked(clickhouse_multi_asset: object) -> None:
    from fundcloud.data import ReadOnlyError

    with pytest.raises(ReadOnlyError):
        clickhouse_multi_asset.write("k", pd.DataFrame())  # type: ignore[attr-defined]


def test_delete_blocked(clickhouse_multi_asset: object) -> None:
    from fundcloud.data import ReadOnlyError

    with pytest.raises(ReadOnlyError):
        clickhouse_multi_asset.delete("k")  # type: ignore[attr-defined]
