"""ClickHouse backend tests — exercised against a fake ``clickhouse_connect``.

The driver is monkey-patched at the module level so no socket is opened.
The fake client records every ``query`` / ``query_df`` invocation so the
tests can assert on the exact SQL emitted, including identifier quoting,
parameter names, and clause ordering. The frame-shape assertions
(MultiIndex pivot, OHLCV canonicalisation, feature passthrough,
``columns=`` filter) run against handcrafted long-format frames returned
from the fake.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------- fakes


@dataclass
class _QueryResult:
    """Shape-compatible stand-in for ``clickhouse_connect.driver.query.QueryResult``."""

    result_rows: list[tuple[Any, ...]]


@dataclass
class _FakeClient:
    """Records every call; returns canned frames keyed by SQL prefix.

    Tests configure ``df_response`` (for ``query_df``) and ``rows_response``
    (for ``query``) to whatever the SQL under test should resolve to.
    """

    df_response: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows_response: list[tuple[Any, ...]] = field(default_factory=list)
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def query_df(self, sql: str, parameters: Mapping[str, Any] | None = None) -> pd.DataFrame:
        self.calls.append(("query_df", sql, dict(parameters or {})))
        return self.df_response.copy()

    def query(self, sql: str, parameters: Mapping[str, Any] | None = None) -> _QueryResult:
        self.calls.append(("query", sql, dict(parameters or {})))
        return _QueryResult(result_rows=list(self.rows_response))

    def close(self) -> None:
        self.calls.append(("close", "", {}))


@pytest.fixture
def fake_ch(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    """Patch ``_require_clickhouse_connect`` to return a fake module."""
    fake = _FakeClient()

    class _FakeModule:
        @staticmethod
        def get_client(**kwargs: Any) -> _FakeClient:
            fake.last_kwargs = kwargs  # type: ignore[attr-defined]
            return fake

    monkeypatch.setattr(
        "fundcloud.data.clickhouse._require_clickhouse_connect",
        lambda: _FakeModule,
    )
    return fake


# --------------------------------------------------------------------- construction


def test_construction_records_explicit_params() -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(
        table="t",
        host="h.example",
        port=8443,
        user="viewer",
        password="secret",
        database="default",
    )
    assert ch.host == "h.example"
    assert ch.port == 8443
    assert ch.user == "viewer"
    assert ch.password == "secret"
    assert ch.database == "default"
    assert ch.read_only is True


def test_constructor_ignores_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting CLICKHOUSE_* env vars must not influence the constructor —
    callers must pass connection params explicitly so the credential
    source is always traceable from the call site."""
    from fundcloud.data import ClickHouse

    monkeypatch.setenv("CLICKHOUSE_HOST", "from-env")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "ignored")
    ch = ClickHouse(host="from-arg", table="t")
    assert ch.host == "from-arg"
    assert ch.password == ""


def test_missing_host_raises() -> None:
    from fundcloud.data import ClickHouse

    with pytest.raises(ValueError, match="requires a host"):
        ClickHouse(table="t")


def test_missing_table_raises() -> None:
    from fundcloud.data import ClickHouse

    with pytest.raises(ValueError, match="non-empty `table`"):
        ClickHouse(host="x", table="")


def test_orphan_timeframe_filter_raises() -> None:
    from fundcloud.data import ClickHouse

    with pytest.raises(ValueError, match="timeframe_col is None"):
        ClickHouse(host="x", table="t", timeframe="1h")


def test_orphan_timeframe_col_raises() -> None:
    """Symmetric guard: a ``timeframe_col`` without a ``timeframe`` filter
    silently collapses multi-resolution rows during ``_postprocess`` —
    the dedup step drops the timeframe column then dedups on
    ``(timestamp, asset_cols)``, merging different resolutions into one."""
    from fundcloud.data import ClickHouse

    with pytest.raises(ValueError, match="timeframe is None"):
        ClickHouse(host="x", table="t", timeframe_col="interval")


def test_bad_ohlcv_map_key_raises() -> None:
    from fundcloud.data import ClickHouse

    with pytest.raises(ValueError, match="canonical OHLCV"):
        ClickHouse(host="x", table="t", ohlcv_map={"foo": "bar"})


def test_empty_asset_cols_raises() -> None:
    from fundcloud.data import ClickHouse

    with pytest.raises(ValueError, match="asset_cols"):
        ClickHouse(host="x", table="t", asset_cols=[])


def test_lazy_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """``read()`` raises a helpful ImportError when the extra is missing."""
    from fundcloud.data import ClickHouse
    from fundcloud.data import clickhouse as ch_mod

    def _broken_loader() -> Any:
        raise ImportError(
            "clickhouse-connect is required for ClickHouse. "
            "Install with: uv add 'fundcloud[data-clickhouse]' or 'fundcloud[data]'."
        )

    monkeypatch.setattr(ch_mod, "_require_clickhouse_connect", _broken_loader)

    ch = ClickHouse(host="x", table="t")
    with pytest.raises(ImportError, match="data-clickhouse"):
        ch.read()


# --------------------------------------------------------------------- read (single-asset)


def test_read_single_asset_passes_through_features(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [100.0, 101.0],
        "close": [101.0, 102.0],
        "rsi_14": [55.0, 60.0],
        "sentiment": [0.1, 0.2],
    })

    ch = ClickHouse(host="x", table="t")
    out = ch.read(start="2024-01-01", end="2024-12-31")

    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name is None
    assert list(out.columns)[:2] == ["open", "close"]  # OHLCV first
    assert "rsi_14" in out.columns and "sentiment" in out.columns

    # SQL emitted exactly once, with parameterised time bounds
    assert len(fake_ch.calls) == 1
    op, sql, params = fake_ch.calls[0]
    assert op == "query_df"
    assert "SELECT * FROM `t`" in sql
    assert "`timestamp` >= {start:DateTime64}" in sql
    assert "`timestamp` <= {end:DateTime64}" in sql
    assert "ORDER BY `timestamp`" in sql
    assert isinstance(params["start"], pd.Timestamp | __import__("datetime").datetime)
    assert isinstance(params["end"], pd.Timestamp | __import__("datetime").datetime)


def test_read_no_default_start_is_applied(fake_ch: _FakeClient) -> None:
    """ClickHouse is storage-like; reads without bounds return everything.

    Matches the behaviour of DuckDB / Parquet so a Catalog cache-read with
    no watermark surfaces all rows in the table, rather than silently
    truncating to the last year (which would hide older history).
    """
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
    })

    ch = ClickHouse(host="x", table="t")
    ch.read()

    _, sql, params = fake_ch.calls[0]
    assert "start" not in params
    assert "end" not in params
    assert ">=" not in sql
    assert "<=" not in sql


def test_read_columns_filter_post_fetch(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "open": [100.0],
        "close": [101.0],
        "rsi_14": [55.0],
    })

    ch = ClickHouse(host="x", table="t")
    out = ch.read(columns=["close", "rsi_14"])
    assert list(out.columns) == ["close", "rsi_14"]


def test_read_with_ohlcv_remap(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "o": [100.0, 101.0],
        "c": [101.0, 102.0],
        "v": [1000.0, 1100.0],
    })

    ch = ClickHouse(
        host="x",
        table="t",
        timestamp_col="ts",
        ohlcv_map={"open": "o", "close": "c", "volume": "v"},
    )
    out = ch.read()
    assert list(out.columns) == ["open", "close", "volume"]


def test_read_feature_cols_none_drops_extras(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
        "rsi_14": [55.0],
    })

    ch = ClickHouse(host="x", table="t", feature_cols=None)
    out = ch.read()
    assert list(out.columns) == ["close"]


def test_read_feature_cols_explicit_list(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
        "rsi_14": [55.0],
        "sentiment": [0.1],
        "ignored": [9.0],
    })

    ch = ClickHouse(host="x", table="t", feature_cols=["rsi_14", "sentiment"])
    out = ch.read()
    assert list(out.columns) == ["close", "rsi_14", "sentiment"]


def test_read_empty_response(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame()
    ch = ClickHouse(host="x", table="t")
    out = ch.read()
    assert out.empty


def test_single_asset_with_key_raises(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(host="x", table="t")
    with pytest.raises(KeyError, match="single-asset mode"):
        ch.read(key="something")


# --------------------------------------------------------------------- read (multi-asset)


def test_read_multi_asset_returns_multiindex(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "ts": pd.to_datetime([
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ]),
        "prefix": ["HKEX", "TSE", "HKEX", "TSE"],
        "code": ["0001", "7203", "0001", "7203"],
        "open": [100.0, 200.0, 101.0, 201.0],
        "close": [101.0, 201.0, 102.0, 202.0],
        "rsi_14": [55.0, 60.0, 56.0, 61.0],
    })

    ch = ClickHouse(
        host="x",
        table="t",
        timestamp_col="ts",
        asset_cols=["prefix", "code"],
        asset_separator=":",
    )
    out = ch.read()

    assert isinstance(out.columns, pd.MultiIndex)
    assert isinstance(out.index, pd.DatetimeIndex)

    # Symbols come from joining prefix:code
    symbols = sorted(set(out.columns.get_level_values(1)))
    assert symbols == ["HKEX:0001", "TSE:7203"]

    # Fields are OHLCV-first, extras after — closes for both symbols present
    fields = list(dict.fromkeys(out.columns.get_level_values(0)))
    assert fields[:2] == ["open", "close"]
    assert "rsi_14" in fields

    # Values land in the right slot
    assert out[("close", "HKEX:0001")].iloc[0] == pytest.approx(101.0)
    assert out[("close", "TSE:7203")].iloc[0] == pytest.approx(201.0)


def test_read_dedup_is_order_independent_when_duplicates_have_distinct_values(
    fake_ch: _FakeClient,
) -> None:
    """ClickHouse-backed sources (MV / ReplacingMergeTree) routinely emit
    several rows per logical ``(timestamp, asset)`` key while merges
    catch up. The ``read()`` pipeline must yield the same row regardless
    of the physical order in which the duplicates arrived — otherwise
    repeated reads of the same table can return different ``close`` /
    ``volume`` values, which silently corrupts downstream backtests.
    """
    from fundcloud.data import ClickHouse

    rows_natural = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "prefix": ["HKEX", "HKEX"],
        "code": ["0001", "0001"],
        "open": [100.0, 100.0],
        "close": [105.0, 100.0],
    })
    rows_reversed = rows_natural.iloc[::-1].reset_index(drop=True)

    ch = ClickHouse(
        host="x",
        table="t",
        timestamp_col="ts",
        asset_cols=["prefix", "code"],
        asset_separator=":",
    )
    fake_ch.df_response = rows_natural
    out_a = ch.read()
    fake_ch.df_response = rows_reversed
    out_b = ch.read()

    assert out_a.equals(out_b), (
        f"dedup is order-dependent: natural→{out_a[('close', 'HKEX:0001')].iloc[0]} "
        f"vs reversed→{out_b[('close', 'HKEX:0001')].iloc[0]}"
    )


def test_read_multi_asset_with_key_filters_to_single_symbol(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "prefix": ["HKEX"],
        "code": ["0001"],
        "open": [100.0],
        "close": [101.0],
    })

    ch = ClickHouse(host="x", table="t", asset_cols=["prefix", "code"])
    out = ch.read(key="HKEX:0001")

    # Pivot still gives a MultiIndex with a single symbol
    assert isinstance(out.columns, pd.MultiIndex)
    assert sorted(set(out.columns.get_level_values(1))) == ["HKEX:0001"]

    # SQL adds two WHERE clauses for the composite key
    _, sql, params = fake_ch.calls[0]
    assert "`prefix` = {asset_0:String}" in sql
    assert "`code` = {asset_1:String}" in sql
    assert params["asset_0"] == "HKEX"
    assert params["asset_1"] == "0001"


def test_read_multi_asset_bad_key_shape_raises(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(host="x", table="t", asset_cols=["prefix", "code"])
    with pytest.raises(KeyError, match="splits into"):
        ch.read(key="HKEX")  # only one part for two-column composite


# --------------------------------------------------------------------- timeframe


def test_timeframe_filter_emits_where_clause(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
    })

    ch = ClickHouse(
        host="x",
        table="t",
        timeframe_col="resolution",
        timeframe="1h",
    )
    ch.read()
    _, sql, params = fake_ch.calls[0]
    assert "`resolution` = {timeframe:String}" in sql
    assert params["timeframe"] == "1h"


def test_timeframe_col_dropped_from_output(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "resolution": ["1h"],
        "close": [100.0],
    })

    ch = ClickHouse(host="x", table="t", timeframe_col="resolution", timeframe="1h")
    out = ch.read()
    assert "resolution" not in out.columns


# --------------------------------------------------------------------- where escape hatch


def test_user_where_clause_appended(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
    })

    ch = ClickHouse(host="x", table="t", where="source = 'whale'")
    ch.read()
    _, sql, _ = fake_ch.calls[0]
    assert "(source = 'whale')" in sql


# --------------------------------------------------------------------- keys / assets


def test_keys_single_asset_returns_empty(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(host="x", table="t")
    assert ch.keys() == []
    # keys() in single-asset mode does NOT hit the network
    assert fake_ch.calls == []


def test_keys_multi_asset_joined_with_separator(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "prefix": ["HKEX", "TSE"],
        "code": ["0001", "7203"],
    })
    ch = ClickHouse(host="x", table="t", asset_cols=["prefix", "code"])
    assert ch.keys() == ["HKEX:0001", "TSE:7203"]

    _, sql, _ = fake_ch.calls[0]
    assert sql.startswith("SELECT DISTINCT `prefix`, `code` FROM `t`")


def test_assets_returns_period_coverage(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "prefix": ["HKEX", "TSE"],
        "code": ["0001", "7203"],
        "start": pd.to_datetime(["2023-01-02", "2023-02-15"]),
        "end": pd.to_datetime(["2024-12-30", "2024-12-30"]),
        "n_rows": [1000, 800],
    })
    ch = ClickHouse(host="x", table="t", asset_cols=["prefix", "code"])
    out = ch.assets()

    assert list(out.columns) == ["asset", "start", "end", "n_rows"]
    assert list(out["asset"]) == ["HKEX:0001", "TSE:7203"]
    assert list(out["n_rows"]) == [1000, 800]

    _, sql, _ = fake_ch.calls[0]
    assert "min(`timestamp`)" in sql
    assert "max(`timestamp`)" in sql
    assert "count(*)" in sql
    assert "GROUP BY `prefix`, `code`" in sql


def test_assets_single_asset_returns_empty(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    ch = ClickHouse(host="x", table="t")
    out = ch.assets()
    assert list(out.columns) == ["asset", "start", "end", "n_rows"]
    assert out.empty


def test_last_index_uses_max_timestamp_query(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.rows_response = [(np.datetime64("2024-12-30T00:00:00"),)]
    ch = ClickHouse(host="x", table="t")
    ts = ch.last_index()
    assert ts == pd.Timestamp("2024-12-30")

    _, sql, _ = fake_ch.calls[0]
    assert "max(`timestamp`)" in sql


def test_exists_runs_limit_one_query(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.rows_response = [(1,)]
    ch = ClickHouse(host="x", table="t", asset_cols=["symbol"])
    assert ch.exists("BTC-USD") is True

    _, sql, params = fake_ch.calls[0]
    assert "LIMIT 1" in sql
    assert "`symbol` = {asset_0:String}" in sql
    assert params["asset_0"] == "BTC-USD"


def test_exists_returns_false_for_arbitrary_key_in_single_asset_mode(
    fake_ch: _FakeClient,
) -> None:
    """Single-asset backends have no concept of a per-key existence check;
    ``read(key=...)`` raises and ``keys()`` returns ``[]``, but
    ``exists("anything")`` used to return ``True`` whenever the table had
    rows. That made arbitrary keys look valid.
    """
    fake_ch.rows_response = [(1,)]
    from fundcloud.data import ClickHouse

    ch = ClickHouse(host="x", table="t")  # asset_cols=None → single-asset
    assert ch.exists("any-spurious-key") is False


# --------------------------------------------------------------------- read-only


def test_write_raises_read_only_error() -> None:
    from fundcloud.data import ClickHouse, ReadOnlyError

    ch = ClickHouse(host="x", table="t")
    with pytest.raises(ReadOnlyError):
        ch.write("k", pd.DataFrame())


def test_delete_raises_read_only_error() -> None:
    from fundcloud.data import ClickHouse, ReadOnlyError

    ch = ClickHouse(host="x", table="t")
    with pytest.raises(ReadOnlyError):
        ch.delete("k")


# --------------------------------------------------------------------- dedup


def test_read_dedups_duplicate_keys(fake_ch: _FakeClient) -> None:
    """Materialised views can emit several rows per (timestamp, asset);
    the backend must dedup before the pivot or pandas raises."""
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-02",
            "2024-01-02",  # duplicate of row 0
            "2024-01-03",
        ]),
        "symbol": ["BTC", "BTC", "BTC"],
        "close": [100.0, 101.0, 102.0],
    })

    ch = ClickHouse(host="x", table="t", asset_cols=["symbol"])
    out = ch.read()
    assert len(out) == 2  # 3 input rows -> 2 unique timestamps
    # keep="last" retains the second of the two duplicates
    assert out[("close", "BTC")].iloc[0] == pytest.approx(101.0)


# --------------------------------------------------------------------- table quoting


def test_db_qualified_table_quoted_correctly(fake_ch: _FakeClient) -> None:
    from fundcloud.data import ClickHouse

    fake_ch.df_response = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02"]),
        "close": [100.0],
    })
    ch = ClickHouse(host="x", table="default.bars_1m")
    ch.read()
    _, sql, _ = fake_ch.calls[0]
    assert "FROM `default`.`bars_1m`" in sql
