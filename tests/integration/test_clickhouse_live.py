"""ClickHouse backend — live integration tests.

Skipped by default (``@pytest.mark.network``). To run:

    set -a; source .temp.env; set +a   # or export the vars another way
    uv run pytest tests/integration/test_clickhouse_live.py -m network -q

Required env vars (read via the same fall-throughs as the constructor):

* ``CLICKHOUSE_HOST`` — required
* ``CLICKHOUSE_PORT``, ``CLICKHOUSE_USER``, ``CLICKHOUSE_PASSWORD``,
  ``CLICKHOUSE_DATABASE`` — recommended
* ``FC_TEST_CLICKHOUSE_TABLE`` — table name to read. Required when
  pointing at a real ClickHouse; the fixture-based docker tests in
  ``tests/integration/test_clickhouse_docker.py`` seed their own.
* ``FC_TEST_CLICKHOUSE_TIMESTAMP_COL`` — defaults to ``timestamp``.
* ``FC_TEST_CLICKHOUSE_ASSET_COLS`` — comma-separated list, e.g.
  ``"prefix,code"``. If unset, the table is read in single-asset mode.
* ``FC_TEST_CLICKHOUSE_TIMEFRAME_COL`` and
  ``FC_TEST_CLICKHOUSE_TIMEFRAME`` — optional pair to filter by interval.

The tests are intentionally schema-agnostic so they tolerate whatever
column layout the configured table actually has.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.network


def _require_env(*names: str) -> str:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    pytest.skip(f"none of {names!r} is set")


def _build_backend() -> object:
    pytest.importorskip("clickhouse_connect")
    from fundcloud.data import ClickHouse

    host = _require_env("CLICKHOUSE_HOST")
    table = _require_env("FC_TEST_CLICKHOUSE_TABLE")
    timestamp_col = os.environ.get("FC_TEST_CLICKHOUSE_TIMESTAMP_COL", "timestamp")
    asset_cols_raw = os.environ.get("FC_TEST_CLICKHOUSE_ASSET_COLS")
    asset_cols = (
        [c.strip() for c in asset_cols_raw.split(",") if c.strip()] if asset_cols_raw else None
    )
    timeframe_col = os.environ.get("FC_TEST_CLICKHOUSE_TIMEFRAME_COL")
    timeframe = os.environ.get("FC_TEST_CLICKHOUSE_TIMEFRAME")
    # The backend no longer reads connection env vars itself; forward
    # whatever the live runner exported so the constructor receives
    # explicit config.
    port_raw = os.environ.get("CLICKHOUSE_PORT")
    user = os.environ.get("CLICKHOUSE_USER")
    password = os.environ.get("CLICKHOUSE_PASSWORD")
    database = os.environ.get("CLICKHOUSE_DATABASE")

    return ClickHouse(
        table=table,
        host=host,
        port=int(port_raw) if port_raw else None,
        user=user,
        password=password,
        database=database,
        timestamp_col=timestamp_col,
        asset_cols=asset_cols,
        timeframe_col=timeframe_col,
        timeframe=timeframe,
    )


# --------------------------------------------------------------- read


def test_read_returns_non_empty_frame() -> None:
    """Read a small recent window; assert shape contracts."""
    ch = _build_backend()
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=14)
    bars = ch.read(start=start, end=end)

    if bars.empty:
        pytest.skip("table returned no rows for the requested window — try a wider range")

    assert isinstance(bars.index, pd.DatetimeIndex), "expected DatetimeIndex"
    assert bars.index.name is None
    assert bars.index.is_monotonic_increasing
    assert len(bars.columns) >= 1


def test_read_with_multiindex_when_asset_cols_set() -> None:
    """If running with asset_cols configured, output must be a MultiIndex frame."""
    if not os.environ.get("FC_TEST_CLICKHOUSE_ASSET_COLS"):
        pytest.skip("FC_TEST_CLICKHOUSE_ASSET_COLS not set — single-asset mode")
    ch = _build_backend()
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=14)
    bars = ch.read(start=start, end=end)
    if bars.empty:
        pytest.skip("table returned no rows for the requested window")
    assert isinstance(bars.columns, pd.MultiIndex), "asset_cols was set but output columns are flat"


# --------------------------------------------------------------- discovery


def test_assets_and_keys_consistent() -> None:
    if not os.environ.get("FC_TEST_CLICKHOUSE_ASSET_COLS"):
        pytest.skip("FC_TEST_CLICKHOUSE_ASSET_COLS not set — keys()/assets() return empty")
    ch = _build_backend()
    keys = ch.keys()
    df = ch.assets()

    assert list(df.columns) == ["asset", "start", "end", "n_rows"]
    if df.empty:
        assert keys == []
        return
    assert set(keys) == set(df["asset"]), "keys() and assets() should agree on the asset list"
    assert (df["n_rows"] > 0).all()
    assert df["start"].le(df["end"]).all()


def test_last_index_returns_timestamp_or_none() -> None:
    ch = _build_backend()
    ts = ch.last_index()
    assert ts is None or isinstance(ts, pd.Timestamp)


# --------------------------------------------------------------- read-only contract


def test_write_blocked_on_live_backend() -> None:
    from fundcloud.data import ReadOnlyError

    ch = _build_backend()
    with pytest.raises(ReadOnlyError):
        ch.write("any", pd.DataFrame())  # type: ignore[attr-defined]
