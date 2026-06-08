"""Clean daily-OHLCV extraction from the ClickHouse ``md_ohlcv_data`` feed.

Research-grade reader for the raw ``default.md_ohlcv_data`` table. That table is a
``SharedReplacingMergeTree`` partitioned by ``(toYYYYMM(timestamp), adapter, timeframe)``
which, for daily US-equity bars, carries two quirks the generic
:class:`fundcloud.data.ClickHouse` backend does **not** resolve correctly:

1. **Un-merged replacing-merge versions.** Several physical rows can share one
   ``(symbol, timeframe, timestamp, …)`` key until a background merge collapses
   them; the authoritative version is the one with the largest ``updated_at``.
2. **Two-to-three stamps per trading day.** The *same* daily bar is emitted at
   ``00:00`` / ``04:00`` / ``20:00`` UTC — identical OHLC, with volume fullest at
   ``20:00`` (= 16:00 ET close). These are one session, not intraday pieces, so
   they must be **collapsed** (keep the latest stamp), never summed.

Both are handled server-side in a single pass::

    argMax(col, (timestamp, updated_at)) GROUP BY symbol, toDate(timestamp)

The latest stamp wins (picking the finalized close bar), the latest ``updated_at``
breaks ties, and the result is exactly one clean row per ``(symbol, date)``.
``FINAL`` is deliberately avoided (it is forbidden on the production cluster).

Garbage epoch rows (``timestamp = 1970-01-02``) are excluded by the mandatory
``start`` floor (>= 2000-01-01). The reader leans on
:class:`fundcloud.data.ClickHouse` purely for connection management and returns the
canonical ``(field, symbol)`` MultiIndex frame the rest of fundcloud consumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from fundcloud.data._columns import canonicalize_ohlcv_order

if TYPE_CHECKING:  # pragma: no cover — type-check only
    from fundcloud.data.clickhouse import ClickHouse

__all__ = ["load_bars", "load_universe"]

#: Hard floor that excludes the ``1970-01-02`` epoch-garbage rows.
EPOCH_FLOOR = pd.Timestamp("2000-01-01", tz="UTC")

_OHLCV = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------- helpers


def _q_ident(name: str) -> str:
    """Backtick-quote a ClickHouse identifier, escaping embedded backticks."""
    return "`" + name.replace("`", "``") + "`"


def _q_table(name: str) -> str:
    """Quote ``db.table`` (or a bare ``table``) preserving the dot separator."""
    if "." in name:
        db, tbl = name.split(".", 1)
        return f"{_q_ident(db)}.{_q_ident(tbl)}"
    return _q_ident(name)


def _coerce_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Coerce a timestamp-like to a tz-aware UTC :class:`pandas.Timestamp`."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _utc_str(ts: pd.Timestamp) -> str:
    """UTC wall-clock as an ISO string for ``toDateTime64(…, 'UTC')`` binding.

    Binding the bounds as a string wrapped in ``toDateTime64(…, 3, 'UTC')`` keeps
    the window unambiguously UTC. A naive ``DateTime64`` param would otherwise be
    read in the ClickHouse *server* timezone, spilling the window by the server's
    UTC offset.
    """
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")


def _build_bars_query(table: str, *, with_symbols: bool) -> str:
    """Build the dedup+collapse SELECT for one time window.

    ``argMax(col, (timestamp, updated_at))`` collapses both the intraday-of-day
    duplication (latest stamp = finalized bar) and the ReplacingMergeTree
    versions (latest ``updated_at``) into one row per ``(symbol, date)``.
    """
    symbol_clause = "  AND symbol IN {symbols:Array(String)}\n" if with_symbols else ""
    cols = ",\n".join(
        f"       toFloat64(argMax({c}, (timestamp, updated_at))) AS {c}" for c in _OHLCV
    )
    return (
        "SELECT symbol,\n"
        "       toDate(timestamp) AS date,\n"
        f"{cols}\n"
        f"FROM {_q_table(table)}\n"
        "WHERE adapter = {adapter:String}\n"
        "  AND asset_type = {asset_type:String}\n"
        "  AND timeframe = {timeframe:String}\n"
        "  AND timestamp >= toDateTime64({start:String}, 3, 'UTC')\n"
        "  AND timestamp < toDateTime64({end:String}, 3, 'UTC')\n"
        f"{symbol_clause}"
        "GROUP BY symbol, date\n"
        "ORDER BY symbol, date"
    )


def _to_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long ``(symbol, date, o/h/l/c/v)`` result into a canonical panel.

    Returns a ``(field, symbol)`` MultiIndex column frame on a tz-aware UTC
    :class:`~pandas.DatetimeIndex`, OHLCV-ordered, sorted by date.
    """
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    panel = df.set_index(["date", "symbol"])[list(_OHLCV)].unstack("symbol")
    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = None
    panel = canonicalize_ohlcv_order(panel)
    return panel.sort_index()


def _open_client(
    *,
    table: str,
    host: str,
    port: int | None,
    user: str | None,
    password: str | None,
    database: str | None,
    ssl: bool,
) -> ClickHouse:
    """Open a :class:`fundcloud.data.ClickHouse` purely for its HTTP client.

    The library uses ``clickhouse-connect`` (HTTPS on 8443); pass ``port=8443``
    for ClickHouse Cloud — *not* the native ``9440`` some profiles advertise.
    """
    from fundcloud.data import ClickHouse

    return ClickHouse(
        table=table,
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        ssl=ssl,
    )


# --------------------------------------------------------------------- public API


def load_bars(
    *,
    host: str,
    start: pd.Timestamp | str = "2005-01-01",
    end: pd.Timestamp | str | None = None,
    symbols: Sequence[str] | None = None,
    adapter: str = "polygon",
    asset_type: str = "stock",
    timeframe: str = "1D",
    table: str = "default.md_ohlcv_data",
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    ssl: bool = True,
    chunk_years: int | None = None,
) -> pd.DataFrame:
    """Read clean daily OHLCV from ``md_ohlcv_data`` into a canonical panel.

    One clean bar per ``(symbol, date)`` is guaranteed by a server-side
    ``argMax(col, (timestamp, updated_at)) GROUP BY symbol, toDate(timestamp)``
    that collapses both the intraday-of-day duplication and the ReplacingMergeTree
    versions in a single pass (see the module docstring).

    Parameters
    ----------
    host
        ClickHouse host. Credentials are passed explicitly — the library never
        reads the environment.
    start, end
        Inclusive-start / exclusive-end window. ``start`` must be ``>=`` 2000-01-01
        to exclude the ``1970-01-02`` epoch-garbage rows; it defaults to
        ``"2005-01-01"``. ``end`` defaults to tomorrow (UTC), so today's bar is
        included.
    symbols
        Restrict to these symbols. ``None`` reads the full ``adapter`` /
        ``asset_type`` / ``timeframe`` slice (large — prefer batching via
        ``chunk_years`` and persisting locally).
    adapter, asset_type, timeframe
        Source filters. Defaults select polygon US-equity daily bars.
    table
        Source table; ``"default.md_ohlcv_data"`` by default.
    port, user, password, database, ssl
        Connection params forwarded to ``clickhouse-connect`` (HTTPS port 8443).
    chunk_years
        If set, pull the window in ``chunk_years``-sized slices and concatenate —
        bounds peak memory on multi-year, full-universe pulls.

    Returns
    -------
    pandas.DataFrame
        ``(field, symbol)`` MultiIndex columns on a tz-aware UTC DatetimeIndex,
        OHLCV-ordered. Empty frame when nothing matches.
    """
    start_ts = _coerce_utc(start)
    if start_ts < EPOCH_FLOOR:
        msg = (
            f"start={start!r} is before {EPOCH_FLOOR.date()}; the floor excludes the "
            "1970-01-02 epoch-garbage rows in md_ohlcv_data"
        )
        raise ValueError(msg)
    end_ts = (
        _coerce_utc(end)
        if end is not None
        else pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    )

    if symbols is not None:
        symbols = list(symbols)
        if not symbols:
            return pd.DataFrame()

    client = _open_client(
        table=table, host=host, port=port, user=user, password=password,
        database=database, ssl=ssl,
    )
    try:
        if chunk_years and chunk_years > 0:
            frames: list[pd.DataFrame] = []
            cursor = start_ts
            step = pd.DateOffset(years=chunk_years)
            while cursor < end_ts:
                nxt = min(_coerce_utc(cursor + step), end_ts)
                frames.append(
                    _query_window(client, table, cursor, nxt, symbols, adapter, asset_type, timeframe)
                )
                cursor = nxt
            frames = [f for f in frames if not f.empty]
            if not frames:
                return pd.DataFrame()
            panel = pd.concat(frames, axis=0).sort_index()
            return canonicalize_ohlcv_order(panel)
        return _query_window(
            client, table, start_ts, end_ts, symbols, adapter, asset_type, timeframe
        )
    finally:
        client.close()


def _query_window(
    client: ClickHouse,
    table: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    symbols: list[str] | None,
    adapter: str,
    asset_type: str,
    timeframe: str,
) -> pd.DataFrame:
    """Run the dedup+collapse query for a single ``[start, end)`` window."""
    sql = _build_bars_query(table, with_symbols=symbols is not None)
    params: dict[str, Any] = {
        "adapter": adapter,
        "asset_type": asset_type,
        "timeframe": timeframe,
        "start": _utc_str(start_ts),
        "end": _utc_str(end_ts),
    }
    if symbols is not None:
        params["symbols"] = symbols
    raw = client.client.query_df(sql, parameters=params)
    return _to_panel(raw)


def load_universe(
    *,
    host: str,
    min_days: int = 250,
    active_only: bool = False,
    max_staleness_days: int = 7,
    floor: pd.Timestamp | str = "2005-01-01",
    adapter: str = "polygon",
    asset_type: str = "stock",
    timeframe: str = "1D",
    table: str = "default.md_ohlcv_data",
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    ssl: bool = True,
) -> list[str]:
    """List symbols with enough daily history in ``md_ohlcv_data``.

    Parameters
    ----------
    min_days
        Keep symbols with at least this many distinct trading days since ``floor``.
    active_only
        When ``True``, also require a bar within ``max_staleness_days`` of today.
        **Default ``False`` keeps delisted names** — the source retains them, so
        the survivorship-free universe is the honest default. ``active_only=True``
        introduces survivorship bias and should only be used for live screening.
    floor
        Earliest date counted (also excludes the 1970 epoch garbage).

    Returns
    -------
    list[str]
        Symbols sorted ascending.
    """
    floor_ts = _coerce_utc(floor)
    params: dict[str, Any] = {
        "adapter": adapter,
        "asset_type": asset_type,
        "timeframe": timeframe,
        "floor": _utc_str(floor_ts),
        "min_days": int(min_days),
    }
    having = "HAVING count(DISTINCT toDate(timestamp)) >= {min_days:UInt32}"
    if active_only:
        having += " AND max(toDate(timestamp)) >= today() - {staleness:UInt32}"
        params["staleness"] = int(max_staleness_days)
    sql = (
        f"SELECT symbol FROM {_q_table(table)}\n"
        "WHERE adapter = {adapter:String}\n"
        "  AND asset_type = {asset_type:String}\n"
        "  AND timeframe = {timeframe:String}\n"
        "  AND timestamp >= toDateTime64({floor:String}, 3, 'UTC')\n"
        "GROUP BY symbol\n"
        f"{having}\n"
        "ORDER BY symbol"
    )
    client = _open_client(
        table=table, host=host, port=port, user=user, password=password,
        database=database, ssl=ssl,
    )
    try:
        res = client.client.query_df(sql, parameters=params)
    finally:
        client.close()
    if res.empty:
        return []
    return res["symbol"].astype(str).tolist()
