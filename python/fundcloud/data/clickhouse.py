"""ClickHouse backend

Reads OHLCV + arbitrary feature columns from a user-shaped Clickhouse table
into the canonical ``(field, symbol)`` MultiIndex DataFrame the rest of
fundcloud expects. Designed for tables where many symbols (and optionally
many timeframes) coexist in a single wide table — including the typical
materialized-view layout for tick / aggregated bar data.

Driver: ``clickhouse-connect`` (HTTPS, port 8443/8123 — works with
ClickHouse Cloud out of the box). Lazy-imported so installs without the
extra still load the rest of ``fundcloud.data``.

Asset identification supports composite keys: pass ``asset_cols=["prefix",
"code"]`` and the values are joined with ``asset_separator`` (default
``":"``) into the ``symbol`` level of the MultiIndex. With
``asset_cols=None`` the WHERE-filtered slice is treated as one anonymous
asset and the result has flat columns.

OHLCV column names are configurable via ``ohlcv_map``; missing OHLCV
columns are simply absent from the output (no error). Every other column
in the table flows through as an extra feature column (``feature_cols="*"``)
unless restricted to a list, or suppressed with ``feature_cols=None``.

The ``where`` argument is a raw SQL fragment ANDed onto every query — the
escape hatch for filters that don't fit the structured options. The user
is responsible for escaping values inside that fragment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pandas as pd

from fundcloud.data._base import BaseBackend
from fundcloud.data._columns import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_ohlcv_columns,
)

if TYPE_CHECKING:  # pragma: no cover — type-check only
    from clickhouse_connect.driver.client import Client

__all__ = ["ClickHouse"]


_FeatureCols = Sequence[str] | Literal["*"] | None


class ClickHouse(BaseBackend):
    """Read OHLCV + feature columns from a Clickhouse table.

    Parameters
    ----------
    table
        Required. Source table name. Use ``"db.table"`` to address a
        non-default database, or pass ``database=`` and a bare table name.
    host, port, user, password, database
        Connection params. ``host`` is required — pass it explicitly
        (read it from your environment / secrets store). The other
        four are optional; ``port`` defaults to clickhouse-connect's
        choice (8443 for HTTPS), ``user`` / ``database`` default to
        ``"default"`` server-side, ``password`` defaults to empty. The
        backend never reads env vars on its own.
    ssl
        Default ``True``. Maps to clickhouse-connect's ``secure=True``,
        which selects HTTPS (port 8443 by default on ClickHouse Cloud).
    asset_cols
        Columns that together identify one asset. ``["symbol"]`` for
        single-column tables; ``["prefix", "code"]`` for HK / JP markets
        where the prefix discriminates the exchange. Default ``None`` —
        the WHERE-filtered slice is treated as one anonymous asset.
    asset_separator
        How to join multi-column asset values into a single symbol
        string for the ``(field, symbol)`` MultiIndex. Default ``":"``.
    timestamp_col
        Default ``"timestamp"``. Column used for the time index and
        ``start``/``end`` filtering.
    timeframe_col, timeframe
        Optional. Pair them to filter rows where ``timeframe_col`` equals
        ``timeframe`` (e.g. keep only ``"1h"`` bars from a table that
        stores every interval). Both default to ``None`` — no filter,
        and the column is not even read.
    ohlcv_map
        Override the source columns that map to canonical OHLCV. Defaults
        to the identity mapping (``open``, ``high``, ``low``, ``close``,
        ``volume`` map to themselves). Override only what differs:
        ``ohlcv_map={"close": "c", "volume": "vol"}`` keeps the others.
    feature_cols
        ``"*"`` (default) — every column not consumed by asset / timestamp
        / timeframe / OHLCV mapping passes through as an extra feature
        column. A list — only those columns. ``None`` — no feature
        columns at all.
    where
        Raw SQL fragment ANDed onto every query (e.g. ``"source =
        'whale'"``). User is responsible for escaping values; this is the
        escape hatch for filters that don't fit the structured options.
    connect_timeout, query_timeout
        Forwarded to ``clickhouse_connect.get_client``.
    """

    name: ClassVar[str] = "clickhouse"

    def __init__(
        self,
        *,
        table: str,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        ssl: bool = True,
        asset_cols: Sequence[str] | None = None,
        asset_separator: str = ":",
        timestamp_col: str = "timestamp",
        timeframe_col: str | None = None,
        timeframe: str | None = None,
        ohlcv_map: Mapping[str, str] | None = None,
        feature_cols: _FeatureCols = "*",
        where: str | None = None,
        connect_timeout: float = 10.0,
        query_timeout: float = 60.0,
    ) -> None:
        if not table:
            raise ValueError("ClickHouse requires a non-empty `table`")

        if not host:
            msg = (
                "ClickHouse requires a host. Pass `host=` explicitly "
                "(read it from your environment or secrets store)."
            )
            raise ValueError(msg)

        self.host = host
        self.port = port
        self.user = user
        self.password = password if password is not None else ""
        self.database = database
        self.ssl = ssl
        self.table = table

        self.asset_cols: tuple[str, ...] | None = (
            tuple(asset_cols) if asset_cols is not None else None
        )
        if self.asset_cols is not None and not self.asset_cols:
            raise ValueError("asset_cols must contain at least one column or be None")
        self.asset_separator = asset_separator
        self.timestamp_col = timestamp_col
        self.timeframe_col = timeframe_col
        self.timeframe = timeframe
        if timeframe is not None and timeframe_col is None:
            raise ValueError(
                "timeframe filter is set but timeframe_col is None — "
                "pass timeframe_col= naming the column to filter on"
            )

        self.ohlcv_map: dict[str, str] = {c: c for c in OHLCV_COLUMNS}
        if ohlcv_map:
            for canonical, source in ohlcv_map.items():
                if canonical not in OHLCV_COLUMNS:
                    msg = (
                        f"ohlcv_map keys must be canonical OHLCV names "
                        f"({list(OHLCV_COLUMNS)!r}); got {canonical!r}"
                    )
                    raise ValueError(msg)
                self.ohlcv_map[canonical] = source

        self.feature_cols: _FeatureCols = (
            list(feature_cols)
            if isinstance(feature_cols, Sequence) and not isinstance(feature_cols, str)
            else feature_cols
        )
        self.where = where
        self.connect_timeout = connect_timeout
        self.query_timeout = query_timeout

        self.read_only = True
        self._client: Client | None = None

    # --------------------------------------------------------------- connection

    @property
    def client(self) -> Client:
        if self._client is None:
            ch = _require_clickhouse_connect()
            kwargs: dict[str, Any] = {
                "host": self.host,
                "username": self.user,
                "password": self.password,
                "secure": self.ssl,
                "connect_timeout": self.connect_timeout,
                "send_receive_timeout": self.query_timeout,
            }
            if self.port is not None:
                kwargs["port"] = self.port
            if self.database is not None:
                kwargs["database"] = self.database
            self._client = ch.get_client(**kwargs)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # --------------------------------------------------------------------- read

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if key is not None and self.asset_cols is None:
            msg = (
                f"ClickHouse is configured in single-asset mode "
                f"(asset_cols=None); cannot read key={key!r}"
            )
            raise KeyError(msg)

        # No default-start: ClickHouse is a storage-like backend (matches
        # DuckDB / Parquet semantics — return whatever's in the table when
        # bounds aren't given). Callers wanting a window pass ``start`` /
        # ``end``; the ``Catalog`` does this automatically via watermarks.
        sql, params = self._build_select_sql(key=key, start=start, end=end)
        df = self.client.query_df(sql, parameters=params)
        if df.empty:
            return pd.DataFrame()
        return self._postprocess(df, columns=columns)

    def keys(self) -> list[str]:
        if self.asset_cols is None:
            return []
        sql, params = self._build_keys_sql()
        df = self.client.query_df(sql, parameters=params)
        if df.empty:
            return []
        cols = list(self.asset_cols)
        return sorted(
            self.asset_separator.join(str(v) for v in row)
            for row in df[cols].itertuples(index=False, name=None)
        )

    def assets(self) -> pd.DataFrame:
        cols = ["asset", "start", "end", "n_rows"]
        if self.asset_cols is None:
            return pd.DataFrame(columns=cols)
        sql, params = self._build_assets_sql()
        df = self.client.query_df(sql, parameters=params)
        if df.empty:
            return pd.DataFrame(columns=cols)
        ac = list(self.asset_cols)
        df["asset"] = df[ac].astype(str).agg(self.asset_separator.join, axis=1)
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
        df["n_rows"] = df["n_rows"].astype("int64")
        return df[cols].sort_values("asset").reset_index(drop=True)

    def exists(self, key: str) -> bool:
        if self.asset_cols is None:
            sql, params = self._build_select_sql(
                key=None, start=None, end=None, projection="1", limit=1
            )
        else:
            sql, params = self._build_select_sql(
                key=key, start=None, end=None, projection="1", limit=1
            )
        try:
            result = self.client.query(sql, parameters=params)
        except KeyError:
            return False
        return bool(result.result_rows)

    def last_index(self, key: str | None = None) -> pd.Timestamp | None:
        if key is not None and self.asset_cols is None:
            return None
        sql, params = self._build_select_sql(
            key=key,
            start=None,
            end=None,
            projection=f"max({_q(self.timestamp_col)}) AS last_ts",
        )
        try:
            result = self.client.query(sql, parameters=params)
        except KeyError:
            return None
        rows = result.result_rows
        if not rows or rows[0][0] is None:
            return None
        return pd.Timestamp(rows[0][0])

    # ----------------------------------------------------------------- internals

    def _build_select_sql(
        self,
        *,
        key: str | None,
        start: pd.Timestamp | str | None,
        end: pd.Timestamp | str | None,
        projection: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        select_clause = projection if projection is not None else "*"
        where_clauses, params = self._build_where(key=key, start=start, end=end)
        sql = f"SELECT {select_clause} FROM {_q_table(self.table)}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        if projection is None:
            sql += f" ORDER BY {_q(self.timestamp_col)}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return sql, params

    def _build_keys_sql(self) -> tuple[str, dict[str, Any]]:
        assert self.asset_cols is not None
        cols_sql = ", ".join(_q(c) for c in self.asset_cols)
        where_clauses, params = self._build_where(key=None, start=None, end=None)
        sql = f"SELECT DISTINCT {cols_sql} FROM {_q_table(self.table)}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += f" ORDER BY {cols_sql}"
        return sql, params

    def _build_assets_sql(self) -> tuple[str, dict[str, Any]]:
        assert self.asset_cols is not None
        cols_sql = ", ".join(_q(c) for c in self.asset_cols)
        ts = _q(self.timestamp_col)
        where_clauses, params = self._build_where(key=None, start=None, end=None)
        sql = (
            f"SELECT {cols_sql}, "
            f"min({ts}) AS start, max({ts}) AS end, count(*) AS n_rows "
            f"FROM {_q_table(self.table)}"
        )
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += f" GROUP BY {cols_sql} ORDER BY {cols_sql}"
        return sql, params

    def _build_where(
        self,
        *,
        key: str | None,
        start: pd.Timestamp | str | None,
        end: pd.Timestamp | str | None,
    ) -> tuple[list[str], dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if start is not None:
            clauses.append(f"{_q(self.timestamp_col)} >= {{start:DateTime64}}")
            params["start"] = pd.Timestamp(start).to_pydatetime()
        if end is not None:
            clauses.append(f"{_q(self.timestamp_col)} <= {{end:DateTime64}}")
            params["end"] = pd.Timestamp(end).to_pydatetime()

        if self.timeframe_col is not None and self.timeframe is not None:
            clauses.append(f"{_q(self.timeframe_col)} = {{timeframe:String}}")
            params["timeframe"] = self.timeframe

        if key is not None:
            assert self.asset_cols is not None  # guarded by caller
            parts = key.split(self.asset_separator)
            if len(parts) != len(self.asset_cols):
                msg = (
                    f"key {key!r} splits into {len(parts)} parts but "
                    f"asset_cols has {len(self.asset_cols)}"
                )
                raise KeyError(msg)
            for i, (col, val) in enumerate(zip(self.asset_cols, parts, strict=True)):
                pname = f"asset_{i}"
                clauses.append(f"{_q(col)} = {{{pname}:String}}")
                params[pname] = val

        if self.where:
            clauses.append(f"({self.where})")

        return clauses, params

    def _postprocess(self, df: pd.DataFrame, *, columns: Sequence[str] | None) -> pd.DataFrame:
        df = df.copy()

        # 1. coerce timestamp column to datetime
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # 2. drop the timeframe column from output (filter applied at SQL)
        if self.timeframe_col and self.timeframe_col in df.columns:
            df = df.drop(columns=[self.timeframe_col])

        # 3. rename source OHLCV column names to canonical
        renames = {
            src: canonical
            for canonical, src in self.ohlcv_map.items()
            if src != canonical and src in df.columns
        }
        if renames:
            df = df.rename(columns=renames)

        # 4. determine which non-asset / non-timestamp columns to keep
        consumed = {self.timestamp_col, *(self.asset_cols or ())}
        canonical_present = [c for c in OHLCV_COLUMNS if c in df.columns]
        if self.feature_cols == "*":
            keep_features = [
                c for c in df.columns if c not in consumed and c not in canonical_present
            ]
        elif self.feature_cols is None:
            keep_features = []
        else:
            # feature_cols is a Sequence[str] here — the str / None branches
            # were handled above. mypy needs ``list(...)`` to narrow the type.
            keep_features = [c for c in list(self.feature_cols) if c in df.columns]
        keep = [self.timestamp_col, *(self.asset_cols or ()), *canonical_present, *keep_features]
        df = df[[c for c in keep if c in df.columns]]

        # 5. dedup on the (timestamp, asset_cols) primary key. Materialised
        # views and ReplacingMergeTree-backed tables routinely emit several
        # rows for the same logical key while merges catch up; without this
        # the pivot below would crash on duplicate index entries. We keep
        # the last row per key — combined with the SQL ``ORDER BY timestamp``
        # this yields a deterministic choice across runs.
        dedup_keys = [self.timestamp_col, *(self.asset_cols or ())]
        df = df.drop_duplicates(subset=dedup_keys, keep="last")

        # 6. pivot for multi-asset, or set index for single. ``_symbol`` is
        # built via ``astype(str)`` so both levels of the unstacked MultiIndex
        # are guaranteed str-typed without an extra rebuild.
        if self.asset_cols is not None:
            ac = list(self.asset_cols)
            df["_symbol"] = df[ac].astype(str).agg(self.asset_separator.join, axis=1)
            df = df.drop(columns=ac)
            df = df.set_index([self.timestamp_col, "_symbol"]).unstack("_symbol")
        else:
            df = df.set_index(self.timestamp_col)
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = None

        # 7. canonicalise field names + ordering
        df = normalize_ohlcv_columns(df)
        df = canonicalize_ohlcv_order(df)
        df = df.sort_index()

        # 8. apply user-side columns= filter (level-0 of MultiIndex; field name otherwise)
        if columns is not None and not df.empty:
            wanted = set(columns)
            if isinstance(df.columns, pd.MultiIndex):
                mask = [c[0] in wanted for c in df.columns]
                df = df.loc[:, mask]
            else:
                df = df[[c for c in df.columns if c in wanted]]

        return df


# -------------------------------------------------------------------- helpers


def _require_clickhouse_connect() -> Any:
    try:
        import clickhouse_connect
    except ImportError as e:  # pragma: no cover — exercised by extras-missing test
        msg = (
            "clickhouse-connect is required for ClickHouse. "
            "Install with: uv add 'fundcloud[data-clickhouse]' or 'fundcloud[data]'."
        )
        raise ImportError(msg) from e
    return clickhouse_connect


def _q(name: str) -> str:
    """Quote a Clickhouse identifier with backticks, escaping any embedded backticks."""
    return "`" + name.replace("`", "``") + "`"


def _q_table(name: str) -> str:
    """Quote ``db.table`` (or just ``table``) preserving the dot separator."""
    if "." in name:
        db, tbl = name.split(".", 1)
        return f"{_q(db)}.{_q(tbl)}"
    return _q(name)
