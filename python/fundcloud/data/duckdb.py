"""DuckDB-backed backend — one table per key inside a single database file."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import ClassVar

import duckdb
import pandas as pd

from fundcloud.data._base import BaseBackend, WriteMode

__all__ = ["DuckDB"]


# Internal column name used to materialise the DataFrame index inside DuckDB.
# The user's original ``index.name`` is intentionally *not* round-tripped;
# ``read`` always yields a frame with ``index.name = None`` to keep behaviour
# consistent across every ``Backend`` implementation.
_TS_COL = "_fc_ts"

# Separator used to flatten ``(field, symbol)`` MultiIndex columns into single
# SQL-safe strings on write, and to rebuild the MultiIndex on read. Chosen so
# it won't collide with real OHLCV field or ticker strings.
_COL_SEP = "||"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns into ``"field||symbol"`` strings.

    Always returns a copy so callers can safely mutate ``index.name`` etc.
    without touching the user's frame.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [_COL_SEP.join(str(p) for p in tup) for tup in out.columns]
    return out


def _unflatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the ``(field, symbol)`` MultiIndex if any column carries the sep."""
    cols = [str(c) for c in df.columns]
    if not any(_COL_SEP in c for c in cols):
        return df
    tuples = [tuple(c.split(_COL_SEP)) for c in cols]
    out = df.copy()
    out.columns = pd.MultiIndex.from_tuples(tuples)
    return out


def _table_name(key: str) -> str:
    # Map "a/b/c" -> "a__b__c" since DuckDB identifiers don't allow slashes
    # but we still want keys to round-trip.
    safe = key.replace("/", "__")
    if not safe or not safe[0].isalpha():
        safe = f"t_{safe}"
    return safe


def _key_from_table(name: str) -> str:
    key = name.removeprefix("t_")
    return key.replace("__", "/")


class DuckDB(BaseBackend):
    """Persist frames as DuckDB tables inside a single database file."""

    name: ClassVar[str] = "duckdb"

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path), read_only=read_only)

    # Context-manager support (preserved from DuckDBStore).
    def __enter__(self) -> DuckDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._con.close()

    # ------------------------------------------------------------------ read

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if key is None:
            ks = self.keys()
            if len(ks) != 1:
                msg = "DuckDB database has multiple keys; pass key= explicitly"
                raise KeyError(msg)
            key = ks[0]
        if not self.exists(key):
            raise KeyError(key)
        table = _table_name(key)
        where: list[str] = []
        params: list[object] = []
        if start is not None:
            where.append(f'"{_TS_COL}" >= ?')
            params.append(pd.Timestamp(start))
        if end is not None:
            where.append(f'"{_TS_COL}" <= ?')
            params.append(pd.Timestamp(end))
        # SELECT * + post-filter in pandas so both flat and MultiIndex (flattened)
        # column layouts honor ``columns=`` uniformly.
        sql = f'SELECT * FROM "{table}"'
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f' ORDER BY "{_TS_COL}"'
        df = self._con.execute(sql, params).fetch_df()
        df[_TS_COL] = pd.to_datetime(df[_TS_COL])
        df = df.set_index(_TS_COL)
        df.index.name = None
        df = _unflatten_columns(df)
        if columns is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                wanted = set(columns)
                mask = [c[0] in wanted for c in df.columns]
                df = df.loc[:, mask]
            else:
                df = df[list(columns)]
        return df

    def keys(self) -> list[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [_key_from_table(r[0]) for r in rows]

    def exists(self, key: str) -> bool:
        table = _table_name(key)
        row = self._con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        return row is not None

    def last_index(self, key: str | None = None) -> pd.Timestamp | None:
        if key is None:
            ks = self.keys()
            if len(ks) != 1:
                return None
            key = ks[0]
        if not self.exists(key):
            return None
        table = _table_name(key)
        row = self._con.execute(f'SELECT max("{_TS_COL}") FROM "{table}"').fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0])

    # ----------------------------------------------------------------- write

    def write(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        mode: WriteMode = "overwrite",
    ) -> None:
        self._check_writable()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df must have a DatetimeIndex")
        table = _table_name(key)
        exists = self.exists(key)
        if mode == "error" and exists:
            raise FileExistsError(key)

        prepared = _flatten_columns(df)
        prepared.index.name = _TS_COL
        prepared = prepared.reset_index()
        self._con.register("__fc_tmp", prepared)
        try:
            if mode == "append" and exists:
                self._con.execute(f'INSERT INTO "{table}" SELECT * FROM __fc_tmp')
            elif mode == "upsert" and exists:
                self._con.execute(f'INSERT INTO "{table}" SELECT * FROM __fc_tmp')
                self._con.execute(
                    f'CREATE OR REPLACE TABLE "{table}" AS '
                    f"SELECT * EXCLUDE (__rk) FROM (SELECT *, ROW_NUMBER() OVER "
                    f'(PARTITION BY "{_TS_COL}" ORDER BY rowid DESC) AS __rk '
                    f'FROM "{table}") WHERE __rk = 1'
                )
            else:
                self._con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM __fc_tmp')
        finally:
            self._con.unregister("__fc_tmp")

    def delete(self, key: str) -> None:
        self._check_writable()
        if self.exists(key):
            self._con.execute(f'DROP TABLE "{_table_name(key)}"')
