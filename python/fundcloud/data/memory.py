"""In-memory backend — keyed dict of DataFrames; great for tests and fixtures.

Subsumes the legacy ``InMemoryStore`` (multi-key cache) and ``PandasData``
(single-frame source wrapper). Pass ``initial={"key": df, ...}`` to preload;
construct empty and call ``write(key, df)`` to populate over time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend, WriteMode

__all__ = ["Memory"]


class Memory(BaseBackend):
    """Dict-backed :class:`fundcloud.data.Backend`."""

    name: ClassVar[str] = "memory"

    def __init__(
        self,
        initial: Mapping[str, pd.DataFrame] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self.read_only = read_only
        self._data: dict[str, pd.DataFrame] = {}
        if initial:
            for key, df in initial.items():
                if not isinstance(df.index, pd.DatetimeIndex):
                    msg = f"frame for key {key!r} must have a DatetimeIndex"
                    raise TypeError(msg)
                self._data[key] = df.sort_index().copy()

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
            if len(self._data) == 1:
                key = next(iter(self._data))
            else:
                msg = "Memory backend has multiple keys; pass key= explicitly"
                raise KeyError(msg)
        if key not in self._data:
            raise KeyError(key)
        df = self._data[key]
        if start is not None or end is not None:
            df = df.loc[start:end]  # type: ignore[misc]
        if columns is not None:
            df = df[list(columns)]
        return df.copy()

    def keys(self) -> list[str]:
        return sorted(self._data)

    def exists(self, key: str) -> bool:
        return key in self._data

    def last_index(self, key: str | None = None) -> pd.Timestamp | None:
        if key is None and len(self._data) == 1:
            key = next(iter(self._data))
        if key is None or key not in self._data or self._data[key].empty:
            return None
        return pd.Timestamp(self._data[key].index[-1])

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
        if mode == "error" and key in self._data:
            raise FileExistsError(key)
        if mode == "append" and key in self._data:
            self._data[key] = pd.concat([self._data[key], df]).sort_index()
            return
        if mode == "upsert" and key in self._data:
            combined = pd.concat([self._data[key], df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            self._data[key] = combined
            return
        # overwrite (or first write under append/upsert)
        self._data[key] = df.sort_index().copy()

    def delete(self, key: str) -> None:
        self._check_writable()
        self._data.pop(key, None)
