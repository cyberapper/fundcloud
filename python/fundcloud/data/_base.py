"""``Backend`` protocol and ``BaseBackend`` ABC for every data backend.

A :class:`Backend` is the unified abstraction for both data providers
(YF, FMP, …) and persistent caches (Parquet, DuckDB, in-memory). Every
backend supports :meth:`read`; writes are gated by the ``read_only``
constructor flag and raise :class:`ReadOnlyError` when locked.

The :meth:`Backend.sync_to` shortcut composes a source backend onto a
sink backend in one call: ``YF(["SPY"]).sync_to(DuckDB("warehouse.duckdb"),
key="us_eq", mode="upsert")``. :class:`fundcloud.data.Catalog` orchestrates
declarative source-to-sink pipelines on top of the same primitive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd

__all__ = ["Backend", "BaseBackend", "ReadOnlyError", "WriteMode"]


WriteMode = Literal["overwrite", "append", "upsert", "error"]


class ReadOnlyError(RuntimeError):
    """Raised when ``write`` / ``delete`` is called on a read-only backend."""


@runtime_checkable
class Backend(Protocol):
    """Unified protocol for any data backend.

    Every backend is readable. Backends with ``read_only=False`` also
    accept :meth:`write` and :meth:`delete`. The ``key`` argument is the
    logical dataset name; single-key sources accept ``key=None``.
    """

    name: ClassVar[str]
    read_only: bool

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame: ...

    def keys(self) -> list[str]: ...

    def exists(self, key: str) -> bool: ...

    def last_index(self, key: str | None = None) -> pd.Timestamp | None: ...

    def write(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        mode: WriteMode = "overwrite",
    ) -> None: ...

    def delete(self, key: str) -> None: ...

    def sync_to(
        self,
        sink: Backend,
        *,
        key: str | None = None,
        source_key: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        mode: WriteMode = "upsert",
    ) -> pd.DataFrame: ...


class BaseBackend(ABC):
    """Default implementations shared by every concrete backend.

    Subclasses must implement :meth:`read` and set the ``name`` ClassVar.
    Writable backends override :meth:`write` and :meth:`delete` (and call
    :meth:`_check_writable` first).
    """

    name: ClassVar[str]
    read_only: bool = False

    @abstractmethod
    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame: ...

    def keys(self) -> list[str]:
        return [self._default_key()]

    def exists(self, key: str) -> bool:
        return key in self.keys()

    def last_index(self, key: str | None = None) -> pd.Timestamp | None:
        try:
            df = self.read(key)
        except KeyError:
            return None
        if df.empty:
            return None
        return pd.Timestamp(df.index.max())

    def write(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        mode: WriteMode = "overwrite",
    ) -> None:
        self._check_writable()
        msg = f"{type(self).__name__} does not implement write()"
        raise NotImplementedError(msg)

    def delete(self, key: str) -> None:
        self._check_writable()
        msg = f"{type(self).__name__} does not implement delete()"
        raise NotImplementedError(msg)

    def sync_to(
        self,
        sink: Backend,
        *,
        key: str | None = None,
        source_key: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        mode: WriteMode = "upsert",
    ) -> pd.DataFrame:
        """Read from ``self`` and write the result to ``sink`` under ``key``.

        ``key`` is the *sink* key (where to land the data). ``source_key`` is
        the *source* key (where to read from); defaults to ``None`` so the
        source picks its own canonical frame (e.g. a network backend ignores
        the key, a single-frame format backend resolves to its lone entry).
        """
        df = self.read(source_key, start=start, end=end)
        target_key = key if key is not None else (
            source_key if source_key is not None else self._default_key()
        )
        sink.write(target_key, df, mode=mode)
        return df

    def _default_key(self) -> str:
        return "default"

    def _check_writable(self) -> None:
        if self.read_only:
            msg = f"{type(self).__name__} is read-only"
            raise ReadOnlyError(msg)
