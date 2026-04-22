"""Parquet-backed backend — one ``.parquet`` file per key under ``root``.

Slashes in keys become filesystem separators, so ``"equity/us/daily"``
lives at ``<root>/equity/us/daily.parquet``. ``last_index()`` peeks at
pyarrow metadata to avoid reading the whole file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend, WriteMode

__all__ = ["Parquet"]


class Parquet(BaseBackend):
    """Per-key parquet files under a root directory."""

    name: ClassVar[str] = "parquet"

    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        self.root = Path(root)
        self.read_only = read_only
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*key.split("/")).with_suffix(".parquet")

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
                msg = "Parquet root has multiple keys; pass key= explicitly"
                raise KeyError(msg)
            key = ks[0]
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        df = pd.read_parquet(path)
        if start is not None or end is not None:
            df = df.loc[start:end]  # type: ignore[misc]
        if columns is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                wanted = set(columns)
                mask = [c[0] in wanted for c in df.columns]
                df = df.loc[:, mask]
            else:
                df = df[list(columns)]
        return df

    def keys(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root).with_suffix("")).replace("\\", "/")
            for p in self.root.rglob("*.parquet")
        )

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def last_index(self, key: str | None = None) -> pd.Timestamp | None:
        if key is None:
            ks = self.keys()
            if len(ks) != 1:
                return None
            key = ks[0]
        path = self._path(key)
        if not path.exists():
            return None
        # Use pyarrow's metadata to avoid reading the whole file.
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path)
        if meta.metadata.num_rows == 0:
            return None
        last_rg = meta.metadata.num_row_groups - 1
        index_cols = [c for c in meta.schema_arrow.names if c.startswith("__index")]
        if not index_cols:
            df = pd.read_parquet(path)
            return pd.Timestamp(df.index[-1]) if len(df) else None
        table = meta.read_row_group(last_rg, columns=index_cols)
        return pd.Timestamp(table.column(0)[-1].as_py())

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
        path = self._path(key)
        if path.exists():
            if mode == "error":
                raise FileExistsError(path)
            if mode == "append":
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df]).sort_index()
            elif mode == "upsert":
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, df])
                df = combined[~combined.index.duplicated(keep="last")].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_index().to_parquet(path)

    def delete(self, key: str) -> None:
        self._check_writable()
        self._path(key).unlink(missing_ok=True)
