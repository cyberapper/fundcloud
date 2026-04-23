"""CSV backend — read-only.

Single file or directory of files. With a directory, each file's stem
becomes a symbol and the result has a two-level ``(field, symbol)``
column index.

Defaults to ``read_only=True``: CSV's append story is messy (no clean
keyed multi-frame layout, ambiguous schema rules), so writes are
disabled by default. Pass ``read_only=False`` to allow overwrite.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend, WriteMode

__all__ = ["CSV"]


class CSV(BaseBackend):
    """Read one or many CSV files into a single frame."""

    name: ClassVar[str] = "csv"

    def __init__(
        self,
        path: str | Path,
        *,
        symbols: Sequence[str] | None = None,
        date_col: str = "date",
        read_only: bool = True,
    ) -> None:
        self.path = Path(path)
        self.symbols = list(symbols) if symbols is not None else []
        self.date_col = date_col
        self.read_only = read_only

    # ------------------------------------------------------------------ read

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if self.path.is_file():
            df = self._read_one(self.path)
            if not self.symbols:
                self.symbols = [self.path.stem]
        else:
            frames: dict[str, pd.DataFrame] = {}
            candidates = sorted(self.path.glob("*.csv"))
            if self.symbols:
                wanted = set(self.symbols)
                candidates = [p for p in candidates if p.stem in wanted]
            if not candidates:
                raise FileNotFoundError(f"No matching CSVs under {self.path}")
            for p in candidates:
                frames[p.stem] = self._read_one(p)
            self.symbols = list(frames)
            df = pd.concat(frames, axis=1)
            cols = df.columns
            assert isinstance(cols, pd.MultiIndex)
            df.columns = cols.swaplevel(0, 1)
            df = df.sort_index(axis=1)

        df = df.sort_index()
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
        if self.path.is_file():
            return [self.path.stem]
        return sorted(p.stem for p in self.path.glob("*.csv"))

    # ----------------------------------------------------------------- write

    def write(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        mode: WriteMode = "overwrite",
    ) -> None:
        self._check_writable()
        if mode != "overwrite":
            msg = f"CSV backend only supports mode='overwrite' (got {mode!r})"
            raise NotImplementedError(msg)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df must have a DatetimeIndex")
        # Treat path as single-file when it has a .csv suffix; else as directory.
        out = self.path if self.path.suffix.lower() == ".csv" else self.path / f"{key}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.sort_index().rename_axis(self.date_col).to_csv(out)

    # ------------------------------------------------------------------ helpers

    def _read_one(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if self.date_col not in df.columns:
            msg = f"{path.name} missing date column {self.date_col!r}"
            raise KeyError(msg)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col).sort_index()
        df.index.name = None
        return df
