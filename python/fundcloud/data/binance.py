"""Binance backend (via the ``ccxt`` package). Read-only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend
from fundcloud.data._columns import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_ohlcv_columns,
)
from fundcloud.data._defaults import default_start_one_year_back

__all__ = ["Binance"]


_CCXT_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1w",
}


class Binance(BaseBackend):
    """Fetch spot OHLCV bars from Binance via ccxt."""

    name: ClassVar[str] = "binance"
    read_only = True

    def __init__(
        self,
        symbols: Sequence[str] | str,
        *,
        interval: str = "1d",
        limit: int = 1000,
        sandbox: bool = False,
    ) -> None:
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        if not self.symbols:
            raise ValueError("Binance requires at least one symbol")
        if interval not in _CCXT_INTERVAL_MAP:
            msg = f"interval {interval!r} not supported by Binance"
            raise ValueError(msg)
        self.interval = interval
        self.limit = int(limit)
        self.sandbox = bool(sandbox)
        self._client: Any | None = None

    # ------------------------------------------------------------------ read

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        start = default_start_one_year_back(start, end)
        ex = self._exchange()
        tf = _CCXT_INTERVAL_MAP[self.interval]
        since_ms = int(pd.Timestamp(start).timestamp() * 1000) if start is not None else None
        until_ms = int(pd.Timestamp(end).timestamp() * 1000) if end is not None else None

        frames: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            frames[sym] = _fetch_all(ex, sym, tf, since_ms, until_ms, self.limit)
        if not any(len(df) for df in frames.values()):
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
        wide.columns = wide.columns.swaplevel(0, 1)
        wide = normalize_ohlcv_columns(wide)
        wide = canonicalize_ohlcv_order(wide).sort_index()
        if columns is not None and not wide.empty and isinstance(wide.columns, pd.MultiIndex):
            wanted = set(columns)
            mask = [c[0] in wanted for c in wide.columns]
            wide = wide.loc[:, mask]
        return wide

    def keys(self) -> list[str]:
        return list(self.symbols)

    # ------------------------------------------------------------------ internals

    def _exchange(self) -> Any:
        if self._client is None:
            ccxt = _require_ccxt()
            self._client = ccxt.binance({"enableRateLimit": True})
            if self.sandbox:
                self._client.set_sandbox_mode(True)
        return self._client


# -------------------------------------------------------------------- helpers


def _require_ccxt() -> Any:
    try:
        import ccxt
    except ImportError as e:
        msg = (
            "ccxt is required for Binance. "
            "Install with: uv add 'fundcloud[data-bn]' or 'fundcloud[data]'."
        )
        raise ImportError(msg) from e
    return ccxt


def _fetch_all(
    exchange: Any,
    symbol: str,
    timeframe: str,
    since_ms: int | None,
    until_ms: int | None,
    limit: int,
) -> pd.DataFrame:
    """Iteratively page through Binance OHLCV data until we exhaust the range."""
    all_rows: list[list[float]] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if until_ms is not None and last_ts >= until_ms:
            break
        if len(batch) < limit:
            break
        cursor = last_ts + 1
    if not all_rows:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("ts").astype(float)
    df.index.name = None
    if until_ms is not None:
        df = df.loc[: pd.Timestamp(until_ms, unit="ms")]
    return df
