"""FinancialModelingPrep (FMP) backend. Read-only."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend
from fundcloud.data._columns import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_field,
    normalize_ohlcv_columns,
)
from fundcloud.data._defaults import default_start_one_year_back
from fundcloud.data._http import HttpClient

__all__ = ["FMP"]

_FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

_FMP_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "daily",
}


class FMP(BaseBackend):
    """Pull OHLCV bars from the FinancialModelingPrep REST API.

    Parameters
    ----------
    symbols
        One ticker (string) or many (sequence).
    interval
        One of ``1m``, ``5m``, ``15m``, ``30m``, ``1h``, ``4h``, ``1d``.
    adjust
        Default ``True``. The FMP daily endpoint returns both the raw
        close and a dividend/split-adjusted close (``adjClose``). When
        ``True``, the adjusted value is published under the canonical
        ``close`` column. Set to ``False`` to keep the raw, as-traded
        prices.
    api_key
        Falls back to the ``FMP_API_KEY`` env var.
    base_url
        Override the default FMP endpoint (useful for tests).
    """

    name: ClassVar[str] = "fmp"
    read_only = True

    def __init__(
        self,
        symbols: Sequence[str] | str,
        *,
        interval: str = "1d",
        adjust: bool = True,
        api_key: str | None = None,
        base_url: str = _FMP_BASE_URL,
    ) -> None:
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        if not self.symbols:
            raise ValueError("FMP requires at least one symbol")
        if interval not in _FMP_INTERVAL_MAP:
            msg = f"interval {interval!r} not supported by FMP"
            raise ValueError(msg)
        self.interval = interval
        self.adjust = adjust
        self._api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self._api_key:
            msg = (
                "FMP requires an API key. Pass `api_key=` or set the "
                "FMP_API_KEY environment variable."
            )
            raise ValueError(msg)
        self._base_url = base_url

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
        frames: dict[str, pd.DataFrame] = {}
        with HttpClient(base_url=self._base_url, params={"apikey": self._api_key}) as client:
            for sym in self.symbols:
                frames[sym] = self._fetch_symbol(client, sym, start=start, end=end)
        frames = {sym: df for sym, df in frames.items() if not df.empty}
        if not frames:
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

    def _fetch_symbol(
        self,
        client: HttpClient,
        symbol: str,
        *,
        start: pd.Timestamp | str | None,
        end: pd.Timestamp | str | None,
    ) -> pd.DataFrame:
        interval = _FMP_INTERVAL_MAP[self.interval]
        if interval == "daily":
            url = f"/historical-price-full/{symbol}"
            params: dict[str, str] = {}
            if start is not None:
                params["from"] = str(pd.Timestamp(start).date())
            if end is not None:
                params["to"] = str(pd.Timestamp(end).date())
            payload = client.get_json(url, params=params)
            rows = payload.get("historical", []) if isinstance(payload, dict) else []
        else:
            url = f"/historical-chart/{interval}/{symbol}"
            params = {}
            if start is not None:
                params["from"] = str(pd.Timestamp(start).date())
            if end is not None:
                params["to"] = str(pd.Timestamp(end).date())
            payload = client.get_json(url, params=params)
            rows = payload if isinstance(payload, list) else []
        if not rows:
            return pd.DataFrame(columns=list(OHLCV_COLUMNS))
        df = pd.DataFrame(rows)
        df.columns = [normalize_field(c) for c in df.columns]  # case-insensitive provider keys
        df["date"] = pd.to_datetime(df["date"])
        if self.adjust and "adj_close" in df.columns:
            # Promote the dividend/split-adjusted close into the canonical
            # ``close`` slot so downstream code always sees adjusted prices.
            df["close"] = df["adj_close"]
        keep = [c for c in OHLCV_COLUMNS if c in df.columns]
        df = df.set_index("date")[keep].astype(float)
        df.index.name = None
        return df.sort_index()
