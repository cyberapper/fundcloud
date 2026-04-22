"""Alpha Vantage backend. Read-only."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend
from fundcloud.data._columns import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_ohlcv_columns,
)
from fundcloud.data._defaults import default_start_one_year_back
from fundcloud.data._http import HttpClient

__all__ = ["AV"]

_AV_BASE_URL = "https://www.alphavantage.co"

# Alpha Vantage uses different endpoint names per resolution and per
# adjustment. Keys: (interval, adjusted) -> (function_name, payload_key).
_AV_FUNCTION_MAP: dict[tuple[str, bool], tuple[str, str]] = {
    ("1d", True): ("TIME_SERIES_DAILY_ADJUSTED", "Time Series (Daily)"),
    ("1d", False): ("TIME_SERIES_DAILY", "Time Series (Daily)"),
    ("1wk", True): ("TIME_SERIES_WEEKLY_ADJUSTED", "Weekly Adjusted Time Series"),
    ("1wk", False): ("TIME_SERIES_WEEKLY", "Weekly Time Series"),
    ("1mo", True): ("TIME_SERIES_MONTHLY_ADJUSTED", "Monthly Adjusted Time Series"),
    ("1mo", False): ("TIME_SERIES_MONTHLY", "Monthly Time Series"),
}


class AV(BaseBackend):
    """Pull daily / weekly / monthly OHLCV bars from Alpha Vantage.

    Parameters
    ----------
    symbols
        One ticker (string) or many (sequence).
    interval
        One of ``1d``, ``1wk``, ``1mo``.
    adjust
        Default ``True``. Hits the ``TIME_SERIES_*_ADJUSTED`` endpoint
        and publishes the dividend/split-adjusted close under the
        canonical ``close`` column. Set to ``False`` for raw,
        as-traded prices via the unadjusted endpoint family
        (``TIME_SERIES_DAILY`` etc., free-tier friendly).
    api_key
        Falls back to ``ALPHAVANTAGE_API_KEY`` or
        ``ALPHA_VANTAGE_API_KEY`` env var.
    base_url
        Override the default endpoint (useful for tests).
    """

    name: ClassVar[str] = "av"
    read_only = True

    def __init__(
        self,
        symbols: Sequence[str] | str,
        *,
        interval: str = "1d",
        adjust: bool = True,
        api_key: str | None = None,
        base_url: str = _AV_BASE_URL,
    ) -> None:
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        if not self.symbols:
            raise ValueError("AV requires at least one symbol")
        if (interval, True) not in _AV_FUNCTION_MAP:
            msg = f"interval {interval!r} not supported by AV"
            raise ValueError(msg)
        self.interval = interval
        self.adjust = adjust
        self._api_key = (
            api_key
            or os.environ.get("ALPHAVANTAGE_API_KEY")
            or os.environ.get("ALPHA_VANTAGE_API_KEY")
        )
        if not self._api_key:
            msg = (
                "AV requires an API key. Pass `api_key=` or set the "
                "ALPHAVANTAGE_API_KEY (or ALPHA_VANTAGE_API_KEY) environment variable."
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
        function, ts_key = _AV_FUNCTION_MAP[(self.interval, self.adjust)]
        frames: dict[str, pd.DataFrame] = {}
        with HttpClient(base_url=self._base_url, params={"apikey": self._api_key}) as client:
            for sym in self.symbols:
                payload = client.get_json(
                    "/query",
                    params={
                        "function": function,
                        "symbol": sym,
                        "outputsize": "full",
                    },
                )
                frames[sym] = _parse(payload.get(ts_key, {}), adjust=self.adjust)
        if not any(len(df) for df in frames.values()):
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
        wide.columns = wide.columns.swaplevel(0, 1)
        wide = normalize_ohlcv_columns(wide)
        wide = canonicalize_ohlcv_order(wide).sort_index()
        if start is not None or end is not None:
            wide = wide.loc[start:end]  # type: ignore[misc]
        if columns is not None and not wide.empty and isinstance(wide.columns, pd.MultiIndex):
            wanted = set(columns)
            mask = [c[0] in wanted for c in wide.columns]
            wide = wide.loc[:, mask]
        return wide

    def keys(self) -> list[str]:
        return list(self.symbols)


# -------------------------------------------------------------------- helpers


def _parse(series: dict[str, dict[str, str]], *, adjust: bool) -> pd.DataFrame:
    """Alpha Vantage returns ``{"YYYY-MM-DD": {"1. open": ...}}`` style.

    The adjusted endpoint ships ``5. adjusted close`` and ``6. volume``;
    the raw endpoint ships ``4. close`` and ``5. volume``. We surface the
    user-selected close into the canonical ``close`` slot.
    """
    if not series:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    close_key = "5. adjusted close" if adjust else "4. close"
    volume_key = "6. volume" if adjust else "5. volume"
    rows = []
    for ts_str, fields in series.items():
        row = {
            "open": float(fields.get("1. open", "nan")),
            "high": float(fields.get("2. high", "nan")),
            "low": float(fields.get("3. low", "nan")),
            "close": float(fields.get(close_key, "nan")),
            "volume": float(fields.get(volume_key, "nan")),
        }
        rows.append((pd.Timestamp(ts_str), row))
    rows.sort(key=lambda r: r[0])
    index = pd.DatetimeIndex([r[0] for r in rows])
    df = pd.DataFrame([r[1] for r in rows], index=index)
    df.index.name = None
    return df
