"""FundCloud market-data backend. Read-only.

Wraps the ``/market/{symbol}/ohlcv`` endpoint of the FundCloud public
platform API, returning OHLCV bars in the same wide, MultiIndex shape
that every other network backend in :mod:`fundcloud.data` produces
(:class:`FMP`, :class:`AV`, :class:`YF`, :class:`Binance`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pandas as pd

from fundcloud._clients.fundcloud import FUNDCLOUD_BASE_URL, FundCloudClient
from fundcloud.data._base import BaseBackend
from fundcloud.data._columns import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_ohlcv_columns,
)

__all__ = ["FundCloud"]

# The public API exposes only daily and weekly bars.
_FUNDCLOUD_INTERVAL_MAP: dict[str, str] = {"1d": "1D", "1wk": "1W"}
# `period` is the API's window enum. We default to the widest (2Y) and
# slice locally with `start` / `end`, mirroring how :class:`AV` handles
# its coarser free-tier API.
_FUNDCLOUD_PERIOD_ENUMS: tuple[str, ...] = ("1M", "3M", "6M", "1Y", "2Y")
_FUNDCLOUD_PERIOD_DEFAULT: str = "2Y"


class FundCloud(BaseBackend):
    """Pull OHLCV bars from the FundCloud market-data API.

    Parameters
    ----------
    symbols
        One ticker (string) or many (sequence). FundCloud's symbology
        auto-detects the asset category (``AAPL`` → US stock,
        ``BTCUSDT`` → crypto, ``EURUSD`` → forex, …).
    interval
        One of ``1d``, ``1wk``. The public API does not expose intraday
        bars.
    period
        Window the API returns before local slicing by ``start`` /
        ``end`` in :meth:`read`. One of ``1M``, ``3M``, ``6M``, ``1Y``,
        ``2Y``; defaults to ``2Y`` (the widest the API offers). Override
        with a smaller window to reduce payload size when a shorter
        history is sufficient.
    adjust
        Default ``True``. The API's ``adjusted=true`` endpoint returns
        split/dividend-adjusted prices in the canonical ``close`` slot,
        matching how :class:`FMP` and :class:`AV` handle adjustment.
        Set to ``False`` for raw, as-traded prices.
    api_key
        Falls back to the ``FUNDCLOUD_API_KEY`` env var.
    base_url
        Override the API base URL (useful in tests).

    Notes
    -----
    Auth uses ``X-API-Key`` via the shared
    :class:`fundcloud._clients.fundcloud.FundCloudClient`, which also
    powers :class:`fundcloud.accounts.fundcloud.FundCloud` — one API
    key, one retry policy, one pagination loop across both layers.
    """

    name: ClassVar[str] = "fundcloud"
    read_only = True

    def __init__(
        self,
        symbols: Sequence[str] | str,
        *,
        interval: str = "1d",
        period: str = _FUNDCLOUD_PERIOD_DEFAULT,
        adjust: bool = True,
        api_key: str | None = None,
        base_url: str = FUNDCLOUD_BASE_URL,
    ) -> None:
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        if not self.symbols:
            raise ValueError("FundCloud requires at least one symbol")
        if interval not in _FUNDCLOUD_INTERVAL_MAP:
            msg = (
                f"interval {interval!r} not supported by FundCloud; "
                f"the public API exposes daily ('1d') and weekly ('1wk') only"
            )
            raise ValueError(msg)
        if period not in _FUNDCLOUD_PERIOD_ENUMS:
            msg = (
                f"period {period!r} not supported by FundCloud; "
                f"expected one of {_FUNDCLOUD_PERIOD_ENUMS}"
            )
            raise ValueError(msg)
        self.interval = interval
        self.period = period
        self.adjust = adjust
        self._client = FundCloudClient(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------ read

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        timeframe = _FUNDCLOUD_INTERVAL_MAP[self.interval]
        params: dict[str, str] = {
            "timeframe": timeframe,
            "period": self.period,
            "adjusted": "true" if self.adjust else "false",
        }
        frames: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            payload = self._client.get(f"/market/{sym}/ohlcv", params=params)
            frames[sym] = _parse_candles(payload)
        # Drop empty frames so we don't pollute the MultiIndex with NaN symbols.
        frames = {sym: df for sym, df in frames.items() if not df.empty}
        if not frames:
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


def _parse_candles(payload: object) -> pd.DataFrame:
    """Parse an ``/market/{symbol}/ohlcv`` JSON body into an OHLCV frame.

    Expected shape (per ``components.schemas.OHLCVResponse``):

    .. code-block:: json

        {
          "symbol": "AAPL",
          "candles": [
            {"timestamp": "2024-01-02T00:00:00Z",
             "open": 186.3, "high": 187.8, "low": 185.8,
             "close": 185.6, "volume": 72_000_000},
            ...
          ],
          "timeframe": "1D",
          "period": "1Y",
          "category": "stock-us"
        }
    """
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    candles = payload.get("candles", [])
    if not isinstance(candles, list) or not candles:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    rows: list[dict[str, float]] = []
    index_vals: list[pd.Timestamp] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        ts_raw = c.get("timestamp")
        if ts_raw is None:
            continue
        ts = pd.Timestamp(ts_raw)
        # The API ships UTC ISO8601 for daily/weekly bars. Strip tz info so
        # the index matches the tz-naive convention used across other backends
        # (YF, FMP, AV, Binance all return tz-naive daily indices).
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        index_vals.append(ts)
        rows.append({
            "open": float(c.get("open", float("nan"))),
            "high": float(c.get("high", float("nan"))),
            "low": float(c.get("low", float("nan"))),
            "close": float(c.get("close", float("nan"))),
            "volume": float(c.get("volume", float("nan"))),
        })
    if not rows:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index_vals))
    df.index.name = None
    return df.sort_index()
