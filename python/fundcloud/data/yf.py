"""Yahoo Finance backend (via the ``yfinance`` package). Read-only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from fundcloud.data._base import BaseBackend
from fundcloud.data._columns import canonicalize_ohlcv_order, normalize_ohlcv_columns
from fundcloud.data._defaults import interval_aware_default_start

__all__ = ["YF"]


_YF_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YF(BaseBackend):
    """Pull OHLCV bars from Yahoo Finance.

    Free, keyless, and best-effort. Under rate-limit or outage, errors
    propagate with no automatic fallback — treat it as a local-dev
    convenience rather than a production data plane.

    Parameters
    ----------
    symbols
        One ticker (string) or many (sequence).
    interval
        Bar interval; one of ``1m``, ``5m``, ``15m``, ``30m``, ``1h``,
        ``1d``, ``1wk``, ``1mo``.
    adjust
        Default ``True``. yfinance applies dividend + split adjustments
        to the close column. Set to ``False`` to get the raw,
        as-traded prices (which then includes a separate
        ``adj_close`` field).
    """

    name: ClassVar[str] = "yf"
    read_only = True

    def __init__(
        self,
        symbols: Sequence[str] | str,
        *,
        interval: str = "1d",
        adjust: bool = True,
    ) -> None:
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        if not self.symbols:
            raise ValueError("YF requires at least one symbol")
        if interval not in _YF_INTERVAL_MAP:
            msg = f"interval {interval!r} not supported by YF"
            raise ValueError(msg)
        self.interval = interval
        self.adjust = adjust

    # ------------------------------------------------------------------ read

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        start = (
            interval_aware_default_start(
                self.interval, pd.Timestamp(end) if end is not None else None
            )
            if start is None
            else start
        )
        yf = _require_yfinance()
        yf_interval = _YF_INTERVAL_MAP[self.interval]
        df = yf.download(
            tickers=self.symbols,
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=self.adjust,
            progress=False,
            group_by="column",
            threads=True,
        )
        out = _normalise(df, symbols=self.symbols)
        if columns is not None and not out.empty and isinstance(out.columns, pd.MultiIndex):
            wanted = set(columns)
            mask = [c[0] in wanted for c in out.columns]
            out = out.loc[:, mask]
        return out

    def keys(self) -> list[str]:
        return list(self.symbols)


# -------------------------------------------------------------------- helpers


def _require_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as e:
        msg = (
            "yfinance is required for YF. "
            "Install with: uv add 'fundcloud[data-yf]' or 'fundcloud[data]'."
        )
        raise ImportError(msg) from e
    return yf


def _normalise(df: pd.DataFrame, *, symbols: Sequence[str]) -> pd.DataFrame:
    """Coerce yfinance's output to the canonical ``Bars`` shape."""
    if df.empty:
        return df
    df = df.copy()
    if not isinstance(df.columns, pd.MultiIndex) and len(symbols) == 1:
        df.columns = pd.MultiIndex.from_tuples([(c, symbols[0]) for c in df.columns])
    df = normalize_ohlcv_columns(df)
    df = canonicalize_ohlcv_order(df)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = None
    return df.sort_index()
