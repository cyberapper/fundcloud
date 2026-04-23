"""Shared live-data helper for the skfolio-style examples.

Every scenario 11-15 takes the same shape of input — a wide ``DataFrame``
of daily closes indexed by date, one column per asset — so we factor the
yfinance pull into one place. Callers can pass either a flat list of
tickers or a mapping ``{display_name: ticker}`` to rename the output
columns (e.g. present "AGG" as ``BONDS_AGG`` in the final table).

The helper returns ``None`` (after printing a friendly message) when
yfinance isn't installed or the network call fails, so examples stay
demoable offline without wrapping every call site in try/except.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import pandas as pd

__all__ = ["pull_closes"]


def pull_closes(
    symbols: Sequence[str] | dict[str, str],
    *,
    years: int = 5,
    end: pd.Timestamp | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Return a wide ``(date × asset)`` frame of daily close prices.

    Parameters
    ----------
    symbols
        Either a list of tickers (used as column names) or a mapping from
        display name to ticker. The mapping form is useful when you want
        to run HRP / MeanRisk output tables with meaningful labels
        (``BONDS_AGG`` rather than ``AGG``) without touching the fetcher.
    years
        Lookback window in years from ``end``.
    end
        Upper bound (inclusive). Defaults to today.
    """
    try:
        from fundcloud.data import YF
    except ImportError:  # pragma: no cover
        if verbose:
            print("This example needs yfinance — `uv add 'fundcloud[data-yf]'`", file=sys.stderr)
        return None

    mapping = dict(symbols) if isinstance(symbols, dict) else {s: s for s in symbols}

    tickers = list(mapping.values())
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.DateOffset(years=years)

    try:
        bars = YF(symbols=tickers, interval="1d").read(start=start_ts, end=end_ts)
    except Exception as e:
        if verbose:
            print(f"yfinance request failed: {e}", file=sys.stderr)
        return None

    if bars.empty:
        if verbose:
            print("yfinance returned an empty frame — retry later.", file=sys.stderr)
        return None

    closes = bars.xs("close", axis=1, level=0)
    # Invert the display→ticker mapping so the output columns carry the
    # human-facing label rather than the raw ticker.
    reverse = {ticker: display for display, ticker in mapping.items()}
    closes = closes.rename(columns=reverse).loc[:, list(mapping.keys())]
    return closes.dropna(how="any")
