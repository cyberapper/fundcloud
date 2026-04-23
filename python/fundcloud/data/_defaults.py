"""Shared defaults applied at the top of every network backend's ``read``.

Network providers can return decades of history when no window is given,
which is rarely what the caller wants and burns bandwidth. Each network
backend resolves an unspecified ``start`` to ``end - 1 year`` via
:func:`default_start_one_year_back` so that a bare
``YF("SPY").read()`` pulls one year, not twenty.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["default_start_one_year_back", "interval_aware_default_start"]


def interval_aware_default_start(
    interval: str,
    end: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return a default start date appropriate for the given data interval."""
    _end = end if end is not None else pd.Timestamp.utcnow()
    _days: dict[str, int] = {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "90m": 60,
        "60m": 730,
        "1h": 730,
        "2h": 730,
        "4h": 730,
    }
    days = _days.get(interval, 365)
    return (_end - pd.Timedelta(days=days)).normalize()


def default_start_one_year_back(
    start: pd.Timestamp | str | None,
    end: pd.Timestamp | str | None,
) -> pd.Timestamp | str | None:
    """Return ``start`` if given; otherwise compute ``end - 1 year``.

    ``end`` defaults to today (normalised to midnight) when also missing.
    Storage backends (``Parquet``, ``DuckDB``, ``Memory``, ``CSV``) do
    *not* call this — they return whatever is cached when no window is
    given, which is the more useful default for a local store.
    """
    if start is not None:
        return start
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now().normalize()
    return end_ts - pd.DateOffset(years=1)
