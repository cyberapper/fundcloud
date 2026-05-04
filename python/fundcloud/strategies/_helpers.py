"""Shared helpers for strategy classes.

Kept private (leading underscore) — not part of the public surface.
"""

from __future__ import annotations

import pandas as pd

from fundcloud.portfolio import Portfolio

__all__ = ["_assets_from_bars", "_current_equity"]


def _assets_from_bars(bars: pd.DataFrame) -> list[str]:
    """Distinct asset tickers from a Bars frame's columns."""
    if isinstance(bars.columns, pd.MultiIndex):
        return list(bars.columns.get_level_values(-1).unique())
    return [str(c) for c in bars.columns]


def _current_equity(portfolio: Portfolio) -> float:
    """Latest portfolio equity available at decide-time.

    The simulator marks-to-market *after* it calls ``Strategy.decide``, so
    on bar ``t`` the most recent ``equity_history`` entry reflects the
    close of bar ``t-1``. On the first bar (and on any analytics-only
    portfolios with no equity history yet) we fall back to live cash —
    which equals starting capital before any orders fill.
    """
    history = portfolio._live.equity_history
    if history:
        return float(history[-1][1])
    return float(portfolio.cash)
