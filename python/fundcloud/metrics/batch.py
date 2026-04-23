"""Panel-wide ("batch") metrics.

Where :mod:`fundcloud.metrics.core` returns a scalar per Series (or a Series
per DataFrame), the ``batch_*`` functions accept a **dict of DataFrames** —
one frame per strategy, each frame's columns are assets — and return a
comparison frame indexed by strategy with one column per metric.

This is the shape you want for grid searches and walk-forward analyses,
where you end up with ``(n_strategies × n_assets × n_periods)`` worth of
returns to summarise.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fundcloud.metrics import core as _core

__all__ = [
    "batch_cvar",
    "batch_max_drawdown",
    "batch_sharpe",
    "batch_sortino",
    "batch_summary",
]


def _reduce_returns(returns: pd.Series | pd.DataFrame) -> pd.Series:
    """Reduce a panel (DataFrame) to a single per-period portfolio return.

    Assumes each row is already a total-portfolio return. If the caller passed
    a per-asset panel, they should aggregate to a strategy-level series first
    (e.g. ``panel @ weights``) — this function conservatively averages.
    """
    if isinstance(returns, pd.Series):
        return returns
    return returns.mean(axis=1)


def batch_sharpe(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> pd.Series:
    rows = {
        name: _core.sharpe(
            _reduce_returns(r),
            risk_free=risk_free,
            periods_per_year=periods_per_year,
        )
        for name, r in strategies.items()
    }
    return pd.Series(rows, name="sharpe", dtype=float)


def batch_sortino(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
    *,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> pd.Series:
    rows = {
        name: _core.sortino(_reduce_returns(r), target=target, periods_per_year=periods_per_year)
        for name, r in strategies.items()
    }
    return pd.Series(rows, name="sortino", dtype=float)


def batch_max_drawdown(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
) -> pd.Series:
    rows = {name: _core.max_drawdown(_reduce_returns(r)) for name, r in strategies.items()}
    return pd.Series(rows, name="max_drawdown", dtype=float)


def batch_cvar(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
    *,
    alpha: float = 0.95,
) -> pd.Series:
    rows = {name: _core.cvar(_reduce_returns(r), alpha=alpha) for name, r in strategies.items()}
    return pd.Series(rows, name="cvar", dtype=float)


def batch_summary(
    strategies: Mapping[str, pd.Series | pd.DataFrame],
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
    cvar_alpha: float = 0.95,
) -> pd.DataFrame:
    """One row per strategy, standard metrics as columns."""
    if not strategies:
        return pd.DataFrame()
    rows = {}
    for name, r in strategies.items():
        s = _reduce_returns(r)
        rows[name] = _core.returns_stats(
            s,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
        ).iloc[:, 0]
    out = pd.DataFrame(rows).T
    # Enforce float dtype; sklearn/skfolio sometimes hands us object columns.
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
