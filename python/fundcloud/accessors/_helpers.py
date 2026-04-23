"""Shared helpers between the ``.fc`` Series and DataFrame accessors.

Kept internal (leading underscore); the accessors only expose what users
should call. Any logic that ought to be callable on its own belongs in
:mod:`fundcloud.portfolio` or :mod:`fundcloud.sim`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from fundcloud.portfolio import Portfolio

__all__ = ["portfolio_from_frame", "portfolios_per_column"]


def portfolio_from_frame(
    df: pd.DataFrame | pd.Series,
    *,
    benchmark: pd.Series | None = None,
    weights: pd.Series | dict[str, float] | None = None,
    name: str | None = None,
) -> Portfolio:
    """Build a :class:`~fundcloud.portfolio.Portfolio` from a returns
    Series / DataFrame.

    Policy
    ------
    * :class:`pandas.Series` → single strategy.
    * :class:`pandas.DataFrame` with a single column → that column is the
      strategy series.
    * :class:`pandas.DataFrame` with multiple columns → weighted sum
      across columns. Defaults to equal-weight across columns unless
      ``weights=`` is supplied.

    This is an opinionated default for the accessor renderers
    (``df.fc.render_html()`` etc.). Callers who want finer control
    should construct :class:`Portfolio` directly.
    """
    from fundcloud.portfolio import Portfolio

    if isinstance(df, pd.Series):
        return Portfolio(returns=df, benchmark=benchmark, name=name or str(df.name or "strategy"))

    if df.shape[1] == 0:
        msg = (
            "portfolio_from_frame received a DataFrame with zero columns — "
            "nothing to render. If you're hitting this via ``.fc.render_*`` "
            "with a string ``benchmark=`` that equals your only column, "
            "drop ``benchmark=`` or pass an external benchmark Series."
        )
        raise ValueError(msg)

    if df.shape[1] == 1:
        col = df.iloc[:, 0]
        return Portfolio(returns=col, benchmark=benchmark, name=name or str(col.name or "strategy"))

    # Multi-column: combine via weights.
    if weights is None:
        n = df.shape[1]
        w = pd.Series(1.0 / n, index=df.columns)
    elif isinstance(weights, pd.Series):
        w = weights.reindex(df.columns).fillna(0.0)
    else:
        w = pd.Series(weights).reindex(df.columns).fillna(0.0)

    # Per-period weights frame (constant weights, replicated).
    weights_frame = pd.DataFrame([w.to_numpy()] * len(df), index=df.index, columns=df.columns)
    combined = (df * weights_frame).sum(axis=1).rename(name or "combined")
    return Portfolio(
        returns=combined, weights=weights_frame, benchmark=benchmark, name=name or "combined"
    )


def portfolios_per_column(
    df: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
) -> list[tuple[str, Portfolio]]:
    """Return ``[(column_name, Portfolio), ...]`` — one independent portfolio
    per column, dropping leading NaN (common for assets that list later than
    the panel's start date).

    Used by the accessor render_* methods when the caller passes a
    multi-column DataFrame without explicit ``weights=`` — per-asset
    rendering in that case beats implicit equal-weight combining.
    """
    from fundcloud.portfolio import Portfolio

    out: list[tuple[str, Portfolio]] = []
    for col in df.columns:
        series = df[col].dropna().rename(str(col))
        out.append((str(col), Portfolio(returns=series, benchmark=benchmark, name=str(col))))
    return out


def is_bars_frame(df: pd.DataFrame) -> bool:
    """Return True when ``df`` looks like an OHLCV ``Bars`` frame.

    Heuristic: MultiIndex columns with top-level names drawn from
    ``{open, high, low, close, volume}``. Used by accessor methods that
    need to dispatch to :class:`Simulator`.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return False
    top_levels = set(map(str, df.columns.get_level_values(0).unique()))
    return bool(top_levels & {"open", "high", "low", "close", "volume"})


def require_bars_frame(df: pd.DataFrame, *, operation: str) -> None:
    """Raise a clear error when the caller expected a Bars frame."""
    if not is_bars_frame(df):
        msg = (
            f"{operation} requires a Bars frame with MultiIndex columns "
            f"(field, symbol) and OHLCV-style fields. "
            f"Got columns of type {type(df.columns).__name__}."
        )
        raise TypeError(msg)


def as_sim_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """Extract simulator-init kwargs from a loose kwargs mapping."""
    keys = {"costs", "slippage", "execution", "cash"}
    return {k: kw[k] for k in keys if k in kw}
