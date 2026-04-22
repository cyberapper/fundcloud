"""OHLCV (``Bars``) utilities — conversion, alignment, resampling.

These are free functions over plain pandas structures. They encode the
canonical data shapes (``Bars``: DatetimeIndex + top-level OHLCV column
labels, optionally a second-level per-asset index) that the rest of the
library relies on.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

__all__ = [
    "align",
    "as_long",
    "as_wide",
    "resample",
    "to_log_returns",
    "to_prices",
    "to_returns",
]

PriceField = Literal["open", "high", "low", "close", "adjusted_close"]


def _require_datetime_index(df: pd.DataFrame | pd.Series, name: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = f"{name} must have a DatetimeIndex, got {type(df.index).__name__}"
        raise TypeError(msg)


def to_prices(bars: pd.DataFrame, field: PriceField = "close") -> pd.DataFrame:
    """Extract a wide per-asset price panel from a ``Bars`` frame.

    Parameters
    ----------
    bars
        Either a wide frame whose columns *are* asset names (then returned as
        is, cast to float), or a frame with a two-level column index where the
        top level is the OHLCV field.
    field
        Which field to pull when ``bars`` has a MultiIndex on the columns.
    """
    _require_datetime_index(bars, "bars")

    if isinstance(bars.columns, pd.MultiIndex):
        if field not in bars.columns.get_level_values(0):
            msg = f"field '{field}' not found in columns {list(bars.columns.levels[0])}"
            raise KeyError(msg)
        prices = cast(pd.DataFrame, bars.xs(field, axis=1, level=0))
    else:
        prices = bars

    return cast(pd.DataFrame, prices.astype(float).sort_index())


def to_returns(
    prices_or_bars: pd.DataFrame | pd.Series,
    *,
    field: PriceField = "close",
    method: Literal["simple", "log"] = "simple",
    dropna: bool = True,
) -> pd.DataFrame | pd.Series:
    """Convert prices to period returns.

    Accepts either a wide price panel, a ``Bars`` DataFrame, or a single price
    ``Series``. Returns have the same shape and index as the input, minus the
    first row if ``dropna`` is True.
    """
    if isinstance(prices_or_bars, pd.Series):
        _require_datetime_index(prices_or_bars, "prices")
        prices_s = prices_or_bars.astype(float).sort_index()
        if method == "log":
            raw = np.log(prices_s / prices_s.shift(1))
        else:
            # Explicit ``fill_method=None`` prevents pandas' forward-fill default
            # (deprecated in pandas 2.1) from fabricating zero-returns on NaN
            # bars — essential when the panel mixes 5-day equities with 7-day
            # crypto, where NaN weekends must stay NaN, not become 0%.
            raw = prices_s.pct_change(fill_method=None)
        r = pd.Series(raw, index=prices_s.index, name=prices_s.name)
        return r.dropna() if dropna else r

    prices = to_prices(prices_or_bars, field=field)
    if method == "log":
        ret = np.log(prices / prices.shift(1))
    else:
        ret = prices.pct_change(fill_method=None)
    return ret.dropna(how="all") if dropna else ret


def to_log_returns(
    prices_or_bars: pd.DataFrame | pd.Series,
    *,
    field: PriceField = "close",
    dropna: bool = True,
) -> pd.DataFrame | pd.Series:
    """Convenience alias for ``to_returns(..., method='log')``."""
    return to_returns(prices_or_bars, field=field, method="log", dropna=dropna)


def align(*frames: pd.DataFrame, how: Literal["inner", "outer"] = "inner") -> list[pd.DataFrame]:
    """Align multiple wide frames onto the same index (and columns).

    Useful for combining prices + factors + signals before optimisation.
    """
    if not frames:
        return []
    aligned_index = frames[0].index
    aligned_cols = frames[0].columns
    op = aligned_index.intersection if how == "inner" else aligned_index.union
    col_op = aligned_cols.intersection if how == "inner" else aligned_cols.union
    for f in frames[1:]:
        aligned_index = op(f.index)
        aligned_cols = col_op(f.columns)
    return [f.reindex(index=aligned_index, columns=aligned_cols) for f in frames]


def resample(
    bars: pd.DataFrame,
    rule: str,
    *,
    agg: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resample a ``Bars`` frame to a coarser frequency.

    Defaults apply the standard OHLCV aggregation: first/max/min/last/sum for
    open/high/low/close/volume, and last for any other column.
    """
    _require_datetime_index(bars, "bars")
    ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    if isinstance(bars.columns, pd.MultiIndex):
        # MultiIndex (field, symbol): aggregate per column, keyed by level-0 field.
        if agg is None:
            per_col = {c: ohlcv_agg.get(str(c[0]).lower(), "last") for c in bars.columns}
        else:
            per_col = {c: agg.get(str(c[0]).lower(), "last") for c in bars.columns}
        return bars.resample(rule).agg(per_col).dropna(how="all")

    if agg is None:
        agg = {c: ohlcv_agg.get(str(c).lower(), "last") for c in bars.columns}
    return bars.resample(rule).agg(agg).dropna(how="all")  # type: ignore[arg-type]


def as_long(wide: pd.DataFrame, *, value_name: str = "value") -> pd.DataFrame:
    """Melt a wide (date × asset) frame to long (date, asset, value)."""
    _require_datetime_index(wide, "wide")
    out = wide.stack(future_stack=True).rename(value_name).reset_index()
    out.columns = ["ts", "asset", value_name]
    return out


def as_wide(
    long: pd.DataFrame,
    *,
    ts: str = "ts",
    asset: str = "asset",
    value: str = "value",
) -> pd.DataFrame:
    """Pivot a long (ts, asset, value) frame to wide (date × asset)."""
    wide = long.pivot(index=ts, columns=asset, values=value)
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns.name = None
    return cast(pd.DataFrame, wide.sort_index())
