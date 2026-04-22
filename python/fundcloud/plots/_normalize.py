"""Input-shape helpers for the plot builders."""

from __future__ import annotations

import pandas as pd

__all__ = ["to_series_list", "to_single_series"]


def to_series_list(
    obj: pd.Series | pd.DataFrame,
    *,
    default_name: str = "strategy",
) -> list[tuple[str, pd.Series]]:
    """Return ``[(label, series), ...]``.

    A :class:`pandas.Series` becomes a single-entry list labelled by its
    ``name`` (falling back to ``default_name``). A :class:`pandas.DataFrame`
    becomes one entry per column, labelled by the column name.
    """
    if isinstance(obj, pd.Series):
        return [(str(obj.name) if obj.name is not None else default_name, obj)]
    if isinstance(obj, pd.DataFrame):
        return [(str(col), obj[col]) for col in obj.columns]
    msg = f"expected pd.Series or pd.DataFrame, got {type(obj).__name__}"
    raise TypeError(msg)


def to_single_series(
    obj: pd.Series | pd.DataFrame,
    *,
    caller: str,
    default_name: str = "strategy",
) -> pd.Series:
    """Coerce a Series-or-single-column-DataFrame to a Series.

    A :class:`pandas.DataFrame` with more than one column raises, guiding the
    user toward a single-asset call. Used by builders that cannot be
    meaningfully overlayed (``monthly_heatmap``).
    """
    if isinstance(obj, pd.Series):
        if obj.name is None:
            return obj.rename(default_name)
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            col = obj.columns[0]
            return obj[col].rename(str(col))
        msg = (
            f"{caller} requires a single series; got a DataFrame with "
            f"{obj.shape[1]} columns. Pass one column (df['asset']) or use "
            f"plots.summary() for a multi-panel view."
        )
        raise ValueError(msg)
    msg = f"expected pd.Series or pd.DataFrame, got {type(obj).__name__}"
    raise TypeError(msg)
