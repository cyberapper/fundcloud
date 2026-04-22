"""Shared benchmark-resolution helper.

Accept ``pd.Series``, a column name from the surrounding returns DataFrame,
or ``None``. Used by every surface that takes ``benchmark=`` —
:func:`fundcloud.plots.summary`, :class:`fundcloud.reports.Tearsheet`, and
the ``.fc`` render methods.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["resolve_benchmark"]


def resolve_benchmark(
    returns: pd.Series | pd.DataFrame | None,
    benchmark: pd.Series | str | None,
) -> pd.Series | None:
    """Normalise ``benchmark`` into a :class:`pandas.Series` or ``None``.

    * ``None`` → ``None``.
    * :class:`pandas.Series` → returned unchanged (preserves name).
    * :class:`str` → looked up as a column of ``returns`` when ``returns``
      is a DataFrame. Unknown names raise :class:`ValueError` so typos
      surface early.

    Callers that render *per-asset* sections (multi-asset HTML/PDF/Excel)
    should drop the benchmark column from the iteration so it doesn't
    appear as its own strategy tab.
    """
    if benchmark is None or isinstance(benchmark, pd.Series):
        return benchmark
    if isinstance(benchmark, str):
        if isinstance(returns, pd.DataFrame) and benchmark in returns.columns:
            return returns[benchmark].rename(benchmark)
        available = list(returns.columns) if isinstance(returns, pd.DataFrame) else []
        msg = (
            f"benchmark={benchmark!r} is a string but no matching column "
            f"is available; pass pd.Series explicitly or include the "
            f"asset in the DataFrame. Columns: {available}"
        )
        raise ValueError(msg)
    msg = f"benchmark must be pd.Series | str | None, got {type(benchmark).__name__}"
    raise TypeError(msg)
