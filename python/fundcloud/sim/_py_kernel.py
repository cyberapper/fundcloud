"""Pure-Python bar-iteration helpers used by :class:`Simulator`.

Two small building blocks: :func:`iterate_bars` yields ``(index, ts, row)``
triples across a Bars DataFrame, and :func:`prices_at` returns a
``asset -> price`` :class:`pandas.Series` for a single bar. The simulator
loop in :mod:`fundcloud.sim.simulator` consumes them directly; Rust-
accelerated paths live in :mod:`fundcloud.kernels`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

__all__ = ["iterate_bars", "prices_at"]


def iterate_bars(
    bars: pd.DataFrame,
) -> Iterator[tuple[int, pd.Timestamp, pd.Series]]:
    """Yield ``(index, timestamp, row)`` tuples per bar.

    ``row`` is the bar's :class:`pandas.Series` — for a MultiIndex frame its
    index is ``(field, asset)``, otherwise it's the column labels.
    """
    for i, (ts, row) in enumerate(bars.iterrows()):
        yield i, ts, row


def prices_at(
    bars: pd.DataFrame,
    idx: int,
    *,
    field: str = "close",
) -> pd.Series:
    """Return a ``Series`` of ``asset -> price`` for bar ``idx``."""
    if isinstance(bars.columns, pd.MultiIndex):
        row = bars.iloc[idx]
        fields = row.index.get_level_values(0)
        mask = fields == field
        sub = row[mask]
        # Drop the field level so the resulting Series is indexed by asset.
        sub.index = sub.index.droplevel(0)
        return sub.astype(float)
    return bars.iloc[idx].astype(float)
