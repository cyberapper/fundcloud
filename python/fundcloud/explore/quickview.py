"""Deprecated alias for :func:`fundcloud.explore.describe`.

Kept for one release. Emits :class:`DeprecationWarning` on call.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fundcloud.explore.describe import describe as _describe

__all__ = ["quickview"]


def quickview(
    df: pd.DataFrame | pd.Series,
    *,
    percentiles: Sequence[float] | None = None,
    include_finance: bool = True,
    output: str | Path | None = None,
    title: str | None = None,
) -> pd.DataFrame:
    """Deprecated. Use :func:`fundcloud.explore.describe` instead.

    ``describe()`` is a super-set of :meth:`pandas.DataFrame.describe` with
    the Fundcloud finance extras that ``quickview`` used to provide. This
    wrapper exists only to keep older code running for one release cycle.
    """
    warnings.warn(
        "fundcloud.explore.quickview is deprecated; use fundcloud.explore.describe instead. "
        "describe() is a super-set of pandas' describe with the same finance extras.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _describe(
        df,
        percentiles=percentiles,
        include_finance=include_finance,
        output=output,
        title=title,
    )
