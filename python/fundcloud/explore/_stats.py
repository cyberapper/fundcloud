"""Per-column descriptive statistics.

Produces the one-row-per-column table that both :func:`profile` and
:func:`compare` render as a scannable summary. Pure numpy/pandas — no
external EDA lib needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["overview", "per_column_stats"]


_NUMERIC_COLS = (
    "mean",
    "std",
    "min",
    "q25",
    "median",
    "q75",
    "max",
    "skew",
    "kurtosis",
    "zeros_pct",
    "inf_pct",
)


def per_column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-column descriptive frame indexed by column name.

    Numeric columns receive the full descriptive stack (quantiles, skew,
    kurtosis, zero and infinity fractions). Non-numeric columns get
    ``NaN`` in the numeric slots. ``mode_value`` / ``mode_freq`` are always
    populated so the alert layer can flag quasi-constant columns regardless
    of dtype.
    """
    rows: list[dict[str, Any]] = []
    total = len(df)
    for name in df.columns:
        col = df[name]
        non_null = col.dropna()
        row: dict[str, Any] = {
            "column": str(name),
            "dtype": str(col.dtype),
            "count": int(col.notna().sum()),
            "missing": int(col.isna().sum()),
            "missing_pct": float(col.isna().mean() * 100) if total else 0.0,
            "distinct": int(col.nunique(dropna=True)),
        }
        for k in _NUMERIC_COLS:
            row[k] = float("nan")

        if pd.api.types.is_numeric_dtype(col) and len(non_null):
            arr = non_null.to_numpy(dtype=float, copy=False)
            finite = arr[np.isfinite(arr)]
            inf_count = int(np.isinf(arr).sum())
            row["mean"] = float(arr.mean())
            row["std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            row["min"] = float(arr.min())
            row["q25"] = float(np.quantile(arr, 0.25))
            row["median"] = float(np.quantile(arr, 0.5))
            row["q75"] = float(np.quantile(arr, 0.75))
            row["max"] = float(arr.max())
            row["skew"] = float(pd.Series(finite).skew()) if len(finite) > 2 else float("nan")
            row["kurtosis"] = (
                float(pd.Series(finite).kurtosis()) if len(finite) > 3 else float("nan")
            )
            row["zeros_pct"] = float((arr == 0).mean() * 100)
            row["inf_pct"] = float(inf_count / total * 100) if total else 0.0

        # mode — works for any dtype; fuels the quasi-constant alert.
        if len(non_null):
            counts = non_null.value_counts()
            row["mode_value"] = counts.index[0]
            row["mode_freq"] = float(counts.iloc[0] / len(non_null) * 100)
        else:
            row["mode_value"] = None
            row["mode_freq"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def overview(df: pd.DataFrame) -> dict[str, Any]:
    """Top-of-report summary: shape, memory, missing, dupes, index hints."""
    dtype_tally = df.dtypes.astype(str).value_counts().to_dict()
    missing_total = int(df.isna().sum().sum())
    missing_cells = missing_total / max(df.size, 1) * 100
    info: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "cells": int(df.size),
        "missing_cells": missing_total,
        "missing_cells_pct": float(missing_cells),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "dtypes": {str(k): int(v) for k, v in dtype_tally.items()},
    }
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        info["index_type"] = "DatetimeIndex"
        info["date_start"] = str(df.index.min().date())
        info["date_end"] = str(df.index.max().date())
    else:
        info["index_type"] = type(df.index).__name__
    return info
