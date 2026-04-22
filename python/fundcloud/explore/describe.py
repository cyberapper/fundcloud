"""``describe()`` — a super-set of :meth:`pandas.DataFrame.describe`.

Combines pandas' canonical descriptive rows (count / mean / std / min /
25% / 50% / 75% / max) with Fundcloud extras (dtype, missing, unique,
skew, kurtosis, zeros_pct, inf_pct) and, when the index is a
:class:`~pandas.DatetimeIndex`, finance-specific fields (Sharpe, CAGR,
max_drawdown, volatility) drawn from :mod:`fundcloud.metrics`.

Returns a one-row-per-column :class:`~pandas.DataFrame`. Optional HTML
output for the (rare) times you want to drop the table into a browser.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from fundcloud.explore import describe
>>> idx = pd.date_range("2024-01-02", periods=252, freq="B")
>>> df = pd.DataFrame({"a": np.linspace(-0.01, 0.01, 252),
...                    "b": np.full(252, 0.0002)}, index=idx)
>>> out = describe(df)
>>> "count" in out.columns and "sharpe" in out.columns
True
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["describe"]


_PANDAS_DEFAULT_PERCENTILES: tuple[float, ...] = (0.25, 0.5, 0.75)


def describe(
    df: pd.DataFrame | pd.Series,
    *,
    percentiles: Sequence[float] | None = None,
    include_finance: bool = True,
    output: str | Path | None = None,
    title: str | None = None,
) -> pd.DataFrame:
    """Return a one-row-per-column super-set of pandas ``describe``.

    Parameters
    ----------
    df
        Frame or series to summarise.
    percentiles
        Quantiles to include (matches ``DataFrame.describe(percentiles=)``).
        Defaults to ``(0.25, 0.5, 0.75)``.
    include_finance
        When ``True`` (default) and ``df`` has a ``DatetimeIndex`` with
        numeric columns, add ``sharpe``, ``cagr``, ``max_drawdown``,
        ``volatility`` rows (returns-interpretation).
    output
        Optional HTML output path. When ``None``, only the DataFrame is
        returned.
    title
        Heading text for the HTML output.

    Returns
    -------
    One row per input column, columns are the metric names. Matches pandas'
    column order for overlapping rows so the frame is a drop-in upgrade.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name or "series")

    pcts = tuple(percentiles) if percentiles is not None else _PANDAS_DEFAULT_PERCENTILES
    pct_labels = [f"{int(p * 100)}%" if (p * 100).is_integer() else f"{p * 100:.1f}%" for p in pcts]

    total = len(df)
    rows: list[dict[str, Any]] = []
    for name in df.columns:
        col = df[name]
        non_null = col.dropna()
        entry: dict[str, Any] = {"column": str(name)}

        # pandas-matching rows (only for numeric — matches DataFrame.describe default).
        if pd.api.types.is_numeric_dtype(col) and len(non_null):
            entry["count"] = int(col.notna().sum())
            entry["mean"] = float(non_null.mean())
            entry["std"] = float(non_null.std(ddof=1)) if len(non_null) > 1 else 0.0
            entry["min"] = float(non_null.min())
            for label, p in zip(pct_labels, pcts, strict=True):
                entry[label] = float(np.quantile(non_null.to_numpy(dtype=float), p))
            entry["max"] = float(non_null.max())
        else:
            for k in ("count", "mean", "std", "min", *pct_labels, "max"):
                entry[k] = float("nan") if k not in {"count"} else int(col.notna().sum())

        # Fundcloud extras — always populated.
        entry["dtype"] = str(col.dtype)
        entry["missing"] = int(col.isna().sum())
        entry["missing_pct"] = float(col.isna().mean() * 100) if total else 0.0
        entry["unique"] = int(col.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(col) and len(non_null):
            arr = non_null.to_numpy(dtype=float, copy=False)
            finite = arr[np.isfinite(arr)]
            entry["skew"] = float(pd.Series(finite).skew()) if len(finite) > 2 else float("nan")
            entry["kurtosis"] = (
                float(pd.Series(finite).kurtosis()) if len(finite) > 3 else float("nan")
            )
            entry["zeros_pct"] = float((arr == 0).mean() * 100)
            entry["inf_pct"] = (
                float(np.isinf(arr).sum() / total * 100) if total else 0.0
            )
        else:
            for k in ("skew", "kurtosis", "zeros_pct", "inf_pct"):
                entry[k] = float("nan")

        # Finance extras — only when the caller opts in *and* the frame
        # looks like a returns panel (DatetimeIndex + numeric).
        if (
            include_finance
            and isinstance(df.index, pd.DatetimeIndex)
            and pd.api.types.is_numeric_dtype(col)
            and len(non_null) >= 2
        ):
            # Local import to avoid circular dependency at module-import time.
            from fundcloud.metrics import cagr, max_drawdown, sharpe, volatility

            try:
                entry["sharpe"] = float(sharpe(col.dropna()))
            except (TypeError, ValueError, ZeroDivisionError):
                entry["sharpe"] = float("nan")
            try:
                entry["cagr"] = float(cagr(col.dropna()))
            except (TypeError, ValueError, ZeroDivisionError):
                entry["cagr"] = float("nan")
            try:
                entry["volatility"] = float(volatility(col.dropna()))
            except (TypeError, ValueError, ZeroDivisionError):
                entry["volatility"] = float("nan")
            try:
                entry["max_drawdown"] = float(max_drawdown(col.dropna()))
            except (TypeError, ValueError, ZeroDivisionError):
                entry["max_drawdown"] = float("nan")
        rows.append(entry)

    summary = pd.DataFrame(rows).set_index("column")

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        heading = title or f"describe: {total} rows by {df.shape[1]} columns"
        path.write_text(_render_html(summary, heading=heading), encoding="utf-8")
    return summary


def _render_html(summary: pd.DataFrame, *, heading: str) -> str:
    # Conservative, dependency-free HTML. Kept tiny because the canonical use
    # of describe() is the returned DataFrame at the REPL — the HTML exists
    # only as a convenience for sharing the table.
    return (
        "<!doctype html><html><head>"
        f"<meta charset='utf-8'><title>{heading}</title>"
        "<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        "padding:24px;color:#1c1c1c}table{border-collapse:collapse;width:100%}"
        "th,td{padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "thead th{background:#f8fafc;font-size:11px;text-transform:uppercase;"
        "letter-spacing:.4px;color:#6b7280}"
        "h1{font-size:22px;margin:0 0 16px}</style></head><body>"
        f"<h1>{heading}</h1>"
        + summary.to_html(float_format=lambda x: f"{x:.4g}")
        + "</body></html>"
    )
