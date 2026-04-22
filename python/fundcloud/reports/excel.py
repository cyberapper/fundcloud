"""Excel tear-sheet renderer via XlsxWriter.

Numbers stay editable: we emit a workbook with up to four sheets (Summary,
Returns, Weights, Benchmark) and add native XlsxWriter charts rather than
images, so analysts can tweak cells and the charts recompute.

The Summary sheet mirrors the HTML sidebar — full metric bundle grouped by
category, with a Benchmark column when a benchmark is configured.

Requires ``fundcloud[reports]`` for xlsxwriter.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fundcloud.metrics import core as _metrics_core
from fundcloud.metrics.periods import period_returns as _period_returns
from fundcloud.metrics.periods import yearly_returns as _yearly_returns
from fundcloud.metrics.summary import drawdown_details as _drawdown_details
from fundcloud.metrics.summary import metrics as _metrics_bundle
from fundcloud.metrics.summary import runup_details as _runup_details
from fundcloud.reports.metric_info import METRIC_INFO, Category, category_order

if TYPE_CHECKING:
    from fundcloud.reports.tearsheet import Tearsheet

__all__ = ["render"]


def _require_xlsxwriter() -> Any:
    try:
        import xlsxwriter  # noqa: F401
    except ImportError as e:  # pragma: no cover — exercised without the extra
        msg = (
            "xlsxwriter is required for Excel rendering. Install with: uv add 'fundcloud[reports]'."
        )
        raise ImportError(msg) from e


def render(ts: Tearsheet, *, path: Path) -> Path:
    _require_xlsxwriter()
    path.parent.mkdir(parents=True, exist_ok=True)
    r = ts.portfolio.returns
    benchmark = ts.benchmark
    weights = ts.portfolio.weights

    returns_df = _build_returns_frame(r, benchmark=benchmark)

    with pd.ExcelWriter(
        path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        strategy_label = str(ts.portfolio.name)
        bench_label = (
            str(benchmark.name) if benchmark is not None and benchmark.name is not None else None
        )
        full_stats = _metrics_bundle(r, benchmark=benchmark)
        bench_stats = _metrics_bundle(benchmark) if benchmark is not None else None
        _write_summary(
            writer,
            ts=ts,
            stats=full_stats,
            bench_stats=bench_stats,
            strategy_label=strategy_label,
            benchmark_label=bench_label,
        )
        _write_period_returns(writer, r, benchmark=benchmark)
        _write_yearly_returns(writer, r, benchmark=benchmark, strategy_label=strategy_label)
        _write_episode_sheet(
            writer,
            sheet_name="Drawdowns",
            df=_drawdowns_view(r, top=10),
            pct_col="Drawdown",
        )
        _write_episode_sheet(
            writer,
            sheet_name="Runups",
            df=_runups_view(r, top=10),
            pct_col="Runup",
        )
        returns_df.to_excel(writer, sheet_name="Returns", index_label="date")
        _add_returns_charts(writer, returns_df, with_benchmark=benchmark is not None)
        if weights is not None and not weights.empty:
            weights.to_excel(writer, sheet_name="Weights", index_label="date")
    return path


def _build_returns_frame(r: pd.Series, *, benchmark: pd.Series | None) -> pd.DataFrame:
    """Return | Cumulative return | Drawdown %, plus matching benchmark columns."""
    cumulative = (1.0 + r.fillna(0.0)).cumprod() - 1.0
    dd = _metrics_core.drawdown_series(r) * 100.0
    cols: dict[str, pd.Series] = {
        "return": r,
        "cumulative_return": cumulative,
        "drawdown_pct": dd,
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index)
        b_cum = (1.0 + b.fillna(0.0)).cumprod() - 1.0
        b_dd = _metrics_core.drawdown_series(b.dropna()).reindex(r.index) * 100.0
        cols["benchmark_return"] = b
        cols["benchmark_cumulative_return"] = b_cum
        cols["benchmark_drawdown_pct"] = b_dd
    return pd.DataFrame(cols)


# --------------------------------------------------------------------- summary


def _write_summary(
    writer: Any,
    *,
    ts: Tearsheet,
    stats: pd.Series,
    bench_stats: pd.Series | None,
    strategy_label: str,
    benchmark_label: str | None,
) -> None:
    """Categorised summary sheet — mirrors the HTML sidebar layout."""
    wb = writer.book
    ws = wb.add_worksheet("Summary")
    writer.sheets["Summary"] = ws

    title_fmt = wb.add_format({"bold": True, "font_size": 16})
    subtitle_fmt = wb.add_format({"italic": True, "font_color": "#6B7280"})
    header_fmt = wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
    section_fmt = wb.add_format(
        {
            "bold": True,
            "bg_color": "#E0E7FF",
            "border": 1,
            "font_size": 11,
            "font_color": "#1F2A44",
        }
    )
    key_fmt = wb.add_format({"bold": True, "border": 1})
    pct_fmt = wb.add_format({"num_format": "0.00%", "border": 1})
    num_fmt = wb.add_format({"num_format": "0.00", "border": 1})
    int_fmt = wb.add_format({"num_format": "0", "border": 1})

    r_series = ts.portfolio.returns
    title = ts.title or f"Tear sheet — {ts.portfolio.name}"
    ws.write(0, 0, title, title_fmt)
    if len(r_series):
        subtitle = (
            f"{r_series.index.min():%Y-%m-%d} → {r_series.index.max():%Y-%m-%d}  ·  "
            f"{len(r_series)} periods  ·  {strategy_label}"
            + (f" vs {benchmark_label}" if benchmark_label else "")
        )
        ws.write(1, 0, subtitle, subtitle_fmt)

    row = 3
    ws.write(row, 0, "Metric", header_fmt)
    ws.write(row, 1, strategy_label, header_fmt)
    if benchmark_label is not None:
        ws.write(row, 2, benchmark_label, header_fmt)
    row += 1

    hidden = {"periods", "start", "end"}
    for cat in category_order():
        cat_rows = _rows_for_category(stats, cat, hidden)
        if not cat_rows:
            continue
        # Section header row — spans two or three columns.
        span_cols = 2 if benchmark_label is None else 3
        ws.merge_range(row, 0, row, span_cols - 1, cat.value, section_fmt)
        row += 1
        for key, val in cat_rows:
            info = METRIC_INFO.get(key)
            label = info.label if info is not None else key
            ws.write(row, 0, label, key_fmt)
            _write_metric_value(ws, row, 1, key, val, pct_fmt, num_fmt, int_fmt, key_fmt)
            if bench_stats is not None and benchmark_label is not None:
                b_val = bench_stats.get(key, float("nan"))
                _write_metric_value(
                    ws, row, 2, key, b_val, pct_fmt, num_fmt, int_fmt, key_fmt
                )
            row += 1

    ws.set_column(0, 0, 30)
    ws.set_column(1, 2, 16)


def _rows_for_category(
    stats: pd.Series, category: Category, hidden: set[str]
) -> list[tuple[str, Any]]:
    """Extract (key, value) pairs belonging to ``category`` from a metrics Series."""
    out: list[tuple[str, Any]] = []
    for key, val in stats.items():
        key_s = str(key)
        if key_s in hidden:
            continue
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            continue
        info = METRIC_INFO.get(key_s)
        if info is None or info.category != category:
            continue
        out.append((key_s, val))
    return out


def _write_metric_value(
    ws: Any,
    row: int,
    col: int,
    key: str,
    val: Any,
    pct_fmt: Any,
    num_fmt: Any,
    int_fmt: Any,
    key_fmt: Any,
) -> None:
    """Write one cell, picking the number format from the metric registry."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        ws.write(row, col, "—", key_fmt)
        return
    info = METRIC_INFO.get(key)
    fmt: Any
    if info is not None and info.fmt in {"pct", "pct4"}:
        fmt = pct_fmt
    elif info is not None and info.fmt == "int":
        fmt = int_fmt
    elif _is_percent(key):
        fmt = pct_fmt
    else:
        fmt = num_fmt
    if fmt is int_fmt:
        ws.write(row, col, int(val), fmt)
    else:
        ws.write(row, col, float(val), fmt)


# --------------------------------------------------------------------- returns charts


def _add_returns_charts(
    writer: Any, df: pd.DataFrame, *, with_benchmark: bool = False
) -> None:
    wb = writer.book
    ws = writer.sheets["Returns"]
    n = len(df)
    if n == 0:
        return

    # Columns: 0=date, 1=return, 2=cumulative_return, 3=drawdown_pct,
    # [4=benchmark_return, 5=benchmark_cumulative_return, 6=benchmark_drawdown_pct]
    cum_chart = wb.add_chart({"type": "line"})
    cum_chart.add_series({
        "name": "Strategy",
        "categories": ["Returns", 1, 0, n, 0],
        "values": ["Returns", 1, 2, n, 2],
        "line": {"color": "#2F6EE6", "width": 1.5},
    })
    if with_benchmark:
        cum_chart.add_series({
            "name": "Benchmark",
            "categories": ["Returns", 1, 0, n, 0],
            "values": ["Returns", 1, 5, n, 5],
            "line": {"color": "#888888", "width": 1.2, "dash_type": "dash"},
        })
    cum_chart.set_title({"name": "Cumulative return"})
    cum_chart.set_legend({"position": "bottom"} if with_benchmark else {"none": True})
    cum_chart.set_y_axis({"num_format": "0.00%"})
    ws.insert_chart("I2", cum_chart, {"x_scale": 1.3, "y_scale": 1.0})

    # Format percent columns so cell display matches the chart y-axis.
    pct_col_fmt = wb.add_format({"num_format": "0.00%"})
    ws.set_column(2, 2, 18, pct_col_fmt)
    if with_benchmark:
        ws.set_column(5, 5, 18, pct_col_fmt)

    dd_chart = wb.add_chart({"type": "line"})
    dd_chart.add_series({
        "name": "Strategy",
        "categories": ["Returns", 1, 0, n, 0],
        "values": ["Returns", 1, 3, n, 3],
        "line": {"color": "#C0392B", "width": 1.2},
    })
    if with_benchmark:
        dd_chart.add_series({
            "name": "Benchmark",
            "categories": ["Returns", 1, 0, n, 0],
            "values": ["Returns", 1, 6, n, 6],
            "line": {"color": "#888888", "width": 1.2, "dash_type": "dash"},
        })
    dd_chart.set_title({"name": "Drawdown (%)"})
    dd_chart.set_legend({"position": "bottom"} if with_benchmark else {"none": True})
    ws.insert_chart("I20", dd_chart, {"x_scale": 1.3, "y_scale": 1.0})


# --------------------------------------------------------------------- helpers


# --------------------------------------------------------------------- new sheets


def _write_period_returns(
    writer: Any, r: pd.Series, *, benchmark: pd.Series | None
) -> None:
    """MTD / 3M / … / All-time sheet, percentage-formatted."""
    df = _period_returns(r, benchmark=benchmark)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    sheet = "Period Returns"
    df.to_excel(writer, sheet_name=sheet, index_label="Period")
    wb = writer.book
    ws = writer.sheets[sheet]
    pct_fmt = wb.add_format({"num_format": "0.00%"})
    ws.set_column(0, 0, 18)
    ws.set_column(1, 1 + len(df.columns) - 1, 16, pct_fmt)


def _write_yearly_returns(
    writer: Any,
    r: pd.Series,
    *,
    benchmark: pd.Series | None,
    strategy_label: str,
) -> None:
    """End-of-year return sheet, percent-formatted, benchmark column first."""
    strat = _yearly_returns(r).rename(strategy_label or "Strategy")
    cols: list[pd.Series] = []
    if benchmark is not None:
        bench_label = (
            str(benchmark.name) if benchmark.name is not None else "Benchmark"
        )
        cols.append(_yearly_returns(benchmark).rename(bench_label))
    cols.append(strat)
    df = pd.concat(cols, axis=1)
    sheet = "Yearly Returns"
    df.to_excel(writer, sheet_name=sheet, index_label="Year")
    wb = writer.book
    ws = writer.sheets[sheet]
    pct_fmt = wb.add_format({"num_format": "0.00%"})
    ws.set_column(0, 0, 10)
    ws.set_column(1, 1 + len(df.columns) - 1, 16, pct_fmt)


def _write_episode_sheet(
    writer: Any, *, sheet_name: str, df: pd.DataFrame, pct_col: str
) -> None:
    """Worst-drawdowns / top-runups sheet with date + pct + int formatting."""
    if df.empty:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        return
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    wb = writer.book
    ws = writer.sheets[sheet_name]
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})
    pct_fmt = wb.add_format({"num_format": "0.00%"})
    int_fmt = wb.add_format({"num_format": "0"})
    for col_idx, col in enumerate(df.columns):
        if col == pct_col:
            ws.set_column(col_idx, col_idx, 14, pct_fmt)
        elif col == "Days":
            ws.set_column(col_idx, col_idx, 10, int_fmt)
        else:
            ws.set_column(col_idx, col_idx, 14, date_fmt)


def _drawdowns_view(r: pd.Series, *, top: int) -> pd.DataFrame:
    dd = _drawdown_details(r)
    if dd.empty:
        return pd.DataFrame(columns=["Started", "Recovered", "Drawdown", "Days"])
    return (
        dd.head(top)[["start", "recovery", "max_drawdown", "duration_days"]]
        .rename(
            columns={
                "start": "Started",
                "recovery": "Recovered",
                "max_drawdown": "Drawdown",
                "duration_days": "Days",
            }
        )
        .reset_index(drop=True)
    )


def _runups_view(r: pd.Series, *, top: int) -> pd.DataFrame:
    ru = _runup_details(r)
    if ru.empty:
        return pd.DataFrame(columns=["Started", "Peaked", "Runup", "Days"])
    return (
        ru.head(top)[["start", "peak", "max_runup", "duration_days"]]
        .rename(
            columns={
                "start": "Started",
                "peak": "Peaked",
                "max_runup": "Runup",
                "duration_days": "Days",
            }
        )
        .reset_index(drop=True)
    )


_PERCENT_KEYS = {
    "total_return",
    "cagr",
    "ann_volatility",
    "downside_volatility",
    "max_drawdown",
    "cvar",
    "value_at_risk",
    "ulcer_index",
    "pain_index",
    "best",
    "worst",
    "avg_return",
    "avg_win",
    "avg_loss",
    "tracking_error",
    "alpha",
    "best_month",
    "worst_month",
    "best_year",
    "worst_year",
}


def _is_percent(name: str) -> bool:
    return name in _PERCENT_KEYS
