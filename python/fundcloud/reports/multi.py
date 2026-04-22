"""Multi-asset tear-sheet rendering.

When a caller hands a multi-column ``pd.DataFrame`` to ``.fc.render_*`` and
does not supply ``weights=``, the expected behaviour is one full tear sheet
per column — tabs in HTML, per-asset sections in PDF, per-asset sheets in
Excel. This module builds those multi-format reports.

The single-asset path in :mod:`fundcloud.reports.html` / ``.pdf`` / ``.excel``
is still used for each column; the helpers here just orchestrate and glue.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fundcloud.metrics.summary import metrics as _metrics_bundle
from fundcloud.plots import plotly as _plt
from fundcloud.reports import formatting as _fmt
from fundcloud.reports._benchmark_plots import rolling_alpha_beta_figure

if TYPE_CHECKING:
    from fundcloud.portfolio import Portfolio

__all__ = ["render_excel", "render_html", "render_pdf"]


_TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )


# --------------------------------------------------------------------- HTML


def render_html(
    portfolios: list[tuple[str, Portfolio]],
    *,
    title: str | None = None,
    benchmark: pd.Series | None = None,
    path: str | Path | None = None,
) -> str | Path:
    """Render a multi-asset HTML tear sheet with CSS tabs for asset selection."""
    if not portfolios:
        msg = "render_html requires at least one (asset_name, Portfolio) pair"
        raise ValueError(msg)

    bench_label: str | None = None
    bench_stats: pd.Series | None = None
    if benchmark is not None:
        bench_label = str(benchmark.name) if benchmark.name is not None else "benchmark"
        bench_stats = _metrics_bundle(benchmark)

    sections: list[dict[str, Any]] = []
    for i, (asset_name, pf) in enumerate(portfolios):
        r = pf.returns
        section_title = asset_name
        stats = _metrics_bundle(r, benchmark=benchmark)
        figs = {
            "cumulative": _plt.cumulative(r, benchmark=benchmark),
            "drawdown": _plt.drawdown(r),
            "rolling_sharpe": _plt.rolling_sharpe(r),
            "distribution": _plt.return_distribution(r),
        }
        monthly_html: str | None = None
        if _has_span_of_months(r):
            monthly_html = _fig_to_html(_plt.monthly_heatmap(r), include_js=False)
        rolling_ab_html: str | None = None
        if benchmark is not None:
            rolling_ab_html = _fig_to_html(
                rolling_alpha_beta_figure(r, benchmark), include_js=False
            )

        sections.append({
            "asset": asset_name,
            "title": section_title,
            "period_start": r.index.min().strftime("%Y-%m-%d") if len(r) else "",
            "period_end": r.index.max().strftime("%Y-%m-%d") if len(r) else "",
            "n_periods": len(r),
            "strategy_label": asset_name,
            "benchmark_label": bench_label,
            "stat_cards": _fmt.stat_cards(stats),
            "sidebar_sections": _fmt.categorized_sections(stats, bench_stats=bench_stats),
            # plotly.js is inlined exactly once — on the very first figure
            # of the very first asset section. Every later figure reuses it.
            "cumulative_html": _fig_to_html(figs["cumulative"], include_js=(i == 0)),
            "drawdown_html": _fig_to_html(figs["drawdown"], include_js=False),
            "rolling_sharpe_html": _fig_to_html(figs["rolling_sharpe"], include_js=False),
            "rolling_alpha_beta_html": rolling_ab_html,
            "distribution_html": _fig_to_html(figs["distribution"], include_js=False),
            "monthly_heatmap_html": monthly_html,
        })

    # Report-level chrome: period span is the union across assets.
    period_start = min(
        (pf.returns.index.min() for _, pf in portfolios if len(pf.returns)), default=None
    )
    period_end = max(
        (pf.returns.index.max() for _, pf in portfolios if len(pf.returns)), default=None
    )

    template = _env().get_template("multi_strategy.html.j2")
    html = template.render(
        title=title or "Fundcloud tear sheet",
        n_assets=len(portfolios),
        period_start=period_start.strftime("%Y-%m-%d") if period_start is not None else "",
        period_end=period_end.strftime("%Y-%m-%d") if period_end is not None else "",
        n_periods=max((len(pf.returns) for _, pf in portfolios), default=0),
        sections=sections,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    if path is not None:
        out = Path(path)
        out.write_text(html, encoding="utf-8")
        return out
    return html


def _fig_to_html(fig: object, *, include_js: bool) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_js else False,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def _has_span_of_months(returns: pd.Series) -> bool:
    if len(returns) < 10:
        return False
    return (returns.index[-1] - returns.index[0]).days >= 60


# --------------------------------------------------------------------- PDF


def render_pdf(
    portfolios: list[tuple[str, Portfolio]],
    *,
    path: str | Path,
    title: str | None = None,
    benchmark: pd.Series | None = None,
    engine: Literal["matplotlib", "weasyprint"] | None = None,
) -> Path:
    """Render one PDF containing a per-asset tear-sheet section.

    Only the matplotlib engine is supported today (uniform landscape letter
    pages across every asset). WeasyPrint multi-asset support can follow.
    """
    if engine not in (None, "matplotlib"):
        msg = (
            "Multi-asset PDF currently supports engine='matplotlib' only. "
            "Pass a DataFrame with a single column or build a MultiTearsheet "
            "when WeasyPrint output is required."
        )
        raise NotImplementedError(msg)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_pdf import PdfPages

    from fundcloud.reports.pdf import (
        _PAGE_SIZE,
        _draw_rolling_series,
        _metric_table_pages,
        _multi_chart_page,
        _require_mpl,
    )

    _, plt = _require_mpl()

    bench_stats = _metrics_bundle(benchmark) if benchmark is not None else None
    bench_label = (
        str(benchmark.name) if benchmark is not None and benchmark.name is not None else None
    )

    with PdfPages(str(out)) as pdf:
        # Cover page — summary across every asset.
        pdf.savefig(_cover_page(plt, title or "Multi-asset tear sheet", portfolios, _PAGE_SIZE))
        plt.close("all")

        for asset_name, pf in portfolios:
            r = pf.returns
            section_title = f"{asset_name} — {title}" if title else asset_name
            subtitle_bits: list[str] = []
            if len(r):
                subtitle_bits.append(f"{r.index.min().date()} → {r.index.max().date()}")
                subtitle_bits.append(f"{len(r)} periods")
            if bench_label:
                subtitle_bits.append(f"{asset_name} vs {bench_label}")
            subtitle = "   ·   ".join(subtitle_bits)

            asset_full = _metrics_bundle(r, benchmark=benchmark)
            for fig in _metric_table_pages(
                plt,
                asset_full,
                bench_stats=bench_stats,
                strategy_label=asset_name,
                benchmark_label=bench_label,
                title=section_title,
                subtitle=subtitle,
            ):
                pdf.savefig(fig)
                plt.close(fig)

            pairs = list(_asset_chart_builders(r, benchmark=benchmark))
            if benchmark is not None:
                from fundcloud.metrics import rolling_alpha as _rolling_alpha
                from fundcloud.metrics import rolling_beta as _rolling_beta

                pairs.append((
                    "Rolling alpha (annualised)",
                    lambda ax, r=r: _draw_rolling_series(
                        ax,
                        _rolling_alpha(r, benchmark, window=63),
                        color="#2F6EE6",
                        reference=0.0,
                    ),
                ))
                pairs.append((
                    "Rolling beta",
                    lambda ax, r=r: _draw_rolling_series(
                        ax,
                        _rolling_beta(r, benchmark, window=63),
                        color="#1F9B64",
                        reference=1.0,
                    ),
                ))
            if _has_span_of_months(r):
                pairs.append(("Monthly returns (%)", None))

            for start in range(0, len(pairs), 2):
                pair = pairs[start : start + 2]
                fig = _multi_chart_page(plt, pair, r=r)
                if fig is None:
                    continue
                pdf.savefig(fig)
                plt.close(fig)
            plt.close("all")

        meta = pdf.infodict()
        meta["Title"] = title or "Fundcloud multi-asset tear sheet"
        meta["Subject"] = "Fundcloud multi-asset tear sheet"
        meta["Keywords"] = "fundcloud portfolio tearsheet multi-asset"
        meta["CreationDate"] = datetime.now()

    return out


def _asset_chart_builders(
    r: pd.Series, *, benchmark: pd.Series | None
) -> tuple[tuple[str, Any], ...]:
    from fundcloud.plots import mpl as _mpl

    return (
        ("Cumulative returns", lambda ax: _mpl._build_cumulative(ax, r, benchmark=benchmark)),
        ("Drawdown (%)", lambda ax: _mpl._build_drawdown(ax, r)),
        ("Rolling Sharpe", lambda ax: _mpl._build_rolling_sharpe(ax, r)),
        ("Return distribution (%)", lambda ax: _mpl._build_return_distribution(ax, r)),
    )


def _cover_page(
    plt: Any, title: str, portfolios: list[tuple[str, Portfolio]], size: tuple[float, float]
) -> Any:
    """Single cover page listing every asset with its core stats."""
    fig = plt.figure(figsize=size)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.94)
    fig.text(0.06, 0.95, title, fontsize=20, fontweight="bold")
    fig.text(
        0.06,
        0.92,
        f"{len(portfolios)} strategies · generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=10,
        color="#475569",
    )

    ax = fig.add_axes((0.06, 0.06, 0.88, 0.80))
    ax.axis("off")
    headers = ["Strategy", "Periods", "CAGR", "Ann vol", "Sharpe", "Max DD", "CVaR"]
    rows: list[list[str]] = []
    for asset_name, pf in portfolios:
        s = pf.summary()
        rows.append([
            asset_name,
            f"{int(s.get('periods', 0))}",
            _fmt_pct(s.get("cagr")),
            _fmt_pct(s.get("ann_volatility")),
            _fmt_num(s.get("sharpe")),
            _fmt_pct(s.get("max_drawdown")),
            _fmt_pct(s.get("cvar")),
        ])
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="right",
        colLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    # First column (strategy name) aligned left.
    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row_i == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(fontweight="bold", color="#334155")
        if col_i == 0:
            cell.set_text_props(ha="left")
    return fig


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100.0:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v: Any) -> str:
    try:
        return f"{float(v):+.2f}"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------- Excel


def render_excel(
    portfolios: list[tuple[str, Portfolio]],
    *,
    path: str | Path,
    title: str | None = None,
    benchmark: pd.Series | None = None,
) -> Path:
    """Render a workbook with an Overview sheet plus one Summary + Returns
    sheet pair per asset.

    Sheet names are prefixed with ``<asset>_`` (capped at 31 chars, the
    Excel limit).
    """
    try:
        import xlsxwriter  # noqa: F401
    except ImportError as e:
        msg = (
            "xlsxwriter is required for Excel rendering. Install with: uv add 'fundcloud[reports]'."
        )
        raise ImportError(msg) from e

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    overview = _build_overview_frame(portfolios)

    bench_stats = _metrics_bundle(benchmark) if benchmark is not None else None
    bench_label = (
        str(benchmark.name) if benchmark is not None and benchmark.name is not None else None
    )

    with pd.ExcelWriter(
        out,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        wb = writer.book
        _write_overview_sheet(writer, title, overview)
        for asset_name, pf in portfolios:
            _write_asset_sheets(
                writer,
                wb,
                asset_name,
                pf,
                benchmark=benchmark,
                bench_stats=bench_stats,
                bench_label=bench_label,
            )
    return out


def _build_overview_frame(portfolios: list[tuple[str, Portfolio]]) -> pd.DataFrame:
    """Compact metric-by-strategy table for the first sheet."""
    from fundcloud.metrics import core as _core

    rows: dict[str, dict[str, Any]] = {}
    for asset_name, pf in portfolios:
        r = pf.returns
        ppy = 252
        rows[asset_name] = {
            "periods": len(r),
            "cagr": float(_core.cagr(r, periods_per_year=ppy)) if len(r) else float("nan"),
            "ann_volatility": float(_core.volatility(r, periods_per_year=ppy))
            if len(r)
            else float("nan"),
            "sharpe": float(_core.sharpe(r, periods_per_year=ppy)) if len(r) else float("nan"),
            "sortino": float(_core.sortino(r, periods_per_year=ppy)) if len(r) else float("nan"),
            "max_drawdown": float(_core.max_drawdown(r)) if len(r) else float("nan"),
            "cvar": float(_core.cvar(r)) if len(r) else float("nan"),
        }
    return pd.DataFrame(rows).T


def _write_overview_sheet(writer: Any, title: str | None, overview: pd.DataFrame) -> None:
    wb = writer.book
    ws = wb.add_worksheet("Overview")
    writer.sheets["Overview"] = ws
    ws.write(
        0, 0, title or "Multi-asset tear sheet", wb.add_format({"bold": True, "font_size": 16})
    )
    ws.write(
        1,
        0,
        f"{len(overview)} strategies",
        wb.add_format({"italic": True, "font_color": "#6B7280"}),
    )
    header_fmt = wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
    cell_fmt = wb.add_format({"border": 1})
    pct_fmt = wb.add_format({"num_format": "0.00%", "border": 1})
    num_fmt = wb.add_format({"num_format": "0.00", "border": 1})

    headers = ["Strategy", *overview.columns]
    for col_i, h in enumerate(headers):
        ws.write(3, col_i, str(h), header_fmt)
    for row_i, (asset_name, row) in enumerate(overview.iterrows(), start=4):
        ws.write(row_i, 0, str(asset_name), cell_fmt)
        for col_i, col in enumerate(overview.columns, start=1):
            val = row[col]
            if pd.isna(val):
                ws.write(row_i, col_i, "—", cell_fmt)
            elif col in {"cagr", "ann_volatility", "max_drawdown", "cvar"}:
                ws.write(row_i, col_i, float(val), pct_fmt)
            elif col == "periods":
                ws.write(row_i, col_i, int(val), cell_fmt)
            else:
                ws.write(row_i, col_i, float(val), num_fmt)
    ws.set_column(0, 0, 24)
    ws.set_column(1, len(headers) - 1, 16)


def _write_asset_sheets(
    writer: Any,
    wb: Any,
    asset_name: str,
    pf: Portfolio,
    *,
    benchmark: pd.Series | None = None,
    bench_stats: pd.Series | None = None,
    bench_label: str | None = None,
) -> None:
    """Add a Summary + Returns sheet pair for a single asset.

    The Summary sheet uses the same categorised layout as the single-asset
    :mod:`fundcloud.reports.excel` path, with an optional benchmark column
    when the workbook is driven by a benchmarked multi-asset render.
    """
    from fundcloud.reports.excel import _build_returns_frame
    from fundcloud.reports.metric_info import METRIC_INFO, category_order

    r = pf.returns
    returns_df = _build_returns_frame(r, benchmark=benchmark)

    summary_sheet = _safe_sheet_name(asset_name, suffix="_Summary")
    returns_sheet = _safe_sheet_name(asset_name, suffix="_Returns")

    # Summary sheet — categorised, benchmark-aware.
    stats = _metrics_bundle(r, benchmark=benchmark)

    ws = wb.add_worksheet(summary_sheet)
    writer.sheets[summary_sheet] = ws
    ws.write(0, 0, asset_name, wb.add_format({"bold": True, "font_size": 16}))
    if len(r):
        subtitle = f"{r.index.min():%Y-%m-%d} → {r.index.max():%Y-%m-%d}  ·  {len(r)} periods"
        if bench_label:
            subtitle += f"  ·  {asset_name} vs {bench_label}"
        ws.write(1, 0, subtitle, wb.add_format({"italic": True, "font_color": "#6B7280"}))

    header_fmt = wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
    section_fmt = wb.add_format({
        "bold": True,
        "bg_color": "#E0E7FF",
        "border": 1,
        "font_size": 11,
        "font_color": "#1F2A44",
    })
    key_fmt = wb.add_format({"bold": True, "border": 1})
    pct_fmt = wb.add_format({"num_format": "0.00%", "border": 1})
    num_fmt = wb.add_format({"num_format": "0.00", "border": 1})
    int_fmt = wb.add_format({"num_format": "0", "border": 1})

    row = 3
    ws.write(row, 0, "Metric", header_fmt)
    ws.write(row, 1, asset_name, header_fmt)
    if bench_label is not None:
        ws.write(row, 2, bench_label, header_fmt)
    row += 1

    hidden = {"periods", "start", "end"}
    for cat in category_order():
        cat_keys = [
            (k, v)
            for k, v in stats.items()
            if str(k) not in hidden
            and not isinstance(v, (pd.Timestamp,))
            and (info := METRIC_INFO.get(str(k))) is not None
            and info.category == cat
        ]
        if not cat_keys:
            continue
        span = 2 if bench_label is None else 3
        ws.merge_range(row, 0, row, span - 1, cat.value, section_fmt)
        row += 1
        for key, val in cat_keys:
            info = METRIC_INFO.get(str(key))
            label = info.label if info is not None else str(key)
            ws.write(row, 0, label, key_fmt)
            _write_asset_metric_value(ws, row, 1, str(key), val, pct_fmt, num_fmt, int_fmt, key_fmt)
            if bench_stats is not None and bench_label is not None:
                b_val = bench_stats.get(str(key), float("nan"))
                _write_asset_metric_value(
                    ws, row, 2, str(key), b_val, pct_fmt, num_fmt, int_fmt, key_fmt
                )
            row += 1

    ws.set_column(0, 0, 30)
    ws.set_column(1, 2, 16)

    # Returns sheet + native charts. Benchmark columns ride along when
    # ``benchmark`` is set, so the charts can overlay strategy vs bench.
    returns_df.to_excel(writer, sheet_name=returns_sheet, index_label="date")
    _add_asset_charts(
        writer, wb, returns_sheet, len(returns_df), with_benchmark=benchmark is not None
    )


def _write_asset_metric_value(
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
    from fundcloud.reports.excel import _is_percent
    from fundcloud.reports.metric_info import METRIC_INFO

    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        ws.write(row, col, "—", key_fmt)
        return
    info = METRIC_INFO.get(key)
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


def _add_asset_charts(
    writer: Any, wb: Any, sheet_name: str, n: int, *, with_benchmark: bool = False
) -> None:
    if n == 0:
        return
    ws = writer.sheets[sheet_name]

    cum_chart = wb.add_chart({"type": "line"})
    cum_chart.add_series({
        "name": "Strategy",
        "categories": [sheet_name, 1, 0, n, 0],
        "values": [sheet_name, 1, 2, n, 2],
        "line": {"color": "#2F6EE6", "width": 1.5},
    })
    if with_benchmark:
        cum_chart.add_series({
            "name": "Benchmark",
            "categories": [sheet_name, 1, 0, n, 0],
            "values": [sheet_name, 1, 5, n, 5],
            "line": {"color": "#888888", "width": 1.2, "dash_type": "dash"},
        })
    cum_chart.set_title({"name": "Cumulative return"})
    cum_chart.set_legend({"position": "bottom"} if with_benchmark else {"none": True})
    cum_chart.set_y_axis({"num_format": "0.00%"})
    ws.insert_chart("I2", cum_chart, {"x_scale": 1.3, "y_scale": 1.0})
    pct_col_fmt = wb.add_format({"num_format": "0.00%"})
    ws.set_column(2, 2, 18, pct_col_fmt)
    if with_benchmark:
        ws.set_column(5, 5, 18, pct_col_fmt)

    dd_chart = wb.add_chart({"type": "line"})
    dd_chart.add_series({
        "name": "Strategy",
        "categories": [sheet_name, 1, 0, n, 0],
        "values": [sheet_name, 1, 3, n, 3],
        "line": {"color": "#C0392B", "width": 1.2},
    })
    if with_benchmark:
        dd_chart.add_series({
            "name": "Benchmark",
            "categories": [sheet_name, 1, 0, n, 0],
            "values": [sheet_name, 1, 6, n, 6],
            "line": {"color": "#888888", "width": 1.2, "dash_type": "dash"},
        })
    dd_chart.set_title({"name": "Drawdown (%)"})
    dd_chart.set_legend({"position": "bottom"} if with_benchmark else {"none": True})
    ws.insert_chart("I20", dd_chart, {"x_scale": 1.3, "y_scale": 1.0})


def _safe_sheet_name(asset: str, *, suffix: str = "") -> str:
    """Coerce to an Excel-safe 31-char sheet name (strip colons, slashes)."""
    cleaned = "".join(c for c in str(asset) if c not in set("[]:*?/\\"))
    max_asset_len = 31 - len(suffix)
    return f"{cleaned[:max_asset_len]}{suffix}"
