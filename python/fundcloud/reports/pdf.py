"""PDF tear-sheet renderer.

Two backends:

* **matplotlib** (default) — pure-Python via :class:`matplotlib.backends.
  backend_pdf.PdfPages`. No system dependencies, works anywhere matplotlib
  installs. This is the one that ships.
* **weasyprint** (opt-in, ``engine="weasyprint"``) — HTML + CSS paged
  media for users who want full tear-sheet styling. Needs Pango / GLib /
  cairo system libraries (`brew install pango` on macOS, `apt install
  libpango-1.0-0 libpangoft2-1.0-0` on Debian).

Both backends accept the same :class:`~fundcloud.reports.tearsheet.Tearsheet`
input and produce a print-ready PDF. The matplotlib path is deliberately
lightweight — one stat-table page followed by one chart per page — so
traders and researchers can ship reports without fighting native-lib
install quirks.

Requires the ``fundcloud[viz]`` extra (matplotlib). The ``fundcloud[reports]``
extra pulls in WeasyPrint only if the WeasyPrint engine is used.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from fundcloud.reports import formatting as _fmt

if TYPE_CHECKING:
    from fundcloud.reports.tearsheet import Tearsheet

__all__ = ["render"]


Engine = Literal["matplotlib", "weasyprint"]
_DEFAULT_ENGINE: Engine = "matplotlib"


def render(ts: Tearsheet, *, path: Path, engine: Engine | None = None) -> Path:
    """Render a :class:`Tearsheet` to PDF.

    Parameters
    ----------
    ts
        The tear sheet to render.
    path
        Destination PDF path. Parent directories are created if missing.
    engine
        ``"matplotlib"`` (default) uses pure-Python matplotlib PdfPages.
        ``"weasyprint"`` uses HTML + CSS paged media (requires Pango system
        libraries).
    """
    engine = engine or _DEFAULT_ENGINE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine == "weasyprint":
        return _render_weasyprint(ts, path)
    if engine == "matplotlib":
        return _render_matplotlib(ts, path)
    msg = f"unknown PDF engine: {engine!r}. Use 'matplotlib' or 'weasyprint'."
    raise ValueError(msg)


# ============================================================= matplotlib path


# Every page in the matplotlib PDF is rendered at the same physical size so
# the printed document flows evenly page-to-page. A4 portrait 210x297 mm
# (8.268 x 11.693 in) is the industrial default for financial tear sheets
# outside the US.
_PAGE_SIZE = (8.268, 11.693)


def _render_matplotlib(ts: Tearsheet, path: Path) -> Path:
    _, plt = _require_mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    from fundcloud.metrics.periods import period_returns as _period_returns
    from fundcloud.metrics.periods import yearly_returns as _yearly_returns
    from fundcloud.metrics.summary import drawdown_details as _drawdown_details
    from fundcloud.metrics.summary import metrics as _metrics_bundle
    from fundcloud.metrics.summary import runup_details as _runup_details
    from fundcloud.plots import mpl as _mpl

    r = ts.portfolio.returns
    benchmark = ts.benchmark
    strategy_label = str(ts.portfolio.name)
    bench_label = (
        str(benchmark.name) if benchmark is not None and benchmark.name is not None else None
    )
    full_stats = _metrics_bundle(r, benchmark=benchmark)
    bench_stats = _metrics_bundle(benchmark) if benchmark is not None else None

    title = ts.title or f"Tear sheet — {ts.portfolio.name}"
    subtitle_bits: list[str] = []
    if len(r):
        subtitle_bits.append(f"{r.index.min().date()} → {r.index.max().date()}")
        subtitle_bits.append(f"{len(r)} periods")
    if bench_label:
        subtitle_bits.append(f"{strategy_label} vs {bench_label}")
    subtitle = "   ·   ".join(subtitle_bits)

    with PdfPages(str(path)) as pdf:
        # The categorised metrics table carries every figure the four stat
        # cards used to surface, so the dedicated card cover page is gone —
        # first metrics page just doubles as the title page via its suptitle
        # + subtitle line.
        for fig in _metric_table_pages(
            plt,
            full_stats,
            bench_stats=bench_stats,
            strategy_label=strategy_label,
            benchmark_label=bench_label,
            title=title,
            subtitle=subtitle,
        ):
            pdf.savefig(fig)
            plt.close(fig)

        # Pack two charts per page vertically — A4 portrait has plenty of
        # height; a single chart per page wasted half the page.
        chart_builders: list[tuple[str, Any]] = [
            ("Cumulative returns", lambda ax: _mpl._build_cumulative(ax, r, benchmark=benchmark)),
            ("Drawdown (%)", lambda ax: _mpl._build_drawdown(ax, r)),
            ("Rolling Sharpe", lambda ax: _mpl._build_rolling_sharpe(ax, r)),
            ("Return distribution (%)", lambda ax: _mpl._build_return_distribution(ax, r)),
        ]
        if benchmark is not None:
            from fundcloud.metrics import rolling_alpha as _rolling_alpha
            from fundcloud.metrics import rolling_beta as _rolling_beta

            chart_builders.append(
                (
                    "Rolling alpha (annualised)",
                    lambda ax: _draw_rolling_series(
                        ax,
                        _rolling_alpha(r, benchmark, window=63),
                        color="#2F6EE6",
                        reference=0.0,
                    ),
                )
            )
            chart_builders.append(
                (
                    "Rolling beta",
                    lambda ax: _draw_rolling_series(
                        ax,
                        _rolling_beta(r, benchmark, window=63),
                        color="#1F9B64",
                        reference=1.0,
                    ),
                )
            )
        if _has_span_of_months(r):
            chart_builders.append(("Monthly returns (%)", None))  # marker for heatmap
            chart_builders.append(
                (
                    _yearly_title(bench_label),
                    lambda ax: _build_yearly_bars(ax, r, benchmark=benchmark),
                )
            )

        for pair_start in range(0, len(chart_builders), 2):
            pair = chart_builders[pair_start : pair_start + 2]
            fig = _multi_chart_page(plt, pair, r=r)
            if fig is None:
                continue
            pdf.savefig(fig)
            plt.close(fig)

        # Performance tables: period returns, yearly returns, drawdowns, runups.
        if _has_span_of_months(r):
            period_df = _period_returns(r, benchmark=benchmark)
            if isinstance(period_df, pd.Series):
                period_df = period_df.to_frame()
            fig = _table_page(
                plt,
                title="Period performance",
                df=period_df,
                pct_cols="all",
                index_label="Period",
            )
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

            strategy_yearly = _yearly_returns(r).rename(strategy_label)
            if benchmark is not None:
                yearly_df = pd.concat(
                    [
                        _yearly_returns(benchmark).rename(bench_label or "Benchmark"),
                        strategy_yearly,
                    ],
                    axis=1,
                )
            else:
                yearly_df = strategy_yearly.to_frame()
            fig = _table_page(
                plt,
                title=_yearly_title(bench_label),
                df=yearly_df,
                pct_cols="all",
                index_label="Year",
            )
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

        dd_view = _drawdowns_df(_drawdown_details(r), top=10)
        if not dd_view.empty:
            fig = _table_page(
                plt,
                title="Worst 10 drawdowns",
                df=dd_view,
                pct_cols=["Drawdown"],
                date_cols=["Started", "Recovered"],
                int_cols=["Days"],
            )
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

        runup_view = _runups_df(_runup_details(r), top=10)
        if not runup_view.empty:
            fig = _table_page(
                plt,
                title="Top 10 runups",
                df=runup_view,
                pct_cols=["Runup"],
                date_cols=["Started", "Peaked"],
                int_cols=["Days"],
            )
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

        meta = pdf.infodict()
        meta["Title"] = title
        meta["Subject"] = "Fundcloud tear sheet"
        meta["Keywords"] = "fundcloud portfolio tearsheet"
        meta["CreationDate"] = datetime.now()

    return path


def _chart_page(plt: Any, title: str, build: Any) -> Any:
    """Build a single chart filling a whole portrait A4 page (legacy helper)."""
    try:
        fig = plt.figure(figsize=_PAGE_SIZE, dpi=120)
        ax = fig.add_axes((0.08, 0.09, 0.88, 0.82))
        fig.suptitle(title, x=0.08, ha="left", fontsize=14, fontweight="bold")
        build(ax)
        return fig
    except (TypeError, ValueError):
        plt.close("all")
        return None


def _multi_chart_page(plt: Any, pair: list[tuple[str, Any]], *, r: pd.Series) -> Any:
    """Stack up to two charts vertically on one A4 portrait page.

    Either slot can be the special ``("Monthly returns (%)", None)`` marker,
    which triggers an embedded heatmap + colorbar for that half of the page.
    """
    if not pair:
        return None
    try:
        from fundcloud.plots import mpl as _mpl

        fig = plt.figure(figsize=_PAGE_SIZE, dpi=120)
        # Two generous panels with room for axis labels and titles.
        slot_specs = [
            (0.09, 0.54, 0.86, 0.39),  # top slot
            (0.09, 0.07, 0.86, 0.39),  # bottom slot
        ]
        for (title, build), spec in zip(pair, slot_specs, strict=False):
            ax = fig.add_axes(spec)
            ax.set_title(title, loc="left", fontsize=12, fontweight="600", pad=6)
            if build is None:
                # Heatmap marker — draw directly so we also get the colorbar.
                im = _mpl._build_monthly_heatmap(ax, r, title="")
                fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="%")
            else:
                build(ax)
        return fig
    except (TypeError, ValueError):
        plt.close("all")
        return None


def _yearly_title(benchmark_label: str | None) -> str:
    if benchmark_label:
        return f"EOY returns vs {benchmark_label}"
    return "EOY returns"


def _build_yearly_bars(
    ax: Any, returns: pd.Series, *, benchmark: pd.Series | None
) -> None:
    """Paired grouped-bar chart: benchmark (if any) + strategy per year."""
    import matplotlib.ticker as _mtick

    from fundcloud.metrics.periods import yearly_returns as _yearly_returns

    strat = _yearly_returns(returns.dropna())
    bench = _yearly_returns(benchmark.dropna()) if benchmark is not None else None
    years = sorted(
        set(strat.index).union(set(bench.index)) if bench is not None else set(strat.index)
    )
    if not years:
        ax.text(0.5, 0.5, "No yearly data", ha="center", va="center")
        ax.axis("off")
        return
    import numpy as _np

    x = _np.arange(len(years))
    strat_vals = [float(strat.get(y, _np.nan)) for y in years]
    has_bench = bench is not None
    width = 0.38 if has_bench else 0.6
    if has_bench:
        bench_vals = [float(bench.get(y, _np.nan)) for y in years]
        ax.bar(
            x - width / 2,
            bench_vals,
            width=width,
            color="#F0C36D",
            label=str(benchmark.name) if benchmark is not None and benchmark.name is not None else "Benchmark",
        )
        ax.bar(x + width / 2, strat_vals, width=width, color="#2F6EE6", label="Strategy")
    else:
        ax.bar(x, strat_vals, width=width, color="#2F6EE6", label="Strategy")
    mean_ref = float(_np.nanmean(strat_vals)) if strat_vals else 0.0
    if _np.isfinite(mean_ref):
        ax.axhline(mean_ref, color="#C0392B", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="#444", linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=9)
    ax.yaxis.set_major_formatter(_mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(color="#E7E9EE", linestyle="-", linewidth=0.6, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _drawdowns_df(dd: pd.DataFrame, *, top: int) -> pd.DataFrame:
    if dd.empty:
        return dd
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


def _runups_df(ru: pd.DataFrame, *, top: int) -> pd.DataFrame:
    if ru.empty:
        return ru
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


def _fmt_pct(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{float(v) * 100:.2f}%"


def _fmt_date(v: object) -> str:
    if v is None or pd.isna(v):
        return "—"
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def _fmt_int(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{int(v):d}"


def _table_page(
    plt: Any,
    *,
    title: str,
    df: pd.DataFrame,
    pct_cols: list[str] | str | None = None,
    date_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    index_label: str | None = None,
) -> Any:
    """A4 portrait page carrying a single formatted table."""
    if df.empty:
        return None
    pct_set = set(df.columns) if pct_cols == "all" else set(pct_cols or [])
    date_set = set(date_cols or [])
    int_set = set(int_cols or [])
    include_index = index_label is not None or df.index.name is not None

    # Format cells.
    col_headers: list[str] = []
    if include_index:
        col_headers.append(index_label or str(df.index.name or ""))
    col_headers.extend(str(c) for c in df.columns)
    rows: list[list[str]] = []
    for idx, row in df.iterrows():
        cells: list[str] = []
        if include_index:
            cells.append(_fmt_date(idx) if isinstance(idx, pd.Timestamp) else str(idx))
        for col in df.columns:
            val = row[col]
            if col in pct_set:
                cells.append(_fmt_pct(val))
            elif col in date_set:
                cells.append(_fmt_date(val))
            elif col in int_set:
                cells.append(_fmt_int(val))
            else:
                cells.append(str(val) if not pd.isna(val) else "—")
        rows.append(cells)

    fig = plt.figure(figsize=_PAGE_SIZE, dpi=120)
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold")
    ax = fig.add_axes((0.06, 0.05, 0.88, 0.88))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="upper left",
        cellLoc="right",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    # Header styling.
    for col_idx in range(len(col_headers)):
        header_cell = table[(0, col_idx)]
        header_cell.set_facecolor("#F1F5F9")
        header_cell.set_text_props(fontweight="bold", color="#1c1c1c")
    # First column left-aligned.
    for row_idx in range(1, len(rows) + 1):
        cell = table[(row_idx, 0)]
        cell.set_text_props(ha="left")
    return fig


def _draw_rolling_series(
    ax: Any, series: pd.Series, *, color: str, reference: float
) -> None:
    """Helper for the rolling-α / rolling-β chart pages."""
    ax.plot(series.index, series.values, color=color, linewidth=1.4)
    ax.axhline(reference, color="#444", linewidth=0.6, linestyle=":")
    ax.grid(color="#E7E9EE", linestyle="-", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _rolling_alpha_beta_page(plt: Any, r: pd.Series, benchmark: pd.Series) -> Any:
    """Two-panel page: rolling alpha on top, rolling beta on the bottom."""
    try:
        from fundcloud.metrics import rolling_alpha, rolling_beta

        alpha = rolling_alpha(r, benchmark, window=63)
        beta = rolling_beta(r, benchmark, window=63)

        fig = plt.figure(figsize=_PAGE_SIZE, dpi=120)
        fig.suptitle(
            "Benchmark dynamics (rolling 63-bar)",
            x=0.08,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        ax_a = fig.add_axes((0.10, 0.54, 0.85, 0.36))
        ax_a.plot(alpha.index, alpha.values, color="#2F6EE6", linewidth=1.4)
        ax_a.axhline(0, color="#444", linewidth=0.6, linestyle=":")
        ax_a.set_title("Rolling alpha (annualised)", loc="left", fontsize=11, fontweight="600")
        ax_a.grid(color="#E7E9EE", linestyle="-", linewidth=0.6)
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)

        ax_b = fig.add_axes((0.10, 0.08, 0.85, 0.36))
        ax_b.plot(beta.index, beta.values, color="#1F9B64", linewidth=1.4)
        ax_b.axhline(1, color="#444", linewidth=0.6, linestyle=":")
        ax_b.set_title("Rolling beta", loc="left", fontsize=11, fontweight="600")
        ax_b.grid(color="#E7E9EE", linestyle="-", linewidth=0.6)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        return fig
    except (TypeError, ValueError):
        plt.close("all")
        return None


def _heatmap_page(plt: Any, r: pd.Series) -> Any:
    """Monthly heatmap on the same uniform landscape page as the other charts."""
    from fundcloud.plots import mpl as _mpl

    try:
        fig = plt.figure(figsize=_PAGE_SIZE, dpi=120)
        ax = fig.add_axes((0.08, 0.09, 0.78, 0.82))
        fig.suptitle("Monthly returns (%)", x=0.08, ha="left", fontsize=14, fontweight="bold")
        im = _mpl._build_monthly_heatmap(ax, r, title="")
        # Leave the auto-placed colorbar attached to the axes; explicit
        # shrink keeps it in proportion with the landscape page.
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="%")
        return fig
    except (TypeError, ValueError):
        plt.close("all")
        return None


def _page_header_and_stats(plt: Any, title: str, subtitle: str, stats: pd.Series) -> Any:
    """Legacy: header + cards + flat stats table. Kept for the WeasyPrint
    path which still renders a single summary image. The matplotlib path
    now uses :func:`_page_header_and_cards` + :func:`_metric_table_pages`.
    """
    fig = _page_header_and_cards(plt, title, subtitle, stats)
    ax_table = fig.add_axes((0.06, 0.06, 0.88, 0.58))
    ax_table.axis("off")
    rows = _fmt.stats_rows(stats)
    _draw_table(ax_table, rows)
    return fig


def _page_header_and_cards(plt: Any, title: str, subtitle: str, stats: pd.Series) -> Any:
    """Cover page: title, subtitle, four stat cards. No metrics table —
    the full categorised table renders on the pages that follow."""
    fig = plt.figure(figsize=_PAGE_SIZE)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.94)

    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold")
    if subtitle:
        fig.text(0.06, 0.92, subtitle, fontsize=10, color="#475569")
    fig.text(
        0.94,
        0.95,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        fontsize=9,
        color="#94a3b8",
        ha="right",
    )

    ax_cards = fig.add_axes((0.06, 0.72, 0.88, 0.16))
    ax_cards.axis("off")
    cards = _fmt.stat_cards(stats)
    _draw_cards(ax_cards, cards)
    return fig


def _metric_table_pages(
    plt: Any,
    stats: pd.Series,
    *,
    bench_stats: pd.Series | None,
    strategy_label: str,
    benchmark_label: str | None,
    title: str | None = None,
    subtitle: str | None = None,
) -> list[Any]:
    """Yield one or more A4 portrait pages carrying the full categorised
    metrics table, with a Strategy / Benchmark column pair when a benchmark
    is configured. The table flows across pages when a single A4 can't
    hold every row comfortably.

    The first page doubles as the document title page: ``title`` becomes the
    suptitle and ``subtitle`` (dates · periods · benchmark pairing) renders
    just below in a muted colour.
    """
    from fundcloud.reports.metric_info import METRIC_INFO, category_order

    # Build the flat (category, key, label, strategy_value, bench_value)
    # stream first, so pagination can just chunk rows.
    hidden = {"periods", "start", "end"}
    display_rows: list[tuple[str, str, str, str]] = []
    for cat in category_order():
        rows_in_cat = [
            (str(k), v)
            for k, v in stats.items()
            if str(k) not in hidden
            and not isinstance(v, pd.Timestamp)
            and (info := METRIC_INFO.get(str(k))) is not None
            and info.category == cat
        ]
        if not rows_in_cat:
            continue
        display_rows.append(("__section__", cat.value, "", ""))
        for key, val in rows_in_cat:
            info = METRIC_INFO.get(key)
            label = info.label if info is not None else key
            s_str = _fmt.format_stat(key, float(val) if pd.notna(val) else float("nan"))
            b_str = ""
            if bench_stats is not None:
                b_val = bench_stats.get(key, float("nan"))
                if not isinstance(b_val, pd.Timestamp):
                    b_str = _fmt.format_stat(
                        key, float(b_val) if pd.notna(b_val) else float("nan")
                    )
            display_rows.append((label, s_str, b_str, key))

    if not display_rows:
        return []

    # ~42 rows fit comfortably on an A4 portrait page with font size 9 +
    # generous padding. Chunking preserves the "section header" atomicity.
    rows_per_page = 42
    figs: list[Any] = []
    chunks = _chunk_with_section_headers(display_rows, rows_per_page)
    for page_index, chunk in enumerate(chunks):
        fig = plt.figure(figsize=_PAGE_SIZE)
        suffix = (
            f"  (page {page_index + 1} of {len(chunks)})" if len(chunks) > 1 else ""
        )
        page_title = title if (title and page_index == 0) else (
            f"Metrics — {strategy_label}"
            + (f" vs {benchmark_label}" if benchmark_label else "")
            + suffix
        )
        fig.suptitle(
            page_title,
            x=0.06,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        if page_index == 0 and subtitle:
            fig.text(0.06, 0.935, subtitle, fontsize=9, color="#475569")
            ax = fig.add_axes((0.06, 0.04, 0.88, 0.87))
        else:
            ax = fig.add_axes((0.06, 0.04, 0.88, 0.89))
        ax.axis("off")
        _draw_categorised_table(
            ax,
            chunk,
            strategy_label=strategy_label,
            benchmark_label=benchmark_label,
        )
        figs.append(fig)
    return figs


def _chunk_with_section_headers(
    rows: list[tuple[str, str, str, str]], per_page: int
) -> list[list[tuple[str, str, str, str]]]:
    """Split rows into pages of ~per_page, keeping a section header with
    at least one metric below it (avoids stranding "Calendar" on its own)."""
    pages: list[list[tuple[str, str, str, str]]] = []
    current: list[tuple[str, str, str, str]] = []
    for row in rows:
        current.append(row)
        if len(current) >= per_page and row[0] != "__section__":
            pages.append(current)
            current = []
    if current:
        pages.append(current)
    return pages


def _draw_categorised_table(
    ax: Any,
    rows: list[tuple[str, str, str, str]],
    *,
    strategy_label: str,
    benchmark_label: str | None,
) -> None:
    """Render a single page's worth of rows as a matplotlib table."""
    col_headers: list[str]
    if benchmark_label is None:
        col_headers = ["Metric", strategy_label]
        col_widths = [0.65, 0.35]
    else:
        col_headers = ["Metric", strategy_label, benchmark_label]
        col_widths = [0.50, 0.25, 0.25]

    cell_text: list[list[str]] = []
    cell_colours: list[list[str]] = []
    section_header_rows: set[int] = set()
    for idx, (label, strategy_val, bench_val, _key) in enumerate(rows):
        if label == "__section__":
            # Section header — use strategy_val to carry the category name.
            cell_text.append(
                [strategy_val] + [""] * (len(col_headers) - 1)
            )
            cell_colours.append(["#E0E7FF"] * len(col_headers))
            section_header_rows.add(idx)
            continue
        row_cells = [label, strategy_val]
        if benchmark_label is not None:
            row_cells.append(bench_val or "—")
        cell_text.append(row_cells)
        cell_colours.append(["#FFFFFF"] * len(col_headers))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        colWidths=col_widths,
        cellColours=cell_colours,
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row_idx == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(fontweight="bold", color="#1F2937")
        elif (row_idx - 1) in section_header_rows:
            cell.set_text_props(fontweight="bold", color="#1F2A44")
        elif col_idx > 0:
            # Right-align numeric columns for readability.
            cell.set_text_props(ha="right")


def _draw_cards(ax: Any, cards: list[Any]) -> None:
    if not cards:
        return
    n = len(cards)
    for i, card in enumerate(cards):
        x0 = i / n
        width = 1 / n - 0.01
        ax.text(
            x0 + width / 2,
            0.78,
            str(card.label).upper(),
            fontsize=9,
            color="#64748b",
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            x0 + width / 2,
            0.45,
            str(card.value),
            fontsize=18,
            fontweight="bold",
            color="#0f172a",
            ha="center",
            transform=ax.transAxes,
        )


def _draw_table(ax: Any, rows: list[Any]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No statistics available.", ha="center", va="center", color="#64748b")
        return
    cell_text = [[str(row.label), str(row.value)] for row in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        colWidths=[0.65, 0.35],
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row_idx == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(fontweight="bold", color="#334155")
        if col_idx == 1:
            cell.set_text_props(ha="right")


def _require_mpl() -> tuple[Any, Any]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for PDF rendering. Install with: uv add 'fundcloud[viz]'."
        raise ImportError(msg) from e
    mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    return mpl, plt


# ============================================================ weasyprint path


def _render_weasyprint(ts: Tearsheet, path: Path) -> Path:
    weasyprint = _require_weasyprint()
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from fundcloud.plots import mpl as _mpl

    r = ts.portfolio.returns
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(f"{ts.template}.html.j2")

    benchmark = ts.benchmark
    svgs = {
        "cumulative_html": _fig_to_svg(_mpl.cumulative(r, benchmark=benchmark)),
        "drawdown_html": _fig_to_svg(_mpl.drawdown(r)),
        "rolling_sharpe_html": _fig_to_svg(_mpl.rolling_sharpe(r)),
        "distribution_html": _fig_to_svg(_mpl.return_distribution(r)),
    }
    monthly: str | None = None
    if _has_span_of_months(r):
        monthly = _fig_to_svg(_mpl.monthly_heatmap(r))

    from fundcloud.metrics.summary import metrics as _metrics_bundle

    full = _metrics_bundle(r, benchmark=benchmark)
    bench_stats = _metrics_bundle(benchmark) if benchmark is not None else None
    bench_label = (
        str(benchmark.name) if benchmark is not None and benchmark.name is not None else None
    )
    html = template.render(
        title=ts.title or f"Tear sheet — {ts.portfolio.name}",
        period_start=r.index.min().strftime("%Y-%m-%d") if len(r) else "",
        period_end=r.index.max().strftime("%Y-%m-%d") if len(r) else "",
        n_periods=len(r),
        strategy_label=str(ts.portfolio.name),
        benchmark_label=bench_label,
        stat_cards=_fmt.stat_cards(full),
        sidebar_sections=_fmt.categorized_sections(full, bench_stats=bench_stats),
        cumulative_html=svgs["cumulative_html"],
        drawdown_html=svgs["drawdown_html"],
        rolling_sharpe_html=svgs["rolling_sharpe_html"],
        rolling_alpha_beta_html=None,  # weasyprint stays SVG-only for now
        distribution_html=svgs["distribution_html"],
        monthly_heatmap_html=monthly,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    weasyprint.HTML(string=html).write_pdf(str(path))
    return path


def _require_weasyprint() -> Any:
    try:
        import weasyprint
    except (ImportError, OSError) as e:
        msg = (
            "WeasyPrint engine unavailable. Install with "
            "`uv add 'fundcloud[reports]'` and the system Pango libraries "
            "(macOS: `brew install pango` then "
            "`export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`; Debian: "
            "`apt install libpango-1.0-0 libpangoft2-1.0-0`). "
            "Or use the default matplotlib engine, which has no system deps."
        )
        raise ImportError(msg) from e
    return weasyprint


def _fig_to_svg(fig: Any) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    svg = buf.getvalue()
    if "<?xml" in svg:
        svg = svg.split("?>", 1)[-1].lstrip()
    # Make SVGs responsive inside the CSS grid — avoids right-edge clipping
    # when plotly axis labels run wide.
    return svg.replace(
        "<svg ",
        '<svg style="max-width:100%;height:auto;" preserveAspectRatio="xMidYMid meet" ',
        1,
    )


# ============================================================ helpers


def _has_span_of_months(returns: Any) -> bool:
    if len(returns) < 10:
        return False
    return (returns.index[-1] - returns.index[0]).days >= 60
