"""HTML tear-sheet renderer.

The resulting file is self-contained: plotly figures embed their required JS
inline (no CDN dependency) so a user can email the file or commit it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fundcloud.metrics.periods import period_returns as _period_returns
from fundcloud.metrics.periods import yearly_returns as _yearly_returns
from fundcloud.metrics.summary import drawdown_details as _drawdown_details
from fundcloud.metrics.summary import metrics as _metrics_bundle
from fundcloud.metrics.summary import runup_details as _runup_details
from fundcloud.plots import plotly as _plt
from fundcloud.reports import formatting as _fmt
from fundcloud.reports._benchmark_plots import rolling_alpha_beta_figure

if TYPE_CHECKING:
    from fundcloud.reports.tearsheet import Tearsheet

__all__ = ["render"]


_TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render(ts: Tearsheet, *, path: str | Path | None = None) -> str | Path:
    r = ts.portfolio.returns
    benchmark = ts.benchmark
    env = _env()
    template = env.get_template(f"{ts.template}.html.j2")

    # Figures
    figs = {
        "cumulative": _plt.cumulative(r, benchmark=benchmark),
        "drawdown": _plt.drawdown(r),
        "rolling_sharpe": _plt.rolling_sharpe(r),
        "distribution": _plt.return_distribution(r),
    }
    monthly_html: str | None = None
    if _has_span_of_months(r):
        figs["monthly_heatmap"] = _plt.monthly_heatmap(r)
        monthly_html = _fig_to_html(figs["monthly_heatmap"], include_js=False)
    rolling_ab_html: str | None = None
    if benchmark is not None:
        rolling_ab_html = _fig_to_html(rolling_alpha_beta_figure(r, benchmark), include_js=False)

    cumulative_html = _fig_to_html(figs["cumulative"], include_js=True)
    drawdown_html = _fig_to_html(figs["drawdown"], include_js=False)
    rolling_html = _fig_to_html(figs["rolling_sharpe"], include_js=False)
    distribution_html = _fig_to_html(figs["distribution"], include_js=False)

    # Period + yearly + drawdowns/runups — new performance sections.
    period_table_html: str | None = None
    yearly_table_html: str | None = None
    yearly_bars_html: str | None = None
    if _has_span_of_months(r):
        period_table_html = _style_table(
            _period_returns(r, benchmark=benchmark).to_frame()
            if isinstance(_period_returns(r, benchmark=benchmark), pd.Series)
            else _period_returns(r, benchmark=benchmark),
            pct_cols="all",
        )
        yearly_frame = _yearly_returns_frame(r, benchmark)
        yearly_table_html = _style_table(yearly_frame, pct_cols="all")
        yearly_bars_html = _fig_to_html(
            _plt.yearly_returns_bars(r, benchmark=benchmark), include_js=False
        )

    worst_dd_html = _style_table(
        _drawdowns_view(r),
        pct_cols=["Drawdown"],
        date_cols=["Started", "Recovered"],
        int_cols=["Days"],
    )
    worst_runup_html = _style_table(
        _runups_view(r),
        pct_cols=["Runup"],
        date_cols=["Started", "Peaked"],
        int_cols=["Days"],
    )

    # Stats: metrics() is the superset, feeds both the top cards (via the
    # compact summary view) and the categorised sidebar.
    stats = _metrics_bundle(r, benchmark=benchmark)
    bench_stats: pd.Series | None = None
    benchmark_label: str | None = None
    if benchmark is not None:
        bench_stats = _metrics_bundle(benchmark)
        benchmark_label = str(benchmark.name) if benchmark.name is not None else "benchmark"

    cards = _fmt.stat_cards(stats)
    sidebar = _fmt.categorized_sections(stats, bench_stats=bench_stats)

    html = template.render(
        title=ts.title or f"Tear sheet — {ts.portfolio.name}",
        period_start=r.index.min().strftime("%Y-%m-%d") if len(r) else "",
        period_end=r.index.max().strftime("%Y-%m-%d") if len(r) else "",
        n_periods=len(r),
        strategy_label=str(ts.portfolio.name),
        benchmark_label=benchmark_label,
        stat_cards=cards,
        sidebar_sections=sidebar,
        cumulative_html=cumulative_html,
        drawdown_html=drawdown_html,
        rolling_sharpe_html=rolling_html,
        rolling_alpha_beta_html=rolling_ab_html,
        distribution_html=distribution_html,
        monthly_heatmap_html=monthly_html,
        period_table_html=period_table_html,
        yearly_bars_html=yearly_bars_html,
        yearly_table_html=yearly_table_html,
        worst_drawdowns_html=worst_dd_html,
        worst_runups_html=worst_runup_html,
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
    span_days = (returns.index[-1] - returns.index[0]).days
    return span_days >= 60


def _drawdowns_view(r: pd.Series, *, top: int = 10) -> pd.DataFrame:
    dd = _drawdown_details(r)
    if dd.empty:
        return pd.DataFrame(columns=["Started", "Recovered", "Drawdown", "Days"])
    return (
        dd
        .head(top)[["start", "recovery", "max_drawdown", "duration_days"]]
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


def _runups_view(r: pd.Series, *, top: int = 10) -> pd.DataFrame:
    ru = _runup_details(r)
    if ru.empty:
        return pd.DataFrame(columns=["Started", "Peaked", "Runup", "Days"])
    return (
        ru
        .head(top)[["start", "peak", "max_runup", "duration_days"]]
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


def _yearly_returns_frame(r: pd.Series, benchmark: pd.Series | None) -> pd.DataFrame:
    strategy = _yearly_returns(r).rename(str(r.name) if r.name is not None else "Strategy")
    if benchmark is None:
        return strategy.to_frame()
    bench_name = str(benchmark.name) if benchmark.name is not None else "Benchmark"
    bench_yearly = _yearly_returns(benchmark).rename(bench_name)
    return pd.concat([bench_yearly, strategy], axis=1)


def _fmt_pct(v: object) -> str:
    if v is None or (isinstance(v, float) and (pd.isna(v))):
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


def _style_table(
    df: pd.DataFrame,
    *,
    pct_cols: list[str] | str | None = None,
    date_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
) -> str:
    """Render ``df`` as an HTML ``<table>`` using the shared tear-sheet CSS.

    ``pct_cols="all"`` formats every value as a percentage. Otherwise, pass
    the column names that should receive each formatter; omitted columns
    fall through to ``str``.
    """
    if df.empty:
        return ""
    pct_set = set(df.columns) if pct_cols == "all" else set(pct_cols or [])
    date_set = set(date_cols or [])
    int_set = set(int_cols or [])
    include_index = df.index.name is not None or not isinstance(df.index, pd.RangeIndex)
    index_label = df.index.name or ""
    thead = "".join(f"<th>{col}</th>" for col in df.columns)
    head = (
        f"<thead><tr><th>{index_label}</th>{thead}</tr></thead>"
        if include_index
        else f"<thead><tr>{thead}</tr></thead>"
    )
    rows_html: list[str] = []
    for idx, row in df.iterrows():
        cells = []
        if include_index:
            idx_str = _fmt_date(idx) if isinstance(idx, pd.Timestamp) else str(idx)
            cells.append(f"<td>{idx_str}</td>")
        for col in df.columns:
            val = row[col]
            if col in pct_set:
                cells.append(f"<td>{_fmt_pct(val)}</td>")
            elif col in date_set:
                cells.append(f"<td>{_fmt_date(val)}</td>")
            elif col in int_set:
                cells.append(f"<td>{_fmt_int(val)}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table>{head}<tbody>{''.join(rows_html)}</tbody></table>"
