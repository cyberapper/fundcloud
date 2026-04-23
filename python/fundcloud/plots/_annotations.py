"""On-figure stats / reference lines driven by :mod:`fundcloud.metrics.core`.

Helpers here produce small strings or mutate an existing figure; they are
meant to stay backend-agnostic at the formatting level but the figure-
mutating helpers (``annotate_*``) target plotly. Matplotlib builders keep
their own drawing paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fundcloud.metrics import core as _metrics

__all__ = [
    "annotate_full_period_sharpe",
    "annotate_heatmap_margins",
    "annotate_var_cvar",
    "cumulative_pill",
    "distribution_pill",
    "drawdown_pill",
    "turnover",
]


def _fmt_pct(x: float, *, width: int = 0) -> str:
    if not np.isfinite(x):
        return "—".rjust(width) if width else "—"
    text = f"{x * 100.0:+.2f}%"
    return text.rjust(width) if width else text


def _fmt_num(x: float, *, decimals: int = 2, width: int = 0) -> str:
    if not np.isfinite(x):
        return "—".rjust(width) if width else "—"
    text = f"{x:+.{decimals}f}" if decimals else f"{x:+d}"
    return text.rjust(width) if width else text


def _fmt_date(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _label_width(series_list: list[tuple[str, pd.Series]], *, extra: str = "") -> int:
    """Width for the name column so every row lines up."""
    names = [name for name, _ in series_list]
    if extra:
        names = [*names, extra]
    return max(len(n) for n in names) + 1  # +1 for the trailing colon


def _row_label(name: str, width: int) -> str:
    return f"{name + ':':<{width + 1}}"  # pad AFTER the colon so the colon sits flush


def cumulative_pill(
    series_list: list[tuple[str, pd.Series]],
    *,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
) -> str:
    # Column widths: matches the longest value format we emit.
    W_TOTAL, W_CAGR, W_VOL, W_SHARPE = 11, 9, 9, 7
    bench_name = (
        str(benchmark.name) if benchmark is not None and benchmark.name is not None else "benchmark"
    )
    label_w = _label_width(series_list, extra=bench_name if benchmark is not None else "")

    lines = [
        f"{' ' * (label_w + 1)}"
        f"{'Total':>{W_TOTAL}}{'CAGR':>{W_CAGR}}{'Vol':>{W_VOL}}{'Sharpe':>{W_SHARPE}}"
    ]
    rows: list[tuple[str, pd.Series]] = list(series_list)
    if benchmark is not None:
        rows.append((bench_name, benchmark))
    for name, series in rows:
        total = _fmt_pct(_metrics.total_return(series), width=W_TOTAL)
        cagr = _fmt_pct(_metrics.cagr(series, periods_per_year=periods_per_year), width=W_CAGR)
        vol = _fmt_pct(_metrics.volatility(series, periods_per_year=periods_per_year), width=W_VOL)
        sharpe = _fmt_num(
            _metrics.sharpe(series, periods_per_year=periods_per_year), width=W_SHARPE
        )
        lines.append(f"{_row_label(name, label_w)}{total}{cagr}{vol}{sharpe}")
    return "<br>".join(lines)


def drawdown_pill(series_list: list[tuple[str, pd.Series]]) -> str:
    W_MAX, W_PT, W_AVG = 9, 25, 9  # Peak→Trough widest: "YYYY-MM-DD → YYYY-MM-DD" + padding
    label_w = _label_width(series_list)

    lines = [
        f"{' ' * (label_w + 1)}{'Max DD':>{W_MAX}}{'Peak → Trough':>{W_PT}}{'Avg DD':>{W_AVG}}"
    ]
    for name, series in series_list:
        dd = _metrics.drawdown_series(series)
        if dd.empty:
            lines.append(f"{_row_label(name, label_w)}—")
            continue
        trough_ts = dd.idxmin()
        pre_trough = dd.loc[:trough_ts]
        peak_candidates = pre_trough[pre_trough >= 0]
        peak_ts = peak_candidates.index[-1] if not peak_candidates.empty else dd.index[0]
        max_dd = float(dd.min())
        avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
        peak_trough = f"{_fmt_date(peak_ts)} → {_fmt_date(trough_ts)}"
        lines.append(
            f"{_row_label(name, label_w)}"
            f"{_fmt_pct(max_dd, width=W_MAX)}"
            f"{peak_trough:>{W_PT}}"
            f"{_fmt_pct(avg_dd, width=W_AVG)}"
        )
    return "<br>".join(lines)


def distribution_pill(
    series_list: list[tuple[str, pd.Series]],
    *,
    var_alpha: float = 0.95,
) -> str:
    W_MEAN, W_SIGMA, W_SK, W_KT, W_VAR, W_CVAR = 9, 9, 8, 8, 9, 9
    label_w = _label_width(series_list)

    lines = [
        f"{' ' * (label_w + 1)}"
        f"{'mean':>{W_MEAN}}{'stdev':>{W_SIGMA}}{'skew':>{W_SK}}{'kurt':>{W_KT}}"
        f"{'VaR5':>{W_VAR}}{'CVaR5':>{W_CVAR}}"
    ]
    for name, series in series_list:
        mu = f"{float(series.mean()) * 100.0:+.2f}%"
        sigma = f"{float(series.std(ddof=1)) * 100.0:.2f}%"
        sk = f"{float(_metrics.skew(series)):+.2f}"
        kt = f"{float(_metrics.kurtosis(series)):+.2f}"
        var = f"{float(_metrics.value_at_risk(series, alpha=var_alpha)) * 100.0:+.2f}%"
        cvar = f"{float(_metrics.cvar(series, alpha=var_alpha)) * 100.0:+.2f}%"
        lines.append(
            f"{_row_label(name, label_w)}"
            f"{mu:>{W_MEAN}}{sigma:>{W_SIGMA}}{sk:>{W_SK}}{kt:>{W_KT}}"
            f"{var:>{W_VAR}}{cvar:>{W_CVAR}}"
        )
    return "<br>".join(lines)


def annotate_var_cvar(
    fig: go.Figure,
    series_list: list[tuple[str, pd.Series]],
    *,
    alpha: float = 0.95,
) -> None:
    """Draw VaR / CVaR vertical reference lines on a distribution figure.

    Only the first series gets lines — multi-asset distributions otherwise
    accrete too many verticals to read. Values are shown in the stats pill,
    not as inline annotation text (which used to overlap the panel title
    and axis ticks).
    """
    if not series_list:
        return
    _, series = series_list[0]
    clean = series.dropna()
    var_pct = float(_metrics.value_at_risk(clean, alpha=alpha)) * 100.0
    cvar_pct = float(_metrics.cvar(clean, alpha=alpha)) * 100.0
    fig.add_vline(x=var_pct, line={"color": "#C0392B", "width": 1, "dash": "dash"})
    fig.add_vline(x=cvar_pct, line={"color": "#8E1D13", "width": 1, "dash": "dot"})


def annotate_full_period_sharpe(fig: go.Figure, full_sharpe: float | pd.Series) -> None:
    """Draw a dashed reference line at the full-period Sharpe value.

    When ``full_sharpe`` is a :class:`pandas.Series` (DataFrame input upstream),
    only the mean across columns is drawn — the rolling overlay already
    communicates the per-asset story.
    """
    value = float(full_sharpe.mean()) if isinstance(full_sharpe, pd.Series) else float(full_sharpe)
    if not np.isfinite(value):
        return
    fig.add_hline(
        y=value,
        line={"color": "rgba(0,0,0,0.55)", "width": 1, "dash": "dash"},
        annotation_text=f"full-period Sharpe {value:.2f}",
        annotation_position="top right",
    )


def annotate_heatmap_margins(fig: go.Figure, table: pd.DataFrame) -> None:
    """Add annual totals (right) and monthly averages (bottom) to a heatmap."""
    if table.empty:
        return
    annual = table.sum(axis=1, min_count=1)
    monthly_avg = table.mean(axis=0, skipna=True)
    years = [str(y) for y in table.index]
    for year, value in zip(years, annual.values, strict=True):
        if not np.isfinite(value):
            continue
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.01,
            y=year,
            text=f"{value:+.1f}%",
            showarrow=False,
            xanchor="left",
            font={"size": 10},
        )
    # Monthly averages along the bottom
    for month_label, value in zip(table.columns, monthly_avg.values, strict=True):
        if not np.isfinite(value):
            continue
        fig.add_annotation(
            xref="x",
            yref="paper",
            x=_month_label(month_label),
            y=-0.18,
            text=f"{value:+.1f}",
            showarrow=False,
            yanchor="top",
            font={"size": 10, "color": "rgba(0,0,0,0.65)"},
        )


def _month_label(month_index: int) -> str:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return names[int(month_index) - 1]


def turnover(weights: pd.DataFrame) -> float:
    """Average L1 turnover per period across all assets."""
    if weights.empty or len(weights) < 2:
        return 0.0
    diffs = weights.diff().abs().sum(axis=1).dropna()
    return float(diffs.mean() / 2.0) if not diffs.empty else 0.0
