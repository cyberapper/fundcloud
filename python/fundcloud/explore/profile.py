"""Native profile report — the highest-signal views of a DataFrame.

Replaces the previous ``ydata-profiling`` wrapper with a plotly+Jinja2
implementation that ships in the core install. Covers the six pieces of a
profiling report that a trader / quant actually reads: overview,
per-column stats, histograms, correlation heatmap, missing-pattern panel,
and alerts.

Returns a :class:`ProfileReport` object rather than a bare file path so
Python-first users can interrogate results at the REPL:

>>> from fundcloud.explore import profile
>>> report = profile(df)                      # doctest: +SKIP
>>> report.stats                               # doctest: +SKIP
>>> report.correlations["pearson"]             # doctest: +SKIP
>>> report.alerts                              # doctest: +SKIP
>>> report.to_html("out.html")                 # doctest: +SKIP

Passing ``output=`` still writes the full HTML file as a convenience —
the call then both writes the file and returns the report.
"""

from __future__ import annotations

import datetime as _dt
import numbers
from pathlib import Path

import pandas as pd

from fundcloud.explore import _figures
from fundcloud.explore._alerts import profile_alerts
from fundcloud.explore._report import ProfileReport
from fundcloud.explore._stats import overview, per_column_stats
from fundcloud.explore._template import _BASE_CSS, _TABS_JS, PROFILE_TEMPLATE, env

__all__ = ["profile"]


def profile(
    df: pd.DataFrame,
    *,
    output: str | Path | None = None,
    title: str | None = None,
    sample_rows: int = 5_000,
    embed_plotlyjs: bool = False,
) -> ProfileReport:
    """Build a :class:`ProfileReport` for ``df`` and optionally render HTML.

    Parameters
    ----------
    df
        Frame to profile. A :class:`~pandas.DatetimeIndex` is detected and
        surfaced in the overview.
    output
        Optional HTML path. When provided, the full report is also
        written to disk; the :class:`ProfileReport` is still returned so
        you can read ``.stats`` / ``.alerts`` / etc. at the REPL.
    title
        Optional report title.
    sample_rows
        Row budget for the missingness heatmap. Full panels are
        subsampled to this many rows; quantile / correlation work always
        uses the full frame.
    embed_plotlyjs
        ``False`` (default) loads plotly.js from CDN in the HTML —
        tiny file, needs internet to open. ``True`` inlines the minified
        plotly.js (~3 MB) so the report works offline.

    Returns
    -------
    :class:`ProfileReport`
        Call ``report.to_html(path)`` to write the rich HTML any time, or
        read ``report.stats`` / ``report.correlations`` / ``report.alerts``
        directly in Python.
    """
    numeric_df = df.select_dtypes(include="number")

    stats = per_column_stats(df)
    corr_pearson = numeric_df.corr(method="pearson") if numeric_df.shape[1] >= 2 else None

    correlations: dict[str, pd.DataFrame] = {}
    if corr_pearson is not None and not corr_pearson.empty:
        correlations["pearson"] = corr_pearson
        correlations["spearman"] = numeric_df.corr(method="spearman")

    missing = df.isna().sum()
    alerts = profile_alerts(stats, corr_pearson)
    report_title = title or "Fundcloud data profile"
    ov = overview(df)

    def _build_html(*, embed_plotlyjs: bool = False) -> str:
        histograms_html: list[tuple[str, str]] = []
        for col in numeric_df.columns:
            values = numeric_df[col].dropna().to_numpy(dtype=float, copy=False)
            if len(values) == 0:
                continue
            fig = _figures.histogram(str(col), values)
            histograms_html.append((str(col), _figures.fig_to_div(fig, include_plotlyjs=False)))

        correlation_html: dict[str, str] | None = None
        if correlations:
            fig_pearson = _figures.correlation_heatmap(correlations["pearson"], title="Pearson")
            fig_spearman = _figures.correlation_heatmap(correlations["spearman"], title="Spearman")
            correlation_html = {
                "pearson": _figures.fig_to_div(fig_pearson, include_plotlyjs=False),
                "spearman": _figures.fig_to_div(fig_spearman, include_plotlyjs=False),
            }

        missing_html: dict[str, str] | None = None
        if df.isna().any().any():
            # Shared ordering: the bar chart and the heatmap use the same
            # columns-by-missing-count-descending order, so column N on the
            # bar chart lines up with column N on the heatmap below it.
            column_order = _figures._missing_column_order(df)
            bar_div = _figures.fig_to_div(
                _figures.missing_bar(df, column_order=column_order),
                include_plotlyjs=False,
            )
            if df.shape[1] == 1:
                # Single-column panels: a 1-column heatmap would render as a
                # narrow stripe that looks deceptively like a bar. Use a
                # row-axis timeline instead so the reader sees WHEN values
                # are missing, not just how many.
                sampled_rows = min(len(df), sample_rows)
                caption = f"timeline, 1 column · {sampled_rows} rows sampled"
                heatmap_div = _figures.fig_to_div(
                    _figures.missing_timeline(df, sample_rows=sample_rows),
                    include_plotlyjs=False,
                )
            else:
                sampled_rows = min(len(df), sample_rows)
                caption = (
                    f"{sampled_rows} rows sampled, {df.shape[1]} cols; "
                    f"column order matches bar chart above"
                )
                heatmap_div = _figures.fig_to_div(
                    _figures.missing_heatmap(
                        df, sample_rows=sample_rows, column_order=column_order
                    ),
                    include_plotlyjs=False,
                )
            missing_html = {
                "bar": bar_div,
                "heatmap": heatmap_div,
                "caption": caption,
            }

        template = env.from_string(PROFILE_TEMPLATE)
        return template.render(
            title=report_title,
            subtitle=f"{len(df):,} rows · {df.shape[1]} columns",
            generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
            css=_BASE_CSS,
            tabs_js=_TABS_JS,
            overview=ov,
            stats_table_html=_stats_to_html(stats),
            histograms_html=histograms_html,
            correlation_html=correlation_html,
            missing_html=missing_html,
            alerts=alerts,
            plotlyjs_block=_plotlyjs_block(embed_plotlyjs),
        )

    report = ProfileReport(
        overview=ov,
        stats=stats,
        correlations=correlations,
        missing=missing,
        alerts=alerts,
        title=report_title,
        _html_builder=_build_html,
    )

    if output is not None:
        report.to_html(output, embed_plotlyjs=embed_plotlyjs)
    return report


# ---------------------------------------------------------------------- helpers


def _stats_to_html(stats: pd.DataFrame) -> str:
    display_cols = [
        "dtype",
        "count",
        "missing_pct",
        "distinct",
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
    ]
    view = stats[[c for c in display_cols if c in stats.columns]].copy()
    return view.to_html(
        classes="stats",
        border=0,
        float_format=_number_fmt,
        na_rep="—",
    )


def _number_fmt(x: object) -> str:
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        v = float(x)
        if v != v:
            return "—"
        return f"{v:.4g}"
    return str(x)


def _plotlyjs_block(embed: bool) -> str:
    if embed:
        from plotly.offline import get_plotlyjs  # type: ignore[import-not-found]

        return f"<script>{get_plotlyjs()}</script>"
    return '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
