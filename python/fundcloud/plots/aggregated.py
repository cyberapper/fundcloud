"""Headless aggregation: one Figure with all the key analysed panels.

:func:`summary` composes the individual plot builders into a single
:class:`plotly.graph_objects.Figure` using :func:`plotly.subplots.make_subplots`.
Each analytical section gets a full-width row so labels stay readable even
on dense tear sheets; no subplots inside a row.

The result is static — return it and render where you want
(``fig.show()`` / ``fig.write_html(...)`` / ``fig.write_image(...)``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fundcloud._benchmark import resolve_benchmark as _resolve_benchmark
from fundcloud.metrics.rolling import rolling_alpha, rolling_beta
from fundcloud.plots import plotly as _plt
from fundcloud.plots._normalize import to_series_list
from fundcloud.plots.themes import _resolve_template

__all__ = ["summary"]


def summary(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | str | None = None,
    weights: pd.DataFrame | None = None,
    theme: str | None = None,
    title: str | None = None,
    heatmap_asset: str | None = None,
) -> go.Figure:
    """Return a composite figure with the canonical tearsheet panels.

    Each section uses a full-width row — easier to read than a tiled grid:

    ==  =============================================================
    row content
    ==  =============================================================
    1   Cumulative returns
    2   Drawdown (%)
    3   Rolling Sharpe
    4   Return distribution (%)
    5   Monthly returns heatmap (first series)
    6   Portfolio composition (only when ``weights`` is supplied)
    ==  =============================================================

    Annotations (``annotations=True``) are applied per panel — stats pills,
    VaR/CVaR reference lines, full-period Sharpe reference, heatmap cell
    values. A single color per asset is maintained across rows, and each
    asset contributes one legend entry rather than one per panel.
    """
    # Resolve string benchmark against a DataFrame's column first, so the
    # single-column heatmap asset also sees the string if it matches.
    benchmark = _resolve_benchmark(returns, benchmark)

    include_composition = weights is not None and not weights.empty
    include_benchmark = benchmark is not None
    series_list = to_series_list(returns)
    series_by_name = dict(series_list)

    # Monthly heatmap behaviour: one row per asset when the input is
    # multi-asset, a single row when it's a single series. ``heatmap_asset=``
    # narrows that to one named column.
    if heatmap_asset is not None:
        if heatmap_asset not in series_by_name:
            known = [name for name, _ in series_list]
            msg = f"heatmap_asset={heatmap_asset!r} not in returns columns {known}"
            raise ValueError(msg)
        heatmap_assets = [heatmap_asset]
    else:
        heatmap_assets = [name for name, _ in series_list]

    # Build row layout top-to-bottom. Each layout entry is (kind, asset|None):
    # only ``monthly_heatmap`` rows carry an asset name — other panels already
    # show every asset together.
    layout: list[tuple[str, str | None]] = [
        ("cumulative", None),
        ("drawdown", None),
        ("rolling_sharpe", None),
    ]
    if include_benchmark:
        layout.append(("rolling_alpha", None))
        layout.append(("rolling_beta", None))
    layout.append(("return_distribution", None))
    for hm_name in heatmap_assets:
        layout.append(("monthly_heatmap", hm_name))
    if include_composition:
        layout.append(("composition", None))

    title_base = {
        "cumulative": "Cumulative returns",
        "drawdown": "Drawdown (%)",
        "rolling_sharpe": "Rolling Sharpe",
        "rolling_alpha": "Rolling alpha (annualised)",
        "rolling_beta": "Rolling beta",
        "return_distribution": "Return distribution (%)",
        "composition": "Portfolio composition",
    }

    def _panel_title(kind: str, asset: str | None) -> str:
        if kind == "monthly_heatmap":
            if len(series_list) > 1 or heatmap_asset is not None:
                return f"Monthly returns (%) — {asset}"
            return "Monthly returns (%)"
        return title_base[kind]

    subplot_titles = [_panel_title(k, a) for k, a in layout]
    rows = len(layout)
    # Row index (1-based) for each heatmap, needed so each colorbar can be
    # anchored to its own panel.
    heatmap_rows = [i for i, (k, _) in enumerate(layout, start=1) if k == "monthly_heatmap"]

    # Every row is a single full-width cell — no colspan gymnastics.
    specs: list[list[dict[str, Any]]] = [[{}] for _ in range(rows)]

    # Give each monthly heatmap ~1.6x the baseline row so 13+ years by 12
    # months with in-cell text don't squish into a single visible strip.
    row_weights = [1.6 if k == "monthly_heatmap" else 1.0 for k, _ in layout]
    total = sum(row_weights)
    row_heights = [w / total for w in row_weights]

    fig = make_subplots(
        rows=rows,
        cols=1,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.055,
        row_heights=row_heights,
    )

    # Build each sub-figure with annotations=True so stats pills land on
    # the composite. Heatmap colorbars are anchored to their rows explicitly.
    cum = _plt.cumulative(returns, benchmark=benchmark, annotations=True)
    dd = _plt.drawdown(returns, annotations=True)
    rs = _plt.rolling_sharpe(returns, annotations=True)
    dist = _plt.return_distribution(returns, annotations=True)

    # annotations=False is intentional: the heatmap's in-cell texttemplate
    # already shows each month's return. Adding the annotate_heatmap_margins
    # annotations (annual totals + monthly averages) causes plotly to extend
    # the category axis range inside the composite subplot, collapsing cells
    # to near-zero height. Keep every composite heatmap clean; the margins
    # annotations are available when the heatmap is rendered standalone.
    heatmaps: dict[str, go.Figure] = {
        name: _plt.monthly_heatmap(
            series_by_name[name].rename(name),
            annotations=False,
            colorbar=_heatmap_colorbar_from_weights(row_heights, heatmap_row=hm_row),
        )
        for name, hm_row in zip(heatmap_assets, heatmap_rows, strict=True)
    }

    sub_regular: dict[str, go.Figure] = {
        "cumulative": cum,
        "drawdown": dd,
        "rolling_sharpe": rs,
        "return_distribution": dist,
    }
    if include_benchmark:
        sub_regular["rolling_alpha"] = _rolling_alpha_figure(returns, benchmark)
        sub_regular["rolling_beta"] = _rolling_beta_figure(returns, benchmark)
    if include_composition:
        sub_regular["composition"] = _plt.composition(weights, annotations=True)

    for i, (kind, asset_name) in enumerate(layout, start=1):
        sub = heatmaps[asset_name] if kind == "monthly_heatmap" else sub_regular[kind]
        _place(fig, sub, row=i)

    # Consistent per-asset color + legend dedup across panels.
    _unify_asset_colors_and_legend(fig)

    # Global chrome. Figure height follows the row weights so the heatmap
    # gets a proportional share of vertical space (not a tiny 1/N slot).
    template = _resolve_template(theme)
    fig.update_layout(
        height=int(280 * total) + 120,
        title={"text": title or "Strategy summary", "x": 0.01},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        margin={"l": 60, "r": 40, "t": 90, "b": 60},
    )
    if template is None:
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"family": "Inter, system-ui, sans-serif", "size": 12},
        )
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.2)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.2)")
    else:
        fig.update_layout(template=template)
    return fig


# --------------------------------------------------------------------- helpers


def _rolling_alpha_figure(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 63,
    periods_per_year: int = 252,
) -> go.Figure:
    """One-row figure of rolling α vs ``benchmark``, ready to be harvested."""
    alpha = rolling_alpha(returns, benchmark, window=window, periods_per_year=periods_per_year)
    fig = go.Figure()
    for name, _series in to_series_list(returns):
        col = alpha[name] if isinstance(alpha, pd.DataFrame) else alpha
        series = col.dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=name,
                mode="lines",
                line={"width": 1.6},
                connectgaps=True,
            )
        )
    fig.add_hline(y=0, line={"color": "rgba(0,0,0,0.4)", "width": 1, "dash": "dot"})
    fig.update_yaxes(tickformat=".2%")
    return fig


def _rolling_beta_figure(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 63,
) -> go.Figure:
    """One-row figure of rolling β vs ``benchmark``."""
    beta = rolling_beta(returns, benchmark, window=window)
    fig = go.Figure()
    for name, _series in to_series_list(returns):
        col = beta[name] if isinstance(beta, pd.DataFrame) else beta
        series = col.dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=name,
                mode="lines",
                line={"width": 1.6},
                connectgaps=True,
            )
        )
    fig.add_hline(y=1, line={"color": "rgba(0,0,0,0.4)", "width": 1, "dash": "dot"})
    fig.update_yaxes(tickformat=".2f")
    return fig


def _heatmap_colorbar_from_weights(row_heights: list[float], *, heatmap_row: int) -> dict[str, Any]:
    """Anchor the heatmap colorbar at the heatmap row's paper y-center.

    Works with make_subplots row_heights so the bar tracks the heatmap row
    whether the rows are uniform or not. heatmap_row is 1-indexed from the
    top; paper coords run 0 (bottom) → 1 (top), hence the 1.0 - ... invert.
    """
    # Sum of heights above the target row (from the top).
    above = sum(row_heights[: heatmap_row - 1])
    own = row_heights[heatmap_row - 1]
    y_center = 1.0 - (above + own / 2.0)
    return {
        "x": 1.02,
        "y": y_center,
        "len": own * 0.9,
        "thickness": 10,
        "title": "%",
    }


def _place(fig: go.Figure, sub: go.Figure, *, row: int) -> None:
    """Harvest traces / shapes / annotations from ``sub`` into the composite.

    All xref / yref values are rebased to the composite's axis for ``row``,
    including the tricky ``"x domain"`` / ``"y domain"`` suffixed refs used
    by ``add_vline`` / ``add_hline``.
    """
    target_x = _axis_key("x", row=row)
    target_y = _axis_key("y", row=row)

    for trace in sub.data:
        fig.add_trace(trace, row=row, col=1)

    for annotation in sub.layout.annotations:
        new = annotation.to_plotly_json()
        _rewrite_refs(new, target_x=target_x, target_y=target_y, anchor_paper_to_domain=True)
        fig.add_annotation(**new)

    for shape in sub.layout.shapes:
        new = shape.to_plotly_json()
        _rewrite_refs(new, target_x=target_x, target_y=target_y, anchor_paper_to_domain=False)
        fig.add_shape(**new)

    # Preserve axis attributes the sub-builder set (tick format, but also
    # categorical axis for the heatmap — otherwise row labels like "2014"
    # get numericised and plotly culls years).
    _copy_axis_attrs(fig, sub.layout.xaxis, axis="x", row=row)
    _copy_axis_attrs(fig, sub.layout.yaxis, axis="y", row=row)


def _copy_axis_attrs(fig: go.Figure, sub_axis: Any, *, axis: str, row: int) -> None:
    """Propagate tick / type / autorange settings from a sub-figure axis onto
    the composite's matching row axis. Without this, heatmap-specific axis
    tweaks (tickmode=array, type=category) are lost in the composite."""
    if sub_axis is None:
        return
    update = {}
    for attr in ("tickformat", "tickmode", "tickvals", "ticktext", "type", "autorange"):
        val = getattr(sub_axis, attr, None)
        if val is not None and val != ():
            update[attr] = val
    if not update:
        return
    if axis == "x":
        fig.update_xaxes(row=row, col=1, **update)
    else:
        fig.update_yaxes(row=row, col=1, **update)


def _rewrite_refs(
    item: dict[str, Any],
    *,
    target_x: str,
    target_y: str,
    anchor_paper_to_domain: bool,
) -> None:
    """Remap xref / yref in an annotation or shape dict onto the target subplot.

    * ``"x"`` → ``target_x`` (e.g. ``"x3"``)
    * ``"x domain"`` → ``"{target_x} domain"``
    * ``"paper"`` → ``"{target_x} domain"`` iff ``anchor_paper_to_domain``
      (used for stats pills that should track the subplot, not the figure)
    * Same for y.
    """

    def remap(ref: str | None, axis: str, target: str) -> str | None:
        if ref is None:
            return ref
        if ref == axis:
            return target
        if ref == f"{axis} domain":
            return f"{target} domain"
        if ref == "paper" and anchor_paper_to_domain:
            return f"{target} domain"
        return ref

    if "xref" in item:
        item["xref"] = remap(item["xref"], "x", target_x)
    if "yref" in item:
        item["yref"] = remap(item["yref"], "y", target_y)


def _axis_key(axis: str, *, row: int) -> str:
    """Plotly axis id for a 1-column subplot at ``row`` (row 1 → ``"x"``, row 2 → ``"x2"``, ...)."""
    if row == 1:
        return axis
    return f"{axis}{row}"


_FALLBACK_PALETTE: tuple[str, ...] = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#BCBD22",
    "#17BECF",
)


def _unify_asset_colors_and_legend(fig: go.Figure) -> None:
    """Give every trace with the same legendgroup a single colour + one entry.

    Sub-figures auto-cycle colours independently, so an asset ends up blue
    in the cumulative panel but green in the drawdown panel. This post-pass
    groups by the base asset name (stripping suffixes like ``" (63-bar)"``),
    assigns one palette slot per group, and hides duplicate legend entries.
    """
    palette = _resolve_palette(fig)
    color_by_group: dict[str, str] = {}
    seen_groups: set[str] = set()
    for trace in fig.data:
        # Heatmaps carry their story through the colorbar; a legend entry
        # would be meaningless (single z-field, no category to toggle).
        if getattr(trace, "type", None) == "heatmap":
            trace.showlegend = False
            continue
        name = getattr(trace, "name", None) or ""
        group = _base_group(name)
        trace.legendgroup = group
        if group not in color_by_group:
            color_by_group[group] = (
                "#888888"
                if "benchmark" in group.lower()
                else palette[len(color_by_group) % len(palette)]
            )
        _apply_trace_color(trace, color_by_group[group])
        trace.showlegend = group not in seen_groups
        seen_groups.add(group)


def _resolve_palette(fig: go.Figure) -> list[str]:
    try:
        colorway = fig.layout.template.layout.colorway
    except AttributeError:
        colorway = None
    return list(colorway) if colorway else list(_FALLBACK_PALETTE)


def _base_group(name: str) -> str:
    """Strip builder-specific suffixes so traces of the same asset share a group."""
    if not name:
        return "strategy"
    return name.split(" (")[0].strip() or name


def _apply_trace_color(trace: Any, color: str) -> None:
    """Overwrite the trace's palette-driven colour with ``color``.

    Only applies to scatter / histogram traces; heatmap colorscales are
    left alone. Explicit hex colours set by a sub-builder (e.g. the grey
    dashed benchmark line) are preserved.
    """
    ttype = getattr(trace, "type", None)
    if ttype == "heatmap":
        return
    if ttype == "histogram":
        existing = getattr(trace.marker, "color", None) if trace.marker else None
        if existing is None:
            trace.marker = (trace.marker.to_plotly_json() if trace.marker else {}) | {
                "color": color
            }
        return
    # Scatter / line traces
    if trace.line is not None and trace.line.color is None:
        trace.line = trace.line.to_plotly_json() | {"color": color}
