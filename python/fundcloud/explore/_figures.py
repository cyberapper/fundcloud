"""Plotly figure builders shared by :func:`profile` and :func:`compare`.

Every figure is returned as a :class:`plotly.graph_objects.Figure` so the
rendering layer can embed them with ``pio.to_html(..., include_plotlyjs=...)``
in exactly one place (avoiding duplicate plotly.js payloads).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from plotly.graph_objects import Figure

__all__ = [
    "correlation_delta_heatmap",
    "correlation_heatmap",
    "fig_to_div",
    "histogram",
    "missing_bar",
    "missing_heatmap",
    "missing_timeline",
    "overlay_histogram",
]


def _go() -> object:
    import plotly.graph_objects as go

    return go


def _layout_defaults() -> dict[str, object]:
    return {
        "margin": dict(l=48, r=24, t=16, b=32),
        "height": 240,
        "autosize": True,
        "template": "plotly_white",
        "font": dict(size=12),
    }


def histogram(name: str, values: np.ndarray) -> Figure:
    """Column-level histogram.

    No internal title — the surrounding ``<details><summary>`` block in
    the report template already displays the asset name; a duplicate
    title just wastes vertical space and overlaps with the summary row.
    """
    go = _go()
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    fig = go.Figure(go.Histogram(x=arr, nbinsx=40, marker=dict(color="#3b82f6")))  # type: ignore[attr-defined]
    fig.update_layout(  # type: ignore[attr-defined]
        title=None,
        xaxis_title=None,
        yaxis_title="count",
        showlegend=False,
        **_layout_defaults(),
    )
    # Preserve the asset name for downstream JS / debuggers, but don't render it.
    fig.update_layout(meta=dict(column=name))  # type: ignore[attr-defined]
    return fig  # type: ignore[no-any-return]


def correlation_heatmap(corr: pd.DataFrame, *, title: str | None = None) -> Figure:
    go = _go()
    labels = [str(c) for c in corr.columns]
    values = corr.to_numpy(dtype=float)
    fig = go.Figure(  # type: ignore[attr-defined]
        go.Heatmap(  # type: ignore[attr-defined]
            z=values,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="corr"),
            # Each cell shows its rounded correlation in-place so readers
            # don't need to colour-match against the scale bar.
            text=values,
            texttemplate="%{z:.2f}",
            textfont=dict(size=10),
            hovertemplate="%{x} ↔ %{y}: %{z:.3f}<extra></extra>",
        )
    )
    height = max(320, 18 * len(labels) + 120)
    fig.update_layout(  # type: ignore[attr-defined]
        title=title,
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=32, t=48, b=120),
        height=height,
        autosize=True,
        template="plotly_white",
        font=dict(size=11),
    )
    return fig  # type: ignore[no-any-return]


def _missing_column_order(df: pd.DataFrame) -> list[str]:
    """Columns sorted by missing-count descending. Used by both the bar
    chart and the missingness heatmap so they line up visually."""
    missing = df.isna().sum().sort_values(ascending=False)
    return [str(c) for c in missing.index]


def missing_bar(df: pd.DataFrame, *, column_order: list[str] | None = None) -> Figure:
    go = _go()
    missing = df.isna().sum()
    order = column_order if column_order is not None else _missing_column_order(df)
    values = [int(missing[c]) for c in order]
    fig = go.Figure(  # type: ignore[attr-defined]
        go.Bar(  # type: ignore[attr-defined]
            x=order,
            y=values,
            marker=dict(color="#ef4444"),
        )
    )
    fig.update_layout(  # type: ignore[attr-defined]
        title=None,
        xaxis=dict(title=None, tickangle=-45, automargin=True),
        yaxis_title="missing rows",
        **_layout_defaults(),
    )
    return fig  # type: ignore[no-any-return]


def missing_heatmap(
    df: pd.DataFrame,
    *,
    sample_rows: int = 500,
    column_order: list[str] | None = None,
) -> Figure:
    """Row × column missingness mask.

    Multi-column input renders as a heatmap (red = missing, grey = present)
    with columns in the caller-supplied order (or missing-count-descending
    by default) so it aligns with :func:`missing_bar`.
    """
    go = _go()
    order = column_order if column_order is not None else _missing_column_order(df)
    frame = df[order]
    sampled = (
        frame.sample(n=sample_rows, random_state=0).sort_index()
        if len(frame) > sample_rows
        else frame
    )
    mask = sampled.isna().astype(int).to_numpy()
    fig = go.Figure(  # type: ignore[attr-defined]
        go.Heatmap(  # type: ignore[attr-defined]
            z=mask,
            x=[str(c) for c in sampled.columns],
            y=[str(i) for i in sampled.index],
            zmin=0.0,
            zmax=1.0,
            colorscale=[(0.0, "#f1f5f9"), (1.0, "#ef4444")],
            showscale=False,
        )
    )
    fig.update_layout(  # type: ignore[attr-defined]
        title=None,
        xaxis=dict(tickangle=-45, automargin=True),
        yaxis=dict(showticklabels=False, title="rows (sampled)"),
        margin=dict(l=40, r=24, t=16, b=80),
        height=max(240, min(600, 6 * len(sampled))),
        autosize=True,
        template="plotly_white",
        font=dict(size=11),
    )
    return fig  # type: ignore[no-any-return]


def missing_timeline(df: pd.DataFrame, *, sample_rows: int = 2_000) -> Figure:
    """Single-column missingness timeline — a flat strip coloured by
    presence/absence per row.

    Used in place of :func:`missing_heatmap` when the caller has a single
    column (the heatmap would degenerate to a 1-column stripe that looks
    like a bar, confusing readers).
    """
    go = _go()
    if df.shape[1] != 1:
        msg = "missing_timeline expects a single-column frame; got " f"{df.shape[1]} columns"
        raise ValueError(msg)
    col = df.columns[0]
    ser = df[col]
    sampled = (
        ser.sample(n=sample_rows, random_state=0).sort_index()
        if len(ser) > sample_rows
        else ser
    )
    mask = sampled.isna().astype(int).to_numpy().reshape(1, -1)
    fig = go.Figure(  # type: ignore[attr-defined]
        go.Heatmap(  # type: ignore[attr-defined]
            z=mask,
            x=[str(i) for i in sampled.index],
            y=[str(col)],
            zmin=0.0,
            zmax=1.0,
            colorscale=[(0.0, "#f1f5f9"), (1.0, "#ef4444")],
            showscale=False,
        )
    )
    fig.update_layout(  # type: ignore[attr-defined]
        title=None,
        xaxis=dict(title=None, tickangle=-45, automargin=True, nticks=10),
        yaxis=dict(showticklabels=True, title=None),
        margin=dict(l=60, r=24, t=16, b=80),
        height=140,
        autosize=True,
        template="plotly_white",
        font=dict(size=11),
    )
    return fig  # type: ignore[no-any-return]


def overlay_histogram(
    name: str, a_values: np.ndarray, b_values: np.ndarray, a_label: str, b_label: str
) -> Figure:
    go = _go()
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    fig = go.Figure()  # type: ignore[attr-defined]
    fig.add_trace(  # type: ignore[attr-defined]
        go.Histogram(  # type: ignore[attr-defined]
            x=a, nbinsx=40, name=a_label, opacity=0.55, marker=dict(color="#3b82f6")
        )
    )
    fig.add_trace(  # type: ignore[attr-defined]
        go.Histogram(  # type: ignore[attr-defined]
            x=b, nbinsx=40, name=b_label, opacity=0.55, marker=dict(color="#ef4444")
        )
    )
    fig.update_layout(  # type: ignore[attr-defined]
        title=None,
        barmode="overlay",
        xaxis_title=None,
        yaxis_title="count",
        legend=dict(orientation="h", y=1.08),
        meta=dict(column=name),
        **_layout_defaults(),
    )
    return fig  # type: ignore[no-any-return]


def correlation_delta_heatmap(delta: pd.DataFrame) -> Figure:
    go = _go()
    labels = [str(c) for c in delta.columns]
    values = delta.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1e-6)
    fig = go.Figure(  # type: ignore[attr-defined]
        go.Heatmap(  # type: ignore[attr-defined]
            z=values,
            x=labels,
            y=labels,
            zmin=-vmax,
            zmax=vmax,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="d_corr"),
            text=values,
            texttemplate="%{z:+.2f}",
            textfont=dict(size=10),
            hovertemplate="%{x} ↔ %{y}: %{z:+.3f}<extra></extra>",
        )
    )
    height = max(320, 18 * len(labels) + 120)
    fig.update_layout(  # type: ignore[attr-defined]
        title="Correlation delta (b - a)",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=32, t=48, b=120),
        height=height,
        template="plotly_white",
        font=dict(size=11),
    )
    return fig  # type: ignore[no-any-return]


def fig_to_div(fig: Figure, *, include_plotlyjs: bool | str) -> str:
    """Render a figure as an HTML ``<div>`` with optional plotly.js embed."""
    import plotly.io as pio

    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
