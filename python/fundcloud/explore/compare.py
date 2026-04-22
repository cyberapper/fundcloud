"""Native two-dataset comparison — the highest-signal drift views.

Replaces the previous ``sweetviz`` wrapper with a plotly+Jinja2
implementation that ships in the core install. Produces:

1. Side-by-side overview.
2. Per-column drift table (means, stds, KS, Wasserstein, missing delta).
3. Overlay histograms per shared numeric column.
4. Correlation delta heatmap.
5. Optional target-correlation shift table when ``target`` is given.
6. Alerts for schema changes, distribution shifts, and target-shift.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from fundcloud.explore import _figures
from fundcloud.explore._alerts import compare_alerts
from fundcloud.explore._drift import drift_table
from fundcloud.explore._stats import overview
from fundcloud.explore._template import _BASE_CSS, _TABS_JS, COMPARE_TEMPLATE, env
from fundcloud.explore.profile import _number_fmt, _plotlyjs_block

__all__ = ["compare"]


def compare(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    output: str | Path | None = None,
    names: tuple[str, str] = ("a", "b"),
    target: str | None = None,
    title: str | None = None,
    embed_plotlyjs: bool = False,
) -> Path | str:
    """Generate a self-contained HTML comparison report."""
    shared = [c for c in a.columns if c in b.columns]
    only_a = [c for c in a.columns if c not in b.columns]
    only_b = [c for c in b.columns if c not in a.columns]

    drift = drift_table(a, b, shared)

    numeric_shared = [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
    ]

    overlays: list[tuple[str, str]] = []
    for col in numeric_shared:
        values_a = a[col].dropna().to_numpy(dtype=float, copy=False)
        values_b = b[col].dropna().to_numpy(dtype=float, copy=False)
        if len(values_a) == 0 or len(values_b) == 0:
            continue
        fig = _figures.overlay_histogram(str(col), values_a, values_b, names[0], names[1])
        overlays.append((str(col), _figures.fig_to_div(fig, include_plotlyjs=False)))

    correlation_delta_html = _correlation_delta_html(a, b, numeric_shared)

    target_shift_html, target_shifts = _target_shift(a, b, target, numeric_shared)

    alerts = compare_alerts(drift, list(a.columns), list(b.columns), target_shifts=target_shifts)

    template = env.from_string(COMPARE_TEMPLATE)
    html = template.render(
        title=title or f"Fundcloud compare · {names[0]} vs {names[1]}",
        subtitle=f"{len(a):,} rows ({names[0]}) · {len(b):,} rows ({names[1]})",
        generated_at=_dt.datetime.now().isoformat(timespec="seconds"),
        css=_BASE_CSS,
        tabs_js=_TABS_JS,
        names=names,
        shared=shared,
        only_a=only_a,
        only_b=only_b,
        overview={"a": overview(a), "b": overview(b)},
        drift_table_html=_drift_to_html(drift),
        overlay_histograms_html=overlays,
        correlation_delta_html=correlation_delta_html,
        target_shift_html=target_shift_html,
        alerts=alerts,
        plotlyjs_block=_plotlyjs_block(embed_plotlyjs),
    )

    if output is None:
        return html
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------- helpers


def _drift_to_html(drift: pd.DataFrame) -> str:
    if drift.empty:
        return ""
    return drift.to_html(classes="stats", border=0, float_format=_number_fmt, na_rep="—")


def _correlation_delta_html(a: pd.DataFrame, b: pd.DataFrame, numeric_shared: list[str]) -> str:
    if len(numeric_shared) < 2:
        return ""
    corr_a = a[numeric_shared].corr()
    corr_b = b[numeric_shared].corr()
    delta = corr_b - corr_a
    fig = _figures.correlation_delta_heatmap(delta)
    return _figures.fig_to_div(fig, include_plotlyjs=False)


def _target_shift(
    a: pd.DataFrame,
    b: pd.DataFrame,
    target: str | None,
    numeric_shared: list[str],
) -> tuple[str, dict[str, float] | None]:
    if target is None or target not in numeric_shared:
        return "", None

    feature_cols = [c for c in numeric_shared if c != target]
    if not feature_cols:
        return "", None

    target_a = a[target]
    target_b = b[target]
    rows: list[dict[str, float | str]] = []
    shifts: dict[str, float] = {}
    for col in feature_cols:
        try:
            corr_a = float(a[col].corr(target_a))
            corr_b = float(b[col].corr(target_b))
        except (TypeError, ValueError):
            continue
        if np.isnan(corr_a) and np.isnan(corr_b):
            continue
        delta = (corr_b if not np.isnan(corr_b) else 0.0) - (
            corr_a if not np.isnan(corr_a) else 0.0
        )
        shifts[col] = delta
        rows.append({
            "feature": col,
            f"corr_{target}_a": corr_a,
            f"corr_{target}_b": corr_b,
            "delta": delta,
        })
    if not rows:
        return "", None
    frame = pd.DataFrame(rows).set_index("feature")
    frame = frame.reindex(frame["delta"].abs().sort_values(ascending=False).index)
    return frame.to_html(classes="stats", border=0, float_format=_number_fmt, na_rep="—"), shifts
