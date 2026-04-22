"""Tear-sheet formatting helpers — shared by HTML, PDF, and Excel renderers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fundcloud.reports.metric_info import METRIC_INFO, Category, category_order

__all__ = [
    "CategorySection",
    "MetricRow",
    "StatCard",
    "StatRow",
    "categorized_sections",
    "format_stat",
    "stat_cards",
    "stats_rows",
]


@dataclass(slots=True)
class StatCard:
    label: str
    value: str
    klass: str = ""


@dataclass(slots=True)
class StatRow:
    label: str
    value: str


@dataclass(slots=True)
class MetricRow:
    """One metric row on the sidebar — carries display + hover tooltip text.

    ``bench_value`` is populated on benchmark-comparison sections. ``klass``
    is the CSS class for colouring (``pos`` / ``neg`` / empty).
    """

    key: str
    label: str
    value: str
    bench_value: str = ""
    definition: str = ""
    formula: str = ""
    klass: str = ""


@dataclass(slots=True)
class CategorySection:
    """A named group of metric rows — one per sidebar section."""

    category: str
    rows: list[MetricRow] = field(default_factory=list)


_PCT_METRICS = {
    "total_return",
    "cagr",
    "ann_volatility",
    "downside_volatility",
    "max_drawdown",
    "cvar",
    "value_at_risk",
    "ulcer_index",
    "pain_index",
    "avg_return",
    "avg_win",
    "avg_loss",
    "best",
    "worst",
    "tracking_error",
    "alpha",
    "best_month",
    "worst_month",
    "best_year",
    "worst_year",
}
_RATIO_METRICS = {
    "sharpe",
    "sortino",
    "calmar",
    "omega",
    "adjusted_sortino",
    "smart_sharpe",
    "smart_sortino",
    "payoff_ratio",
    "profit_factor",
    "tail_ratio",
    "common_sense_ratio",
    "gain_to_pain_ratio",
    "pain_ratio",
    "ulcer_performance_index",
    "treynor_ratio",
    "information_ratio",
    "beta",
    "r_squared",
    "correlation",
    "up_capture",
    "down_capture",
    "capture_ratio",
    "probabilistic_sharpe",
    "kelly_criterion",
    "win_rate",
    "exposure",
    "risk_of_ruin",
}


def format_stat(name: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    info = METRIC_INFO.get(name)
    if info is not None:
        if info.fmt == "pct":
            return f"{value * 100.0:.2f}%"
        if info.fmt == "pct4":
            return f"{value * 100.0:.4f}%"
        if info.fmt == "ratio":
            return f"{value:.2f}"
        if info.fmt == "int":
            return f"{int(value):d}"
    # Fall-through heuristics for keys without METRIC_INFO entries.
    if name in _PCT_METRICS:
        return f"{value * 100.0:.2f}%"
    if name in _RATIO_METRICS:
        return f"{value:.2f}"
    if name == "periods":
        return f"{int(value):d}"
    return f"{value:.4f}"


def _klass_for(name: str, value: float) -> str:
    """Return CSS class for coloured value rendering (pos / neg / "")."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    info = METRIC_INFO.get(name)
    sign = info.sign if info is not None else "neutral"
    v = float(value)
    if sign == "pos":
        return "pos" if v > 0 else ""
    if sign == "neg":
        return "neg" if v < 0 else ""
    if sign == "signed":
        if v > 0:
            return "pos"
        if v < 0:
            return "neg"
    return ""


_ACRONYMS = {
    "cagr",
    "cvar",
    "var",
    "ir",
    "r_squared",
    "vix",
}


def _label(name: str) -> str:
    if name in _ACRONYMS:
        return name.replace("_", "-").upper()
    # Keep common acronyms uppercase inside multi-word labels.
    words = []
    for w in name.split("_"):
        if w in {"cagr", "cvar", "var", "ir"}:
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)


def stat_cards(stats: pd.Series) -> list[StatCard]:
    """Top-of-page stat cards — four high-signal metrics."""
    wanted = [
        ("cagr", "CAGR"),
        ("sharpe", "Sharpe"),
        ("max_drawdown", "Max drawdown"),
        ("cvar", "CVaR (95%)"),
    ]
    cards: list[StatCard] = []
    for key, label in wanted:
        if key not in stats.index:
            continue
        val = stats[key]
        formatted = format_stat(key, float(val) if pd.notna(val) else float("nan"))
        klass = ""
        if pd.notna(val) and isinstance(val, (int, float, np.floating)):
            if key in {"max_drawdown", "cvar"}:
                klass = "neg" if float(val) < 0 else ""
            elif float(val) > 0:
                klass = "pos"
            elif float(val) < 0:
                klass = "neg"
        cards.append(StatCard(label=label, value=formatted, klass=klass))
    return cards


def stats_rows(stats: pd.Series) -> list[StatRow]:
    """Row-per-metric table for the bottom of the tear sheet."""
    return [
        StatRow(label=_label(k), value=format_stat(k, float(v) if pd.notna(v) else float("nan")))
        for k, v in stats.items()
    ]


def categorized_sections(
    stats: pd.Series,
    *,
    bench_stats: pd.Series | None = None,
) -> list[CategorySection]:
    """Group ``stats`` into sidebar sections keyed by :class:`Category`.

    Metrics with no entry in :data:`METRIC_INFO` are bucketed into a
    trailing ``Other`` section so unfamiliar keys stay visible.

    ``bench_stats`` — when provided, each section's metric rows include the
    benchmark value in ``bench_value``; the sidebar template renders it as
    a second column so strategy and benchmark sit side by side.
    """
    _hidden = {"periods", "start", "end"}
    by_cat: dict[Category, list[MetricRow]] = {c: [] for c in category_order()}
    other: list[MetricRow] = []
    for key, raw in stats.items():
        if str(key) in _hidden:
            continue
        if isinstance(raw, (pd.Timestamp, np.datetime64)):
            continue  # start/end timestamps aren't displayed as metric rows
        info = METRIC_INFO.get(str(key))
        try:
            val = float(raw) if pd.notna(raw) else float("nan")
        except (TypeError, ValueError):
            continue  # non-numeric metadata (strings etc.) don't belong here
        value_str = format_stat(str(key), val)
        bench_str = ""
        if bench_stats is not None and str(key) in bench_stats.index:
            bench_raw = bench_stats[str(key)]
            if not isinstance(bench_raw, (pd.Timestamp, np.datetime64)):
                try:
                    bench_val = float(bench_raw) if pd.notna(bench_raw) else float("nan")
                    bench_str = format_stat(str(key), bench_val)
                except (TypeError, ValueError):
                    pass
        row = MetricRow(
            key=str(key),
            label=info.label if info is not None else _label(str(key)),
            value=value_str,
            bench_value=bench_str,
            definition=info.definition if info is not None else "",
            formula=info.formula if info is not None else "",
            klass=_klass_for(str(key), val),
        )
        if info is None:
            other.append(row)
        else:
            by_cat[info.category].append(row)

    sections: list[CategorySection] = []
    for cat in category_order():
        rows = by_cat[cat]
        if rows:
            sections.append(CategorySection(category=cat.value, rows=rows))
    if other:
        sections.append(CategorySection(category="Other", rows=other))
    return sections
