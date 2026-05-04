"""Smoke tests for the pattern visualization helpers.

We don't render the figures — just verify the trace count and basic
metadata so regressions in the trace assembly are caught.
"""

from __future__ import annotations

import fundcloud  # noqa: F401  — registers the .fc accessor
import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import EVENTS_COLUMNS, Direction, Pattern
from fundcloud.plots.patterns import plot_pattern_event, plot_patterns_overview


def _bars(n: int = 200, asset: str = "AAA") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, size=n))
    df = pd.DataFrame(
        {
            ("open", asset): close,
            ("high", asset): close + 0.5,
            ("low", asset): close - 0.5,
            ("close", asset): close,
            ("volume", asset): np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def _event(bars: pd.DataFrame, asset: str = "AAA") -> dict:
    fs = bars.index[80]
    fe = bars.index[100]
    return {
        "pattern": Pattern.DOUBLE_BOTTOM,
        "asset": asset,
        "direction": Direction.BULLISH,
        "formation_start": fs,
        "formation_end": fe,
        "breakout_ts": fe,
        "entry_price": 100.0,
        "breakout_price": 100.0,
        "target_price": 105.0,
        "stop_price": 95.0,
        "quality": 75.0,
        "variant": "STRICT_ADAM_ADAM",
        "pivots": [
            {"ts": bars.index[80], "price": 95.0, "kind": "LOW"},
            {"ts": bars.index[90], "price": 100.0, "kind": "HIGH"},
            {"ts": bars.index[100], "price": 95.0, "kind": "LOW"},
        ],
        "meta": {"trend_lines": [], "features": {}},
    }


def test_plot_pattern_event_returns_figure_with_traces() -> None:
    bars = _bars()
    fig = plot_pattern_event(_event(bars), bars, padding=10)
    # 1 candlestick + 1 high-pivot scatter + 1 low-pivot scatter +
    # 3 level lines (entry, target, stop) — entry+breakout dedup.
    assert len(fig.data) >= 5
    # Check at least the candlestick is present.
    candle_traces = [t for t in fig.data if t.type == "candlestick"]
    assert len(candle_traces) == 1


def test_plot_pattern_event_accepts_series() -> None:
    bars = _bars()
    series = pd.Series(_event(bars))
    fig = plot_pattern_event(series, bars, padding=10)
    assert fig.data  # non-empty


def test_plot_pattern_event_skips_levels_when_disabled() -> None:
    bars = _bars()
    no_levels = plot_pattern_event(_event(bars), bars, show_levels=False)
    with_levels = plot_pattern_event(_event(bars), bars, show_levels=True)
    assert len(with_levels.data) > len(no_levels.data)


def test_plot_patterns_overview_handles_empty_events() -> None:
    bars = _bars()
    empty = pd.DataFrame(columns=EVENTS_COLUMNS)
    fig = plot_patterns_overview(empty, bars, asset="AAA")
    # candlestick still rendered, no event markers
    assert len(fig.data) == 1
    assert fig.data[0].type == "candlestick"


def test_plot_patterns_overview_marks_events() -> None:
    bars = _bars()
    rows = [_event(bars), _event(bars)]
    rows[1]["breakout_ts"] = bars.index[120]
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)
    fig = plot_patterns_overview(events, bars, asset="AAA")
    # candlestick + at least one formation-shape polyline (lines+markers)
    shape_traces = [
        t for t in fig.data if t.type == "scatter" and "lines" in getattr(t, "mode", "")
    ]
    assert len(shape_traces) >= 1


def test_plot_asset_patterns_legend_groups_per_pattern() -> None:
    """The all-patterns single-asset chart should produce one legend
    entry per *pattern* (not per event), with the others belonging to
    the same legendgroup but hidden from the legend.
    """
    from fundcloud.plots.patterns import plot_asset_patterns

    bars = _bars(n=400)
    fig = plot_asset_patterns(bars, "AAA", min_quality=0.0)
    # Every formation-shape trace should be a Scatter with a legendgroup.
    shape_traces = [
        t for t in fig.data if t.type == "scatter" and "lines" in getattr(t, "mode", "")
    ]
    # On a synthetic series patterns may or may not fire — at least
    # the candlestick trace exists and the function ran without crashing.
    assert fig.data
    # If any shape traces present, exactly one per legendgroup is shown.
    if shape_traces:
        showlegend_per_group: dict[str, int] = {}
        for t in shape_traces:
            grp = getattr(t, "legendgroup", "")
            if t.showlegend:
                showlegend_per_group[grp] = showlegend_per_group.get(grp, 0) + 1
        for grp, count in showlegend_per_group.items():
            assert count == 1, f"legendgroup {grp} has {count} legend entries; expected 1"


def test_plot_pattern_event_horizon_adds_shapes() -> None:
    """When ``horizon`` is set, the figure should carry two layout shapes
    (the two vertical lines) and a vrect (the holding-window shade).
    """
    bars = _bars(n=400)
    no_horizon = plot_pattern_event(_event(bars), bars, horizon=None)
    with_horizon = plot_pattern_event(_event(bars), bars, horizon=20)
    n_shapes_no = len(no_horizon.layout.shapes or ())
    n_shapes_yes = len(with_horizon.layout.shapes or ())
    assert n_shapes_yes >= n_shapes_no + 2  # two new vertical lines
    # Annotation labels should mention "breakout" and "h=20".
    annotation_texts = [a.text for a in (with_horizon.layout.annotations or ())]
    assert any("breakout" in (t or "") for t in annotation_texts)
    assert any("h=20" in (t or "") for t in annotation_texts)


def test_accessor_plot_pattern_event_returns_figure() -> None:
    bars = _bars()
    fig = bars.fc.plot_pattern_event(_event(bars))
    assert fig.data


def test_accessor_plot_patterns_runs_indicator() -> None:
    """``bars.fc.plot_patterns`` should accept Pattern + asset and return a Figure."""
    bars = _bars(n=400)
    fig = bars.fc.plot_patterns(Pattern.DOUBLE_BOTTOM, asset="AAA")
    # Always at least the candlestick; events on this synthetic series may
    # be zero, so we don't require markers.
    assert fig.data
    assert fig.data[0].type == "candlestick"


def test_plot_pattern_event_rejects_flat_dataframe() -> None:
    flat = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(TypeError, match=r"MultiIndex"):
        plot_pattern_event(
            {
                "asset": "X",
                "formation_start": flat.index[0],
                "formation_end": flat.index[1],
                "breakout_ts": flat.index[1],
                "pivots": [],
                "meta": {},
                "direction": Direction.BULLISH,
                "pattern": Pattern.DOUBLE_BOTTOM,
                "entry_price": 1.0,
                "breakout_price": 1.0,
                "target_price": float("nan"),
                "stop_price": float("nan"),
                "quality": 0.0,
                "variant": None,
            },
            flat,
        )
