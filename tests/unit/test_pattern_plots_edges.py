"""Edge-case coverage for ``fundcloud.plots.patterns``.

The happy paths are pinned in ``test_pattern_plots.py``; this file fills
the optional-branch logic codecov flagged on the FLG-1015 PR — empty
inputs, NaN levels, off-grid timestamps, single-side pivot lists, the
trend-lines sub-renderer, the horizon-shading caps in
``plot_asset_patterns`` and ``plot_patterns_overview``, and the small
``_event_to_dict`` / ``_bar_offset_ts`` helpers.

Like the other plot tests, we don't render — we just assert on figure
structure (trace counts, layout shapes, annotations, titles).
"""

from __future__ import annotations

import fundcloud  # noqa: F401  — registers .fc accessor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fundcloud.features.patterns import EVENTS_COLUMNS, Direction, Pattern
from fundcloud.plots.patterns import (
    _add_horizon_markers,
    _add_horizon_markers_silent,
    _add_levels,
    _add_pattern_shape,
    _add_pivots,
    _add_trend_lines,
    _bar_offset_ts,
    _event_to_dict,
    _slice_window,
    plot_asset_patterns,
    plot_pattern_event,
    plot_patterns_overview,
)


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


def _event(
    bars: pd.DataFrame,
    *,
    asset: str = "AAA",
    direction: Direction = Direction.BULLISH,
    pattern: Pattern = Pattern.DOUBLE_BOTTOM,
    formation_idx: tuple[int, int] = (80, 100),
    target: float | None = 105.0,
    stop: float | None = 95.0,
    pivots: list[dict] | None = None,
    trend_lines: list[dict] | None = None,
    variant: str | None = "STRICT_ADAM_ADAM",
    quality: float = 75.0,
) -> dict:
    fs_idx, fe_idx = formation_idx
    fs = bars.index[fs_idx]
    fe = bars.index[fe_idx]
    if pivots is None:
        pivots = [
            {"ts": bars.index[fs_idx], "price": 95.0, "kind": "LOW"},
            {"ts": bars.index[(fs_idx + fe_idx) // 2], "price": 100.0, "kind": "HIGH"},
            {"ts": bars.index[fe_idx], "price": 95.0, "kind": "LOW"},
        ]
    return {
        "pattern": pattern,
        "asset": asset,
        "direction": direction,
        "formation_start": fs,
        "formation_end": fe,
        "breakout_ts": fe,
        "entry_price": 100.0,
        "breakout_price": 100.0,
        "target_price": float("nan") if target is None else target,
        "stop_price": float("nan") if stop is None else stop,
        "quality": quality,
        "variant": variant,
        "pivots": pivots,
        "meta": {"trend_lines": trend_lines or [], "features": {}},
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestSliceWindow:
    def test_off_grid_timestamps_use_timedelta_fallback(self) -> None:
        bars = _bars()
        ohlc = bars.xs("AAA", level=-1, axis=1)
        # Pick timestamps deliberately *not* in the grid (12-hour offset).
        off1 = ohlc.index[20] + pd.Timedelta(hours=12)
        off2 = ohlc.index[40] + pd.Timedelta(hours=12)
        out = _slice_window(ohlc, off1, off2, padding=5)
        # Fallback uses pandas label-slicing; we just verify it returned a
        # sane slice (covers the formation interval) without crashing.
        assert not out.empty
        assert out.index[0] <= ohlc.index[40]


class TestEventToDict:
    def test_rejects_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Series or dict"):
            _event_to_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_passes_dict_through(self) -> None:
        d = {"asset": "AAA"}
        assert _event_to_dict(d) is d


class TestAddPivots:
    def test_empty_pivots_returns_quietly(self) -> None:
        fig = go.Figure()
        _add_pivots(fig, [])
        assert len(fig.data) == 0

    def test_only_high_pivots_skips_low_trace(self) -> None:
        fig = go.Figure()
        _add_pivots(
            fig,
            [
                {"ts": pd.Timestamp("2020-01-01"), "price": 105.0, "kind": "HIGH"},
                {"ts": pd.Timestamp("2020-01-02"), "price": 106.0, "kind": "HIGH"},
            ],
        )
        names = [t.name for t in fig.data]
        assert "High pivot" in names
        assert "Low pivot" not in names


class TestAddLevels:
    def test_nan_levels_are_skipped(self) -> None:
        fig = go.Figure()
        ev = {
            "entry_price": float("nan"),
            "breakout_price": float("nan"),
            "target_price": float("nan"),
            "stop_price": float("nan"),
        }
        _add_levels(fig, ev, x_range=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")))
        assert len(fig.data) == 0


class TestAddTrendLines:
    def test_empty_trend_lines_returns_quietly(self) -> None:
        bars = _bars()
        ohlc = bars.xs("AAA", level=-1, axis=1)
        fig = go.Figure()
        _add_trend_lines(fig, [], ohlc, bars, "AAA")
        assert len(fig.data) == 0

    def test_renders_trend_line_within_window(self) -> None:
        bars = _bars(n=150)
        ohlc = bars.xs("AAA", level=-1, axis=1).iloc[40:80]
        fig = go.Figure()
        _add_trend_lines(
            fig,
            [
                {
                    "slope": 0.1,
                    "intercept": 100.0,
                    "start_index": 30,
                    "end_index": 90,
                }
            ],
            ohlc,
            bars,
            "AAA",
        )
        assert any(t.name == "Trend line" for t in fig.data)

    def test_trend_line_outside_window_is_skipped(self) -> None:
        bars = _bars(n=150)
        ohlc_window = bars.xs("AAA", level=-1, axis=1).iloc[40:60]
        fig = go.Figure()
        _add_trend_lines(
            fig,
            [
                # Entirely before the window.
                {
                    "slope": 0.0,
                    "intercept": 100.0,
                    "start_index": 0,
                    "end_index": 10,
                },
                # Out-of-range start_index → IndexError fallback skip.
                {
                    "slope": 0.0,
                    "intercept": 100.0,
                    "start_index": 9_999,
                    "end_index": 10_001,
                },
            ],
            ohlc_window,
            bars,
            "AAA",
        )
        assert len(fig.data) == 0


class TestAddPatternShape:
    def test_too_few_pivots_is_no_op(self) -> None:
        fig = go.Figure()
        _add_pattern_shape(
            fig,
            {
                "pivots": [{"ts": pd.Timestamp("2020-01-01"), "price": 1.0, "kind": "HIGH"}],
                "pattern": Pattern.DOUBLE_TOP,
                "direction": Direction.BEARISH,
                "breakout_ts": pd.Timestamp("2020-01-02"),
                "quality": float("nan"),
                "variant": None,
                "asset": "AAA",
            },
        )
        assert len(fig.data) == 0


class TestBarOffsetTs:
    def test_unknown_breakout_returns_none(self) -> None:
        bars = _bars()
        out = _bar_offset_ts(bars, "AAA", pd.Timestamp("1999-01-01", tz="UTC"), horizon=5)
        assert out is None

    def test_offset_past_end_returns_none(self) -> None:
        bars = _bars(n=20)
        last = bars.index[-1]
        out = _bar_offset_ts(bars, "AAA", last, horizon=5)
        assert out is None

    def test_normal_offset_returns_timestamp(self) -> None:
        bars = _bars(n=50)
        out = _bar_offset_ts(bars, "AAA", bars.index[10], horizon=5)
        assert out == bars.index[15]


class TestAddHorizonMarkers:
    def test_nonpositive_horizon_is_no_op(self) -> None:
        bars = _bars()
        fig = go.Figure()
        _add_horizon_markers(fig, _event(bars), bars, horizon=0)
        assert not fig.layout.shapes

    def test_horizon_past_series_end_is_no_op(self) -> None:
        bars = _bars(n=120)
        fig = go.Figure()
        # breakout near the end + huge horizon → horizon_end is None.
        ev = _event(bars, formation_idx=(100, 115))
        _add_horizon_markers(fig, ev, bars, horizon=999)
        assert not fig.layout.shapes


class TestAddHorizonMarkersSilent:
    def test_offset_past_end_is_no_op(self) -> None:
        bars = _bars(n=120)
        fig = go.Figure()
        _add_horizon_markers_silent(fig, _event(bars, formation_idx=(100, 115)), bars, horizon=999)
        # vrect would normally land in layout.shapes; with no horizon end
        # the function should bail before adding it.
        assert not fig.layout.shapes

    def test_renders_vrect_for_bearish_direction(self) -> None:
        bars = _bars(n=200)
        ev = _event(bars, direction=Direction.BEARISH, formation_idx=(50, 70))
        fig = go.Figure()
        _add_horizon_markers_silent(fig, ev, bars, horizon=10)
        assert fig.layout.shapes  # vrect added


# ---------------------------------------------------------------------------
# Top-level entrypoints
# ---------------------------------------------------------------------------


class TestPlotPatternEventErrors:
    def test_empty_window_raises(self) -> None:
        bars = _bars(n=200)
        ev = _event(bars)
        # Force formation timestamps far outside the bar range; the
        # KeyError-fallback in ``_slice_window`` then yields an empty
        # slice for ``plot_pattern_event``.
        ev["formation_start"] = pd.Timestamp("1990-01-01", tz="UTC")
        ev["formation_end"] = pd.Timestamp("1990-02-01", tz="UTC")
        with pytest.raises(ValueError, match="no bars in plotting window"):
            plot_pattern_event(ev, bars, padding=5, horizon=None)

    def test_renders_with_trend_lines(self) -> None:
        bars = _bars(n=200)
        ev = _event(
            bars,
            trend_lines=[{"slope": 0.0, "intercept": 100.0, "start_index": 60, "end_index": 130}],
        )
        fig = plot_pattern_event(ev, bars, padding=10)
        assert any(t.name == "Trend line" for t in fig.data)


class TestPlotAssetPatterns:
    def test_all_nan_asset_raises(self) -> None:
        # Asset present in the columns but every OHLC row is NaN — the
        # ``dropna`` step in ``_select_asset_ohlc`` empties the slice and
        # ``plot_asset_patterns`` should bail with a clear error.
        bars = _bars()
        for field in ("open", "high", "low", "close"):
            bars[(field, "AAA")] = float("nan")
        with pytest.raises(ValueError, match="no bars for asset"):
            plot_asset_patterns(bars, "AAA")

    def test_explicit_pattern_iterable_with_strings(self) -> None:
        bars = _bars(n=400)
        fig = plot_asset_patterns(
            bars,
            "AAA",
            patterns=["double_top", Pattern.DOUBLE_BOTTOM],
            min_quality=0.0,
            horizon=20,
            show_horizon_for_top=2,
        )
        assert fig.data
        # Title carries the horizon annotation.
        title_text = fig.layout.title.text or ""
        assert "h=20" in title_text


class TestPlotPatternsOverview:
    def test_max_events_caps_rows(self) -> None:
        bars = _bars(n=200)
        rows = []
        for i in range(5):
            rows.append(_event(bars, formation_idx=(60 + i * 10, 80 + i * 10)))
        events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)
        fig = plot_patterns_overview(events, bars, asset="AAA", max_events=2, horizon=10)
        # 1 candlestick + 2 formation polylines (one per kept event); the
        # second event should also pick up a horizon shade in layout.
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(scatter_traces) == 2

    def test_single_pattern_title_uses_pattern_name(self) -> None:
        bars = _bars(n=200)
        events = pd.DataFrame(
            [_event(bars, pattern=Pattern.HEAD_AND_SHOULDERS)], columns=EVENTS_COLUMNS
        )
        fig = plot_patterns_overview(events, bars, asset="AAA", horizon=15)
        title_text = fig.layout.title.text or ""
        assert "head_and_shoulders" in title_text
        assert "h=15" in title_text
