"""Tests for ``features.patterns._events`` projection helpers.

These cover the FORMATION / DECAY signal modes, the empty-events early
return, NaN-skip branches, the ``decay_bars`` validation, and the small
``_nan_or_float`` helper. The BREAKOUT mode and the happy-path
``build_events_frame`` are exercised indirectly by the accessor and
detector tests; here we focus on the branches codecov flagged as
uncovered on the FLG-1015 PR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import EVENTS_COLUMNS, Pattern, SignalMode
from fundcloud.features.patterns._enums import Direction, coerce
from fundcloud.features.patterns._events import (
    _nan_or_float,
    build_events_frame,
    empty_signal_series,
    events_to_signal,
)


def _index(n: int = 10) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _events_frame(
    *,
    index: pd.DatetimeIndex,
    formation_start: int,
    formation_end: int,
) -> pd.DataFrame:
    raw = [
        {
            "name": Pattern.HEAD_AND_SHOULDERS.value,
            "direction": Direction.BEARISH.value,
            "formation_start": formation_start,
            "formation_end": formation_end,
            "entry_price": 101.0,
            "breakout_price": 100.0,
            "quality": 0.7,
            "variant": None,
            "pivots": [],
            "features": {},
            "trend_lines": [],
        }
    ]
    return build_events_frame(raw, asset="AAA", index=index)


class TestEventsToSignal:
    def test_empty_events_returns_zero_series(self) -> None:
        idx = _index()
        empty = pd.DataFrame(columns=EVENTS_COLUMNS)
        out = events_to_signal(empty, index=idx, mode=SignalMode.BREAKOUT)
        assert out.shape == (len(idx),)
        assert (out == 0.0).all()

    def test_formation_mode_marks_inclusive_window(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=2, formation_end=5)
        out = events_to_signal(events, index=idx, mode=SignalMode.FORMATION)
        # Bars 2..5 inclusive should be 1.0; rest 0.0.
        expected = np.zeros(len(idx))
        expected[2:6] = 1.0
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_formation_mode_accepts_string_alias(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=0, formation_end=0)
        out = events_to_signal(events, index=idx, mode="formation")
        assert out.iloc[0] == 1.0
        assert (out.iloc[1:] == 0.0).all()

    def test_formation_mode_skips_nan_timestamps(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=1, formation_end=3)
        events.loc[0, "formation_start"] = pd.NaT
        out = events_to_signal(events, index=idx, mode=SignalMode.FORMATION)
        assert (out == 0.0).all()

    def test_formation_mode_skips_unknown_timestamps(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=1, formation_end=3)
        # Replace with timestamps that aren't on the output grid.
        outside = pd.Timestamp("1999-01-01")
        events.loc[0, "formation_start"] = outside
        out = events_to_signal(events, index=idx, mode=SignalMode.FORMATION)
        assert (out == 0.0).all()

    def test_decay_mode_linear_decay_then_zero(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=2, formation_end=4)
        out = events_to_signal(events, index=idx, mode=SignalMode.DECAY, decay_bars=4)
        # breakout_ts == formation_end == idx[4]; decay over 4 bars.
        expected = np.zeros(len(idx))
        for k in range(4):
            expected[4 + k] = 1.0 - k / 4
        np.testing.assert_allclose(out.to_numpy(), expected)

    def test_decay_mode_truncates_at_index_end(self) -> None:
        idx = _index(n=5)
        events = _events_frame(index=idx, formation_start=2, formation_end=4)
        out = events_to_signal(events, index=idx, mode=SignalMode.DECAY, decay_bars=10)
        # Only one bar fits past the breakout (idx[4]).
        assert out.iloc[4] == 1.0
        assert (out.iloc[:4] == 0.0).all()

    def test_decay_mode_keeps_higher_overlapping_value(self) -> None:
        idx = _index(n=10)
        # Two events that overlap; the later, fresher breakout should win
        # at its own bar (1.0) rather than be overwritten by the older
        # event's tail.
        raw = [
            {
                "name": Pattern.DOUBLE_TOP.value,
                "direction": Direction.BEARISH.value,
                "formation_start": 0,
                "formation_end": 2,
                "entry_price": 100.0,
                "breakout_price": 100.0,
                "quality": 1.0,
                "variant": None,
                "pivots": [],
                "features": {},
                "trend_lines": [],
            },
            {
                "name": Pattern.DOUBLE_TOP.value,
                "direction": Direction.BEARISH.value,
                "formation_start": 2,
                "formation_end": 4,
                "entry_price": 100.0,
                "breakout_price": 100.0,
                "quality": 1.0,
                "variant": None,
                "pivots": [],
                "features": {},
                "trend_lines": [],
            },
        ]
        events = build_events_frame(raw, asset="AAA", index=idx)
        out = events_to_signal(events, index=idx, mode=SignalMode.DECAY, decay_bars=4)
        # Bar 4 should be the second event's start (1.0), not the first
        # event's decayed tail (0.5).
        assert out.iloc[4] == pytest.approx(1.0)

    def test_decay_mode_rejects_nonpositive_window(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=1, formation_end=2)
        with pytest.raises(ValueError, match="decay_bars must be positive"):
            events_to_signal(events, index=idx, mode=SignalMode.DECAY, decay_bars=0)

    def test_decay_mode_skips_unknown_breakout_timestamps(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=1, formation_end=2)
        events.loc[0, "breakout_ts"] = pd.Timestamp("1999-01-01")
        out = events_to_signal(events, index=idx, mode=SignalMode.DECAY)
        assert (out == 0.0).all()

    def test_breakout_mode_skips_nan_timestamps(self) -> None:
        idx = _index()
        events = _events_frame(index=idx, formation_start=2, formation_end=3)
        events.loc[0, "breakout_ts"] = pd.NaT
        out = events_to_signal(events, index=idx, mode=SignalMode.BREAKOUT)
        assert (out == 0.0).all()


class TestSmallHelpers:
    def test_empty_signal_series_shape(self) -> None:
        idx = _index(n=4)
        out = empty_signal_series(idx)
        assert out.dtype == np.float64
        assert out.shape == (4,)
        assert (out == 0.0).all()
        assert out.index.equals(idx)

    def test_nan_or_float_passthrough(self) -> None:
        assert _nan_or_float(2) == 2.0
        assert _nan_or_float("3.5") == 3.5

    def test_nan_or_float_on_none(self) -> None:
        assert np.isnan(_nan_or_float(None))


class TestCoerceErrorPath:
    def test_coerce_unknown_value_lists_valid_options(self) -> None:
        with pytest.raises(ValueError, match="unknown SignalMode") as exc:
            coerce("nonsense", SignalMode)
        msg = str(exc.value)
        assert "'breakout'" in msg
        assert "'formation'" in msg
        assert "'decay'" in msg
