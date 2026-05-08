"""Tests for the pattern methods on ``DataFrameAccessor``.

Each method is a thin one-liner over the indicator / metrics layer; we
just lock the API contract: enum-or-string acceptance, expected output
shape, and clean error messages on unknown patterns.
"""

from __future__ import annotations

import fundcloud  # noqa: F401  — registers the .fc accessor
import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import EVENTS_COLUMNS, Pattern


def _make_bars(n: int = 200) -> pd.DataFrame:
    """Tiny single-asset MultiIndex Bars frame."""
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, size=n)))
    high = close * (1.0 + rng.uniform(0.001, 0.005, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.005, size=n))
    df = pd.DataFrame(
        {
            ("open", "AAA"): close,
            ("high", "AAA"): high,
            ("low", "AAA"): low,
            ("close", "AAA"): close,
            ("volume", "AAA"): np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def test_list_patterns_returns_all_nine_enum_values() -> None:
    bars = _make_bars()
    listed = bars.fc.list_patterns()
    assert len(listed) == 9
    assert set(listed) == set(Pattern)


def test_detect_pattern_accepts_enum_or_string() -> None:
    bars = _make_bars()
    via_str = bars.fc.detect_pattern("double_bottom")
    via_enum = bars.fc.detect_pattern(Pattern.DOUBLE_BOTTOM)
    assert via_str.shape == via_enum.shape
    assert list(via_str.columns) == list(via_enum.columns) == ["AAA"]


def test_pattern_events_returns_canonical_schema() -> None:
    bars = _make_bars()
    events = bars.fc.pattern_events(Pattern.DOUBLE_BOTTOM)
    assert list(events.columns) == list(EVENTS_COLUMNS)


def test_evaluate_pattern_panel_shape_matches_horizons() -> None:
    bars = _make_bars(n=400)
    panel = bars.fc.evaluate_pattern(Pattern.DOUBLE_BOTTOM, horizons=(5, 10, 20))
    assert panel.index.tolist() == [5, 10, 20]
    assert "hit_rate" in panel.columns
    assert "baseline_hit" in panel.columns


def test_unknown_pattern_raises_with_helpful_message() -> None:
    bars = _make_bars()
    with pytest.raises(ValueError, match=r"unknown pattern.*valid:"):
        bars.fc.detect_pattern("not_a_real_pattern")


def test_detect_pattern_requires_bars_frame() -> None:
    """A flat-column DataFrame should be rejected at the accessor boundary."""
    flat = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises((TypeError, ValueError)):
        flat.fc.detect_pattern("double_bottom")
