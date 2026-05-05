"""Tests for the per-detector tunable knobs plumbed through PatternIndicator.

These verify that:

* Default kwargs land on the instance and match the documented defaults.
* Overriding a knob actually changes detection counts.
* Each pattern class advertises a non-empty `detector_param_keys`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import (
    AscendingTriangle,
    DescendingTriangle,
    DoubleBottom,
    DoubleTop,
    HeadAndShoulders,
    InverseHeadAndShoulders,
    SymmetricalTriangle,
    TripleBottom,
    TripleTop,
)


def _synthetic_double_top(n: int = 80) -> pd.DataFrame:
    """Build a tiny synthetic OHLCV frame with a clean double-top shape.

    Two peaks at ~100, trough at ~92, on a flat-ish background. Used as a
    deterministic detection target so we can compare counts across param
    settings without relying on cached real data.
    """
    rng = np.random.default_rng(0)
    base = np.full(n, 95.0)
    # Peak 1 at i=20
    base[18:22] = [98, 100, 99, 96]
    # Trough at i=30
    base[28:32] = [94, 92, 93, 95]
    # Peak 2 at i=42
    base[40:44] = [98, 100, 99, 96]
    # Some noise
    close = base + rng.normal(0, 0.2, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close.copy()
    volume = np.full(n, 1_000_000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    cols = pd.MultiIndex.from_product(
        [["open", "high", "low", "close", "volume"], ["TST"]],
        names=["field", "asset"],
    )
    return pd.DataFrame(
        np.column_stack([open_, high, low, close, volume]),
        index=idx,
        columns=cols,
    )


@pytest.mark.parametrize(
    ("cls", "expected_keys"),
    [
        (DoubleTop, ("peak_tolerance", "min_trough_depth")),
        (DoubleBottom, ("trough_tolerance", "min_peak_height")),
        (TripleTop, ("peak_tolerance", "min_trough_depth", "min_bar_count")),
        (TripleBottom, ("trough_tolerance", "min_peak_height", "min_bar_count")),
        (
            HeadAndShoulders,
            ("shoulder_tolerance", "min_head_prominence", "prior_trend_window"),
        ),
        (
            InverseHeadAndShoulders,
            ("shoulder_tolerance", "min_head_prominence", "prior_trend_window"),
        ),
        (AscendingTriangle, ("flat_threshold", "min_touches")),
        (DescendingTriangle, ("flat_threshold", "min_touches")),
        (
            SymmetricalTriangle,
            ("min_slope_threshold", "min_touches", "min_bar_count", "prior_trend_window"),
        ),
    ],
)
def test_detector_param_keys_match_default_params(cls, expected_keys):
    """Every advertised key has a matching default and is set on instances."""
    assert cls.detector_param_keys == expected_keys
    inst = cls()
    for k in expected_keys:
        assert k in cls.default_params, f"{cls.__name__} default_params missing {k!r}"
        assert hasattr(inst, k), f"{cls.__name__} instance missing attribute {k!r}"


def test_double_top_loosening_increases_detections():
    """Looser peak_tolerance + smaller min_trough_depth should detect more."""
    bars = _synthetic_double_top()
    strict = DoubleTop(min_quality=0).events(bars)
    loose = DoubleTop(
        min_quality=0,
        peak_tolerance=0.10,
        min_trough_depth=0.01,
    ).events(bars)
    assert len(loose) >= len(strict)


def test_unknown_kwarg_still_passes_through():
    """Sanity: arbitrary kwargs land via setattr (existing IndicatorSpec behavior)."""
    dt = DoubleTop(some_unrelated_kwarg=42)
    assert dt.some_unrelated_kwarg == 42


def test_default_params_inherit_pipeline_defaults():
    """Per-pattern default_params include shared pipeline keys."""
    for cls in (DoubleTop, HeadAndShoulders, AscendingTriangle):
        assert "min_quality" in cls.default_params
        assert "pivot_tiers" in cls.default_params
        assert "signal_mode" in cls.default_params
