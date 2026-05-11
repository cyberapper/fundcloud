"""Tests for ``features.patterns._scan_all``.

Covers the registry filter, subset selection, per-pattern condition /
params overrides, error paths for unknown names, the empty-bars no-op
shape, and the plug-in extensibility contract (a custom-registered
``PatternIndicator`` subclass appears in scan results without any code
change to ``scan_all_patterns`` itself).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.indicators.base import _REGISTRY, register_indicator
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    Pattern,
    PatternCondition,
    PatternIndicator,
    registered_pattern_indicators,
    scan_all_patterns,
)
from fundcloud.features.patterns._enums import EntryRule


def _synthetic_bars(n: int = 80) -> pd.DataFrame:
    """Synthetic OHLCV with a clean double-top — same fixture style as the
    rest of the pattern test suite (see ``test_pattern_detector_params``)."""
    rng = np.random.default_rng(0)
    base = np.full(n, 95.0)
    base[18:22] = [98, 100, 99, 96]
    base[28:32] = [94, 92, 93, 95]
    base[40:44] = [98, 100, 99, 96]
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


def _flat_bars(n: int = 80) -> pd.DataFrame:
    """Bars with no formations — every detector should return zero events."""
    close = np.full(n, 100.0)
    high = close + 0.01
    low = close - 0.01
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    cols = pd.MultiIndex.from_product(
        [["open", "high", "low", "close", "volume"], ["TST"]],
        names=["field", "asset"],
    )
    return pd.DataFrame(
        np.column_stack([close, high, low, close, np.full(n, 1.0)]),
        index=idx,
        columns=cols,
    )


def test_registered_pattern_indicators_returns_all_builtins():
    registry = registered_pattern_indicators()
    expected = {p.value for p in Pattern}
    assert set(registry.keys()) == expected
    for cls in registry.values():
        assert issubclass(cls, PatternIndicator)


def test_scan_all_returns_canonical_schema():
    bars = _synthetic_bars()
    events = scan_all_patterns(bars)
    assert list(events.columns) == list(EVENTS_COLUMNS)


def test_scan_all_concatenates_across_detectors():
    bars = _synthetic_bars()
    events = scan_all_patterns(bars)
    # The synthetic fixture is engineered for double-top; we only assert
    # at least one detection landed and that the pattern column is populated.
    assert not events.empty
    patterns_seen = {_pattern_value(p) for p in events["pattern"]}
    assert patterns_seen.issubset({p.value for p in Pattern})


def test_scan_all_subset_with_enum():
    bars = _synthetic_bars()
    events = scan_all_patterns(bars, patterns=[Pattern.DOUBLE_TOP])
    if not events.empty:
        assert {_pattern_value(p) for p in events["pattern"]} == {Pattern.DOUBLE_TOP.value}


def test_scan_all_subset_with_strings():
    bars = _synthetic_bars()
    events = scan_all_patterns(bars, patterns=["double_top", "double_bottom"])
    if not events.empty:
        seen = {_pattern_value(p) for p in events["pattern"]}
        assert seen.issubset({"double_top", "double_bottom"})


def test_scan_all_unknown_pattern_raises_keyerror():
    bars = _synthetic_bars()
    with pytest.raises(KeyError):
        scan_all_patterns(bars, patterns=["not_a_real_pattern"])


def test_scan_all_unknown_condition_key_raises_keyerror():
    bars = _synthetic_bars()
    bogus = {"not_a_real_pattern": PatternCondition(entry_rule=EntryRule.ON_BREAKOUT)}
    with pytest.raises(KeyError):
        scan_all_patterns(bars, conditions=bogus)


def test_scan_all_unknown_params_key_raises_keyerror():
    bars = _synthetic_bars()
    with pytest.raises(KeyError):
        scan_all_patterns(bars, params={"not_a_real_pattern": {"min_quality": 0.0}})


def test_scan_all_empty_when_no_detections():
    bars = _flat_bars()
    events = scan_all_patterns(bars)
    assert events.empty
    assert list(events.columns) == list(EVENTS_COLUMNS)


def test_scan_all_per_pattern_params_override():
    """Lowering min_quality on a single pattern routes the kwarg correctly.

    Doesn't assert detection counts (those depend on detector internals);
    just verifies the override path doesn't blow up and the schema holds.
    """
    bars = _synthetic_bars()
    events = scan_all_patterns(
        bars,
        patterns=[Pattern.DOUBLE_TOP],
        params={Pattern.DOUBLE_TOP: {"min_quality": 0.0}},
    )
    assert list(events.columns) == list(EVENTS_COLUMNS)


def test_scan_all_picks_up_plugin_pattern():
    """Custom detector registered at runtime joins scan_all automatically.

    This is the plug-in contract — the user pays nothing to integrate
    beyond decorating their class with ``@register_indicator``.
    """
    custom_name = "_test_dummy_pattern"

    @register_indicator(custom_name)
    class _DummyPattern(PatternIndicator):
        pattern_name = custom_name

        def _scan(self, fields, index, *, asset):
            return pd.DataFrame(columns=list(EVENTS_COLUMNS))

    try:
        registry = registered_pattern_indicators()
        assert custom_name in registry
        bars = _flat_bars()
        events = scan_all_patterns(bars, patterns=[custom_name])
        assert list(events.columns) == list(EVENTS_COLUMNS)
    finally:
        # Clean up registry mutation so other tests stay deterministic.
        _REGISTRY.pop(custom_name, None)


def _pattern_value(value):
    return value.value if hasattr(value, "value") else str(value)
