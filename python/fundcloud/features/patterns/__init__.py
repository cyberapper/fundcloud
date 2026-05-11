"""Chart-pattern feature subpackage.

Exposes the public enums (:class:`Pattern`, :class:`Direction`,
:class:`SignalMode`, :class:`EntryRule`, :class:`ExitRule`,
:class:`TargetMethod`, :class:`StopMethod`), the :class:`PatternCondition`
descriptor, the :class:`PatternIndicator` base, and one indicator class
per registered pattern.

Example:
    >>> from fundcloud.features.patterns import HeadAndShoulders, Pattern
    >>> indicator = HeadAndShoulders()                      # doctest: +SKIP
    >>> signals = indicator.fit_transform(bars)             # doctest: +SKIP
    >>> events = indicator.events(bars)                     # doctest: +SKIP
    >>> Pattern.HEAD_AND_SHOULDERS.value
    'head_and_shoulders'
"""

from __future__ import annotations

from fundcloud.features.patterns._apply_condition import apply_condition
from fundcloud.features.patterns._base import (
    DEFAULT_PIVOT_TIERS,
    PIVOT_TIER_LONG,
    PIVOT_TIER_MEDIUM,
    PIVOT_TIER_SHORT,
    PatternIndicator,
)
from fundcloud.features.patterns._condition import PatternCondition
from fundcloud.features.patterns._enums import (
    Direction,
    EntryRule,
    ExitRule,
    Pattern,
    SignalMode,
    StopMethod,
    TargetMethod,
    coerce,
)
from fundcloud.features.patterns._events import (
    EVENTS_COLUMNS,
    build_events_frame,
    events_to_signal,
)
from fundcloud.features.patterns._scan_all import (
    registered_pattern_indicators,
    scan_all_patterns,
)
from fundcloud.features.patterns.ascending_triangle import AscendingTriangle
from fundcloud.features.patterns.descending_triangle import DescendingTriangle
from fundcloud.features.patterns.double_bottom import DoubleBottom
from fundcloud.features.patterns.double_top import DoubleTop
from fundcloud.features.patterns.head_and_shoulders import HeadAndShoulders
from fundcloud.features.patterns.inverse_head_and_shoulders import InverseHeadAndShoulders
from fundcloud.features.patterns.symmetrical_triangle import SymmetricalTriangle
from fundcloud.features.patterns.triple_bottom import TripleBottom
from fundcloud.features.patterns.triple_top import TripleTop

__all__ = [
    "DEFAULT_PIVOT_TIERS",
    "EVENTS_COLUMNS",
    "PIVOT_TIER_LONG",
    "PIVOT_TIER_MEDIUM",
    "PIVOT_TIER_SHORT",
    "AscendingTriangle",
    "DescendingTriangle",
    "Direction",
    "DoubleBottom",
    "DoubleTop",
    "EntryRule",
    "ExitRule",
    "HeadAndShoulders",
    "InverseHeadAndShoulders",
    "Pattern",
    "PatternCondition",
    "PatternIndicator",
    "SignalMode",
    "StopMethod",
    "SymmetricalTriangle",
    "TargetMethod",
    "TripleBottom",
    "TripleTop",
    "apply_condition",
    "build_events_frame",
    "coerce",
    "events_to_signal",
    "registered_pattern_indicators",
    "scan_all_patterns",
]
