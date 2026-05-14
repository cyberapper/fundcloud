"""Public enum surface for chart-pattern detection.

Each enum subclasses ``(str, Enum)`` so values round-trip cleanly through
JSON, kwargs, and the Rust `_core` boundary, matching the existing
:class:`fundcloud.optimize._LocalRiskMeasure` and
:class:`fundcloud.reports.metric_info.Category` conventions.

Public APIs accept ``EnumType | str`` for ad-hoc scripting; the
:func:`coerce` helper does the str→Enum conversion at the boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

__all__ = [
    "Direction",
    "EntryRule",
    "ExitRule",
    "Pattern",
    "SignalMode",
    "StopMethod",
    "TargetMethod",
    "coerce",
]


class Pattern(str, Enum):
    """Registered chart-pattern identifiers.

    Patterns are pure shape labels — no direction is implied by the name.
    Whether to trade or grade a given event as bullish, bearish, or neutral
    is a *strategy choice* expressed via :class:`PatternCondition.direction`
    and :class:`PatternStrategy(direction=...)`. The library does not carry
    a classical-TA prior.
    """

    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"


class Direction(str, Enum):
    """Direction a strategy assigns to a pattern (caller-supplied, never
    inferred by the library)."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalMode(str, Enum):
    """How an event list projects onto the per-bar signal panel."""

    BREAKOUT = "breakout"
    FORMATION = "formation"
    DECAY = "decay"


class EntryRule(str, Enum):
    """Rule for when a strategy opens a position on a detected pattern."""

    ON_BREAKOUT = "on_breakout"
    ON_FORMATION_COMPLETE = "on_formation_complete"
    ON_PULLBACK = "on_pullback"


class ExitRule(str, Enum):
    """Rule for when a strategy closes the position."""

    TARGET_OR_STOP = "target_or_stop"
    TIME_STOP = "time_stop"
    TRAILING_STOP = "trailing_stop"


class TargetMethod(str, Enum):
    """How target_price is computed from the pattern's geometry."""

    MEASURED_MOVE = "measured_move"
    FIB_1_618 = "fib_1_618"
    FIXED_ATR = "fixed_atr"


class StopMethod(str, Enum):
    """How stop_price is computed from the pattern's geometry."""

    BELOW_PIVOT = "below_pivot"
    ATR_MULTIPLE = "atr_multiple"
    FIXED_PCT = "fixed_pct"


_E = TypeVar("_E", bound=Enum)


def coerce(value: _E | str, enum_cls: type[_E]) -> _E:
    """Coerce ``value`` to ``enum_cls`` — accepts both Enum and string forms.

    Raises ``ValueError`` with a helpful message that lists every valid
    value when ``value`` does not resolve.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid = ", ".join(repr(member.value) for member in enum_cls)
        msg = f"unknown {enum_cls.__name__}: {value!r}; valid: {valid}"
        raise ValueError(msg) from exc
