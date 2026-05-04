"""``PatternCondition`` — entry/exit descriptor with intuitive presets.

A ``PatternCondition`` separates "what to look for" (the pattern feature)
from "how to act on it" (the entry / exit / target / stop rules).
Detectors ship a sensible preset so users can say
``HeadAndShoulders()`` and get the textbook breakout-and-target behaviour;
advanced users override via ``.override(...)`` or by passing a custom
condition to the indicator's constructor.

Mirrors the frozen-dataclass convention used by
:class:`fundcloud.strategies.scheduler.Cadence`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from fundcloud.features.patterns._enums import (
    EntryRule,
    ExitRule,
    StopMethod,
    TargetMethod,
    coerce,
)

__all__ = ["PatternCondition"]


@dataclass(frozen=True, slots=True)
class PatternCondition:
    """Entry/exit descriptor for a chart-pattern strategy.

    All fields use Enums (no string literals) — public APIs that accept
    ``str`` coerce at the boundary via :func:`._enums.coerce`.
    """

    entry_rule: EntryRule = EntryRule.ON_BREAKOUT
    exit_rule: ExitRule = ExitRule.TARGET_OR_STOP
    target_method: TargetMethod = TargetMethod.MEASURED_MOVE
    stop_method: StopMethod = StopMethod.BELOW_PIVOT
    time_stop_bars: int | None = None
    atr_window: int = 14
    atr_multiple: float = 2.0
    #: Used by ``StopMethod.FIXED_PCT`` — fraction of entry price.
    fixed_pct: float = 0.05
    #: Used by ``TargetMethod.FIB_1_618`` — multiplier of pattern_height.
    fib_target_multiple: float = 1.618

    def override(self, **kwargs: Any) -> PatternCondition:
        """Return a new ``PatternCondition`` with the given fields replaced.

        String values are coerced to their Enum types so callers can
        write ``cond.override(entry_rule="on_pullback")`` without
        importing the enum class.
        """
        valid_names = {f.name for f in fields(self)}
        unknown = set(kwargs) - valid_names
        if unknown:
            valid = ", ".join(sorted(valid_names))
            msg = f"unknown PatternCondition fields: {sorted(unknown)}; valid: {valid}"
            raise TypeError(msg)

        coerced: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "entry_rule":
                coerced[key] = coerce(value, EntryRule)
            elif key == "exit_rule":
                coerced[key] = coerce(value, ExitRule)
            elif key == "target_method":
                coerced[key] = coerce(value, TargetMethod)
            elif key == "stop_method":
                coerced[key] = coerce(value, StopMethod)
            else:
                coerced[key] = value
        return replace(self, **coerced)
