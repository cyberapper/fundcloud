"""Symmetrical Triangle chart-pattern indicator.

Continuation: a falling upper trend line (resistance) combined with a
rising lower trend line (support). Direction is inferred from the prior
trend — the pattern resolves :class:`Direction.BULLISH` after an uptrend,
:class:`Direction.BEARISH` after a downtrend (and bullish as a fallback
when the prior trend is exactly flat).

Because the channel collapses to zero near the apex, a fraction-of-channel
tolerance vanishes; the Rust detector switches to an absolute-price
tolerance equal to 5% of the starting gap when validating that bars stay
within the triangle.

The Rust detector enforces:

* Upper line normalised slope ≤ -``min_slope_threshold`` (falling
  resistance).
* Lower line normalised slope ≥ +``min_slope_threshold`` (rising support).
* Lines must converge (end gap < start gap, both positive).
* Every bar inside the formation stays within the channel under the
  absolute-price tolerance.
* Formation ≥ 10 bars.

Overlapping detections are deduplicated, keeping the one with more pivots.

See :mod:`fundcloud.features.patterns._base` for the shared input/output
contract.
"""

from __future__ import annotations

from fundcloud.features.indicators.base import register_indicator
from fundcloud.features.patterns._base import PatternIndicator
from fundcloud.features.patterns._condition import PatternCondition
from fundcloud.features.patterns._enums import (
    EntryRule,
    ExitRule,
    Pattern,
    StopMethod,
    TargetMethod,
)

__all__ = ["SymmetricalTriangle"]


@register_indicator(Pattern.SYMMETRICAL_TRIANGLE.value)
class SymmetricalTriangle(PatternIndicator):
    """Symmetrical Triangle continuation (direction inferred from prior trend)."""

    pattern_name = "symmetrical_triangle"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
