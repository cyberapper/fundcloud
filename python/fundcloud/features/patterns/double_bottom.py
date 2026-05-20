"""Double Bottom chart-pattern indicator.

Bullish reversal: two troughs at approximately the same level separated
by an intervening peak. Breakout above the peak (the neckline) confirms
the pattern; measured move targets the neckline plus the pattern height.

The Rust detector enforces:

* Sequence ``L-H-L``.
* Troughs within 1.5% of each other (default).
* Peak height ≥ 3% above the average trough.
* Formation ≥ 5 bars between the troughs.

Each detection carries a Bulkowski variant tag in
``events["variant"]`` — see :class:`DoubleTop` for the meaning of the
``STRICT_*`` / ``WEAK_*`` and ``ADAM`` / ``EVE`` segments.

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

__all__ = ["DoubleBottom"]


@register_indicator(Pattern.DOUBLE_BOTTOM.value)
class DoubleBottom(PatternIndicator):
    """Bullish "Double Bottom" reversal."""

    pattern_name = "double_bottom"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
    detector_param_keys = ("trough_tolerance", "min_peak_height")
    default_params = {
        **PatternIndicator.default_params,
        "min_quality": 75.0,
        "trough_tolerance": 0.015,
        "min_peak_height": 0.03,
    }
