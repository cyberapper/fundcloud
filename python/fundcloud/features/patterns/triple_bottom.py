"""Triple Bottom chart-pattern indicator.

Bullish reversal: three troughs at approximately the same level with two
intervening peaks. Distinguished from Inverse Head-and-Shoulders by all
three troughs being roughly equal. Confirmation requires a close above
the *highest* intervening peak (Bulkowski's confirmation level).

The Rust detector enforces:

* Sequence ``L-H-L-H-L``.
* Each trough within 2% of the trio's mean (default).
* Pattern height ≥ 2% (highest peak above the mean trough).
* Formation ≥ 10 bars between the first and last trough.

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

__all__ = ["TripleBottom"]


@register_indicator(Pattern.TRIPLE_BOTTOM.value)
class TripleBottom(PatternIndicator):
    """Bullish "Triple Bottom" reversal."""

    pattern_name = "triple_bottom"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
    detector_param_keys = ("trough_tolerance", "min_peak_height", "min_bar_count")
    default_params = {
        **PatternIndicator.default_params,
        "trough_tolerance": 0.02,
        "min_peak_height": 0.02,
        "min_bar_count": 10,
    }
