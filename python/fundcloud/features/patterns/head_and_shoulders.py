"""Head and Shoulders chart-pattern indicator.

Bearish reversal: a peak (head) flanked by two lower peaks (shoulders) of
similar height, sitting above a neckline drawn through the two
intervening lows. The breakout below the neckline projects a measured
move target equal to the head-to-neckline distance below the neckline.

The Rust detector enforces:

* Sequence ``H-L-H-L-H``.
* Head strictly above both shoulders.
* Shoulders within 10% of each other (default).
* Head prominence ≥ 3% above the average shoulder.
* Formation ≥ 8 bars.
* Prior trend slope > 0 (uptrend before reversal).

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

__all__ = ["HeadAndShoulders"]


@register_indicator(Pattern.HEAD_AND_SHOULDERS.value)
class HeadAndShoulders(PatternIndicator):
    """Bearish "Head and Shoulders" reversal."""

    pattern_name = "head_and_shoulders"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
