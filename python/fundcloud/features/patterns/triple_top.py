"""Triple Top chart-pattern indicator.

Bearish reversal: three peaks at approximately the same level with two
intervening troughs. Distinguished from Head-and-Shoulders by all three
peaks being roughly equal — a head-and-shoulders has a prominent middle
peak (head). Confirmation requires a close below the *lowest*
intervening trough (Bulkowski's confirmation level).

The Rust detector enforces:

* Sequence ``H-L-H-L-H``.
* Each peak within 2% of the trio's mean (default).
* Pattern depth ≥ 2% (lowest trough below the mean peak).
* Formation ≥ 10 bars between the first and last peak.

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

__all__ = ["TripleTop"]


@register_indicator(Pattern.TRIPLE_TOP.value)
class TripleTop(PatternIndicator):
    """Bearish "Triple Top" reversal."""

    pattern_name = "triple_top"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
    detector_param_keys = (
        "peak_tolerance",
        "min_trough_depth",
        "min_bar_count",
        "boundary_tolerance",
    )
    default_params = {
        **PatternIndicator.default_params,
        "min_quality": 71.0,
        "peak_tolerance": 0.02,
        "min_trough_depth": 0.02,
        "min_bar_count": 10,
        # Resistance line must not be pierced by more than 0.5% of the
        # peak level between the three peaks.
        "boundary_tolerance": 0.005,
    }
