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
    detector_param_keys = (
        "shoulder_tolerance",
        "min_head_prominence",
        "prior_trend_window",
    )
    default_params = {
        **PatternIndicator.default_params,
        # Calibrated against a real-data corpus (~50 US large/mid-caps +
        # ETFs + commodity/FX proxies, 2018-2026 dailies) to preserve the
        # top-X% selectivity of the old ``min_quality=50`` floor under the
        # current scorer. Real-data threshold is -6 points below the
        # synthetic-GBM recommendation (73) — H&S formations in real
        # markets are noisier and the synthetic GBM run was too strict.
        # See ``docs/guides/patterns/knobs.md`` for the full table.
        "min_quality": 67.0,
        "shoulder_tolerance": 0.10,
        "min_head_prominence": 0.03,
        "prior_trend_window": 20,
    }
