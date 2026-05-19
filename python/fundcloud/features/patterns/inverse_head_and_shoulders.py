"""Inverse Head and Shoulders chart-pattern indicator.

Bullish reversal: a trough (head) flanked by two higher troughs (shoulders)
of similar depth, sitting below a neckline drawn through the two
intervening highs. The breakout above the neckline projects a measured
move target equal to the neckline-to-head distance above the neckline.

The Rust detector enforces:

* Sequence ``L-H-L-H-L``.
* Head strictly below both shoulders.
* Shoulders within 10% of each other (default).
* Head prominence ≥ 3% below the average shoulder.
* Formation ≥ 8 bars.
* Prior trend slope < 0 (downtrend before reversal).

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

__all__ = ["InverseHeadAndShoulders"]


@register_indicator(Pattern.INVERSE_HEAD_AND_SHOULDERS.value)
class InverseHeadAndShoulders(PatternIndicator):
    """Bullish "Inverse Head and Shoulders" reversal."""

    pattern_name = "inverse_head_and_shoulders"
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
        # current scorer. Real-data threshold is -5 points below the
        # synthetic-GBM recommendation (73) — H&S formations in real
        # markets are noisier and the synthetic GBM run was too strict.
        # See ``docs/guides/patterns/knobs.md`` for the full table.
        "min_quality": 68.0,
        "shoulder_tolerance": 0.10,
        "min_head_prominence": 0.03,
        "prior_trend_window": 20,
    }
