"""Double Top chart-pattern indicator.

Bearish reversal: two peaks at approximately the same level separated by
an intervening trough. Breakout below the trough (the neckline) confirms
the pattern; measured move targets the neckline minus the pattern height.

The Rust detector enforces:

* Sequence ``H-L-H``.
* Peaks within 1.5% of each other (default).
* Trough depth ≥ 3% below the average peak.
* Formation ≥ 5 bars between the peaks.

Each detection carries a Bulkowski variant tag in
``events["variant"]`` — one of
``STRICT_ADAM_ADAM`` / ``STRICT_ADAM_EVE`` / ``STRICT_EVE_ADAM`` /
``STRICT_EVE_EVE`` / ``WEAK_*`` — describing whether resistance held
on both tests (STRICT) or was marginally breached (WEAK), plus the
Adam (narrow spike) vs Eve (rounded reversal) shape of each peak.

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

__all__ = ["DoubleTop"]


@register_indicator(Pattern.DOUBLE_TOP.value)
class DoubleTop(PatternIndicator):
    """Bearish "Double Top" reversal."""

    pattern_name = "double_top"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
    detector_param_keys = ("peak_tolerance", "min_trough_depth")
    default_params = {
        **PatternIndicator.default_params,
        # Calibrated against a synthetic GBM corpus to preserve the top-X%
        # selectivity of the old ``min_quality=50`` floor under the new
        # anchor-only ``trendline_r2`` scorer. See
        # ``docs/guides/patterns/knobs.md`` for the full table.
        "min_quality": 75.0,
        "peak_tolerance": 0.015,
        "min_trough_depth": 0.03,
    }
