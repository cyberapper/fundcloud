"""Ascending Triangle chart-pattern indicator.

Bullish continuation: a near-horizontal upper trend line (resistance)
combined with a rising lower trend line (support). Repeated tests of
resistance with progressively higher lows are read as accumulation;
breakout above resistance projects a measured move equal to the widest
part of the triangle (the gap at formation start).

The Rust detector enforces:

* Asymmetric flatness for the upper line: full ``flat_threshold`` upward,
  70% of it downward (a strongly-falling resistance is the signature of
  a symmetric or descending triangle, not ascending).
* Lower line normalised slope > 0 (rising support).
* Lines must converge (end gap < start gap, both positive).
* Every bar inside the formation stays within the channel
  (`validate_boundaries`, 2% of channel width tolerance).
* Formation ≥ 8 bars.

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

__all__ = ["AscendingTriangle"]


@register_indicator(Pattern.ASCENDING_TRIANGLE.value)
class AscendingTriangle(PatternIndicator):
    """Bullish "Ascending Triangle" continuation."""

    pattern_name = "ascending_triangle"
    condition = PatternCondition(
        entry_rule=EntryRule.ON_BREAKOUT,
        exit_rule=ExitRule.TARGET_OR_STOP,
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
    )
    detector_param_keys = ("flat_threshold", "min_touches")
    default_params = {
        **PatternIndicator.default_params,
        # Calibrated against a synthetic GBM corpus to preserve the top-X%
        # selectivity of the old ``min_quality=50`` floor under the new
        # anchor-only ``trendline_r2`` scorer. See
        # ``docs/guides/patterns/knobs.md`` for the full table.
        "min_quality": 74.0,
        "flat_threshold": 0.005,
        "min_touches": 2,
    }
