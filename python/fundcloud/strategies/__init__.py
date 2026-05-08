"""Decision-logic presets and extension points.

Two built-in presets (:class:`Hold` and :class:`DCA`) cover the common
retail / allocation workflows. Custom strategies subclass
:class:`BaseStrategy` and may optionally register themselves via
:func:`register_strategy` so they become discoverable by name.
"""

from __future__ import annotations

from fundcloud.strategies.base import (
    BaseStrategy,
    Context,
    register_strategy,
    registered_strategies,
)
from fundcloud.strategies.dca import DCA
from fundcloud.strategies.hold import Hold, RebalanceSpec
from fundcloud.strategies.pattern import PatternStrategy
from fundcloud.strategies.scheduler import Cadence, Scheduler

__all__ = [
    "DCA",
    "BaseStrategy",
    "Cadence",
    "Context",
    "Hold",
    "PatternStrategy",
    "RebalanceSpec",
    "Scheduler",
    "register_strategy",
    "registered_strategies",
]
