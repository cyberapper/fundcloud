"""The simulator engine.

``Simulator`` and ``SimResult`` are lazy-loaded to avoid a circular import
against ``fundcloud.strategies`` — strategies need ``Order`` from this package
and the simulator needs ``BaseStrategy`` from ``fundcloud.strategies``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fundcloud.sim.costs import CostModel, FixedBps, NoCost, PerShare
from fundcloud.sim.execution import ExecutionModel, NextBarClose, NextBarOpen
from fundcloud.sim.orders import Order, OrderKind, OrderSide
from fundcloud.sim.slippage import HalfSpread, NoSlippage, SlippageModel
from fundcloud.sim.trades import Trade, TradeReason

__all__ = [
    "CostModel",
    "ExecutionModel",
    "FixedBps",
    "HalfSpread",
    "NextBarClose",
    "NextBarOpen",
    "NoCost",
    "NoSlippage",
    "Order",
    "OrderKind",
    "OrderSide",
    "PerShare",
    "SimResult",
    "Simulator",
    "SlippageModel",
    "Trade",
    "TradeReason",
]


_LAZY = {"Simulator", "SimResult"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from fundcloud.sim import simulator

        return getattr(simulator, name)
    raise AttributeError(f"module 'fundcloud.sim' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise lazy-loaded ``Simulator`` / ``SimResult`` for tab-completion."""
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # pragma: no cover
    from fundcloud.sim.simulator import SimResult, Simulator
