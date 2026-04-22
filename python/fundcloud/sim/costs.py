"""Cost models — the commission side of a fill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["CostModel", "FixedBps", "NoCost", "PerShare"]


@runtime_checkable
class CostModel(Protocol):
    """Fee charged to cash for a fill."""

    def fee(self, *, price: float, qty: float) -> float: ...


@dataclass(frozen=True, slots=True)
class NoCost:
    """Zero-cost model for tests and textbook examples."""

    def fee(self, *, price: float, qty: float) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class FixedBps:
    """Proportional-to-notional fee, in basis points (1 bp = 0.01 %).

    Minimum fee can be set via ``minimum``.
    """

    bps: float = 5.0
    minimum: float = 0.0

    def fee(self, *, price: float, qty: float) -> float:
        notional = abs(price * qty)
        return max(self.minimum, notional * self.bps * 1e-4)


@dataclass(frozen=True, slots=True)
class PerShare:
    """Flat per-share commission (typical US equity broker pricing)."""

    rate: float = 0.005
    minimum: float = 1.0

    def fee(self, *, price: float, qty: float) -> float:
        return max(self.minimum, abs(qty) * self.rate)
