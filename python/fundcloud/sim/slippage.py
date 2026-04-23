"""Slippage models — the price-adjust side of a fill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

__all__ = ["HalfSpread", "NoSlippage", "SlippageModel"]


@runtime_checkable
class SlippageModel(Protocol):
    """Adjust the raw reference price into an achievable fill price."""

    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        """Return ``(fill_price, slippage_bps)``."""


@dataclass(frozen=True, slots=True)
class NoSlippage:
    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        return price, 0.0


@dataclass(frozen=True, slots=True)
class HalfSpread:
    """Pay half the bid-ask spread (quoted in basis points) on every fill."""

    spread_bps: float = 2.0

    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        half = self.spread_bps / 2.0
        adj = 1.0 + half * 1e-4 if side == "buy" else 1.0 - half * 1e-4
        return price * adj, half
