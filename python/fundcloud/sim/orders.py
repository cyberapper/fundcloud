"""``Order`` — intended transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = ["Order", "OrderKind", "OrderSide"]


OrderSide = Literal["buy", "sell"]
OrderKind = Literal["market", "limit"]


@dataclass(frozen=True, slots=True)
class Order:
    """Instruction to trade. Frozen so strategies can hand one up safely.

    Exactly one of ``qty`` or ``notional`` must be set (``qty`` wins if both
    are). ``qty`` is positive for buys and negative for sells — the
    :attr:`side` field is redundant for market orders but makes limit logic
    and audit trails clearer.
    """

    ts: pd.Timestamp
    asset: str
    side: OrderSide
    qty: float | None = None
    notional: float | None = None
    kind: OrderKind = "market"
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if self.qty is None and self.notional is None:
            raise ValueError("Order needs qty or notional")
        if self.kind == "limit" and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        if self.qty is not None and self.qty == 0:
            raise ValueError("Order qty must be non-zero")

    # ---------------------------------------------------------------- helpers

    def with_qty(self, qty: float) -> Order:
        """Return a new Order with ``qty`` set and ``notional`` cleared."""
        return Order(
            ts=self.ts,
            asset=self.asset,
            side=self.side,
            qty=qty,
            notional=None,
            kind=self.kind,
            limit_price=self.limit_price,
        )

    def signed_qty(self) -> float:
        """Directional qty: positive for buys, negative for sells."""
        if self.qty is None:
            raise ValueError("signed_qty requires a resolved qty")
        return self.qty if self.side == "buy" else -self.qty
