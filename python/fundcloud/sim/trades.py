"""``Trade`` — executed transaction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fundcloud.sim.orders import Order

__all__ = ["Trade"]


@dataclass(frozen=True, slots=True)
class Trade:
    """A filled :class:`Order`. The portfolio applies these to mutate state.

    Attributes
    ----------
    order
        The original :class:`Order` that produced this fill.
    ts
        Timestamp at which the fill executed.
    asset
        Asset being traded (mirrors ``order.asset`` for convenience).
    qty
        Signed quantity. Positive for buys, negative for sells.
    price
        Fill price after slippage is applied.
    fee
        Commission / exchange fee charged to cash.
    slippage_bps
        Slippage applied vs the reference price, in basis points.
    """

    order: Order
    ts: pd.Timestamp
    asset: str
    qty: float
    price: float
    fee: float = 0.0
    slippage_bps: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.price
