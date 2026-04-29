"""``Trade`` — executed transaction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fundcloud.sim.orders import Order

__all__ = ["Trade"]


@dataclass(frozen=True, slots=True)
class Trade:
    """A filled :class:`Order`. The portfolio applies these to mutate state.

    Trades are the simulator's output unit: each one books a quantity
    of an asset at a fill price, charges a fee against cash, and
    records the slippage applied vs the reference price.

    Attributes
    ----------
    order
        The original :class:`~fundcloud.sim.Order` that produced this
        fill.
    ts
        Timestamp at which the fill executed.
    asset
        Asset being traded (mirrors ``order.asset`` for convenience).
    qty
        Signed quantity. Positive for buys, negative for sells.
    price
        Fill price after slippage is applied.
    fee
        Commission / exchange fee charged to cash. Always
        non-negative.
    slippage_bps
        Slippage applied vs the reference price, in basis points
        (positive number). ``0.0`` under :class:`~fundcloud.sim.NoSlippage`.
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
        """Signed dollar value of the fill: ``qty * price``.

        Positive for buys (cash outflow), negative for sells (cash
        inflow). Note this excludes fees — the simulator subtracts
        ``fee`` from cash separately.
        """
        return self.qty * self.price
