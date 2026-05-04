"""``Trade`` — executed transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from fundcloud.sim.orders import Order

__all__ = ["Trade", "TradeReason"]


TradeReason = Literal["signal", "stop_loss", "take_profit", "trailing_stop"]
"""Why a :class:`Trade` was emitted.

``"signal"`` — a strategy-emitted :class:`Order` that filled normally.
``"stop_loss"`` — a forced exit synthesised by the simulator's intra-bar
stop-loss check (long stop tripped on bar low, short stop on bar high).
``"take_profit"`` — a forced exit synthesised by the simulator's
intra-bar take-profit check (long TP tripped on bar high, short TP on
bar low).
``"trailing_stop"`` — a forced exit synthesised by the simulator's
intra-bar trailing-stop check (the trail level ratchets in the
favourable direction with each bar's high/low, then triggers like a
fixed stop on the unfavourable side).

When several stops could fire on the same bar, the conservative
arbitration is: any stop (fixed or trailing) beats take-profit; among
``sl_stop`` and ``tsl_stop`` the *tighter* effective level wins (the
one that's closer to current price). The simulator records the source
so analytics can split realised P&L between discretionary exits,
defensive stop-outs, profit-taking, and trail-following.
"""


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
    reason
        Why the trade fired — see :data:`TradeReason` for the full
        enumeration. ``"signal"`` (default) for strategy-driven fills;
        ``"stop_loss"`` / ``"take_profit"`` / ``"trailing_stop"`` for
        forced exits synthesised by the simulator's intra-bar bracket
        check. Surfaces in the trades DataFrame so analytics can split
        realised P&L between discretionary exits, defensive stop-outs,
        profit-taking, and trail-following.
    """

    order: Order
    ts: pd.Timestamp
    asset: str
    qty: float
    price: float
    fee: float = 0.0
    slippage_bps: float = 0.0
    reason: TradeReason = "signal"

    @property
    def notional(self) -> float:
        """Signed dollar value of the fill: ``qty * price``.

        Positive for buys (cash outflow), negative for sells (cash
        inflow). Note this excludes fees — the simulator subtracts
        ``fee`` from cash separately.
        """
        return self.qty * self.price
