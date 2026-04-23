"""``Hold`` — buy-and-hold, optionally rebalanced."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fundcloud.portfolio import Portfolio
from fundcloud.sim.orders import Order
from fundcloud.strategies.base import BaseStrategy, Context
from fundcloud.strategies.scheduler import Scheduler

__all__ = ["Hold", "RebalanceSpec"]


WeightsLike = Mapping[str, float] | pd.Series | Callable[[pd.DataFrame], Mapping[str, float]]


@dataclass(slots=True, frozen=True)
class RebalanceSpec:
    """Optional rebalance policy for :class:`Hold`."""

    horizon: str = "monthly"
    tolerance: float = 0.0


class Hold(BaseStrategy):
    """Allocate to target weights once, hold; optionally rebalance.

    Parameters
    ----------
    weights
        Either a mapping of ``asset -> weight`` or a callable receiving the
        ``init`` warm-up window and returning such a mapping. Weights must sum
        to 1.
    rebalance
        If supplied, restore target weights at each cadence boundary.
    start
        Optional lock-out: don't place the first allocation before ``start``.

    Examples
    --------
    Buy-and-hold 60/40 equity / bonds, no rebalancing (weights drift with
    prices):

    >>> from fundcloud.strategies import Hold
    >>> Hold({"SPY": 0.6, "AGG": 0.4})  # doctest: +ELLIPSIS
    <fundcloud.strategies.hold.Hold object at ...>

    Quarterly-rebalanced 60/40 with a 5 %-drift tolerance:

    >>> from fundcloud.strategies import RebalanceSpec
    >>> Hold(
    ...     {"SPY": 0.6, "AGG": 0.4},
    ...     rebalance=RebalanceSpec(horizon="91D", tolerance=0.05),
    ... )  # doctest: +ELLIPSIS
    <fundcloud.strategies.hold.Hold object at ...>
    """

    def __init__(
        self,
        weights: WeightsLike,
        *,
        rebalance: RebalanceSpec | None = None,
        start: pd.Timestamp | str | None = None,
    ) -> None:
        self._weights_spec = weights
        self._rebalance = rebalance
        self._start = pd.Timestamp(start) if start is not None else None
        self._resolved_weights: dict[str, float] = {}
        self._triggered_once: bool = False
        self._rebalance_triggers: set[pd.Timestamp] = set()

    # --------------------------------------------------------------- lifecycle

    def init(self, bars: pd.DataFrame, portfolio: Portfolio) -> None:
        w = self._weights_spec(bars) if callable(self._weights_spec) else self._weights_spec
        if isinstance(w, pd.Series):
            w = w.to_dict()
        total = sum(w.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"Hold weights must sum to 1, got {total}"
            raise ValueError(msg)
        self._resolved_weights = {k: float(v) for k, v in w.items()}

        if self._rebalance is not None:
            cadence = Scheduler.from_horizon(self._rebalance.horizon)
            self._rebalance_triggers = set(cadence.triggers(bars.index))

    def decide(self, ctx: Context) -> list[Order]:
        if self._start is not None and ctx.ts < self._start:
            return []
        should_allocate = not self._triggered_once
        should_rebalance = (
            self._rebalance is not None
            and ctx.ts in self._rebalance_triggers
            and self._triggered_once
        )
        if not (should_allocate or should_rebalance):
            return []

        orders = _orders_to_reach_weights(
            ctx,
            self._resolved_weights,
            tolerance=(self._rebalance.tolerance if self._rebalance else 0.0),
        )
        if orders:
            self._triggered_once = True
        return orders


# -------------------------------------------------------------------- helpers


def _orders_to_reach_weights(
    ctx: Context,
    weights: Mapping[str, float],
    *,
    tolerance: float = 0.0,
) -> list[Order]:
    """Emit the orders needed to bring current positions to target weights."""
    prices = _current_prices(ctx)
    if not prices:
        return []
    equity = ctx.portfolio.cash
    live_positions = {
        asset: pos.qty for asset, pos in ctx.portfolio._live.positions.items() if pos.qty
    }
    for asset, qty in live_positions.items():
        if asset in prices:
            equity += qty * prices[asset]

    if equity <= 0:
        return []

    orders: list[Order] = []
    for asset, target_w in weights.items():
        px = prices.get(asset)
        if px is None or px <= 0:
            continue
        target_qty = (equity * target_w) / px
        current_qty = live_positions.get(asset, 0.0)
        delta = target_qty - current_qty
        if abs(delta) * px < tolerance * equity:
            continue
        if delta == 0:
            continue
        orders.append(
            Order(
                ts=ctx.ts,
                asset=asset,
                side="buy" if delta > 0 else "sell",
                qty=abs(delta),
            )
        )
    return orders


def _current_prices(ctx: Context) -> dict[str, float]:
    """Latest close price per asset, from the current bar."""
    bar = ctx.bar
    out: dict[str, float] = {}
    if isinstance(bar.index, pd.MultiIndex):
        for (field, asset), val in bar.items():
            if field == "close" and pd.notna(val):
                out[asset] = float(val)
    else:
        for asset, val in bar.items():
            if pd.notna(val):
                out[str(asset)] = float(val)
    return out


# Keep `Any` used to avoid unused-import complaints when the module shrinks.
_ = Any
