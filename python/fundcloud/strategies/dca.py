"""``DCA`` — dollar-cost averaging preset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd

from fundcloud.portfolio import Portfolio
from fundcloud.sim.orders import Order
from fundcloud.strategies.base import BaseStrategy, Context
from fundcloud.strategies.hold import _current_prices
from fundcloud.strategies.scheduler import Cadence, Scheduler

__all__ = ["DCA"]


HorizonName = Literal["daily", "weekly", "monthly"]


class DCA(BaseStrategy):
    """Invest a fixed amount at a fixed cadence.

    Parameters
    ----------
    amount
        Either a scalar (distributed across ``weights``) or a mapping
        ``asset -> dollars``.
    horizon
        Cadence — ``"daily"``, ``"weekly"`` (7 calendar days), ``"monthly"``,
        or a :class:`Cadence` for arbitrary steps.
    weights
        Optional. When omitted with a scalar ``amount``, DCA spreads the
        deposit equally across every asset in the ``bars`` frame it sees
        at :meth:`init`. Provide an explicit mapping (fractions summing
        to 1) to weight the split unevenly.
    start, end
        Optional window inside which DCA fires.
    sell_on_end
        When ``True``, close all positions on the last fire after ``end``.

    Examples
    --------
    Single-asset weekly DCA into SPY — the classic retail deposit:

    >>> from fundcloud.strategies import DCA
    >>> DCA(amount=500.0, horizon="weekly", weights={"SPY": 1.0})  # doctest: +ELLIPSIS
    <fundcloud.strategies.dca.DCA object at ...>

    Multi-asset monthly allocation with explicit dollar buckets per leg:

    >>> DCA({"SPY": 300.0, "AGG": 200.0}, horizon="monthly")  # doctest: +ELLIPSIS
    <fundcloud.strategies.dca.DCA object at ...>

    Scalar amount with no weights — equal-weight over whatever assets
    the bars frame contains:

    >>> DCA(500.0, horizon="weekly")  # doctest: +ELLIPSIS
    <fundcloud.strategies.dca.DCA object at ...>
    """

    def __init__(
        self,
        amount: float | Mapping[str, float],
        *,
        horizon: HorizonName | Cadence | str = "monthly",
        weights: Mapping[str, float] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        sell_on_end: bool = False,
    ) -> None:
        self._scalar_amount: float | None
        if isinstance(amount, Mapping):
            self._amounts: dict[str, float] = {k: float(v) for k, v in amount.items()}
            self._scalar_amount = None
        else:
            self._scalar_amount = float(amount)
            if weights is None:
                # Defer the per-asset split to init(), where we can read
                # the bars frame and divide equally across its assets.
                self._amounts = {}
            else:
                total_w = sum(weights.values())
                if abs(total_w - 1.0) > 1e-6:
                    msg = f"DCA weights must sum to 1, got {total_w}"
                    raise ValueError(msg)
                self._amounts = {k: self._scalar_amount * float(v) for k, v in weights.items()}
        self._horizon = horizon
        self._start = pd.Timestamp(start) if start is not None else None
        self._end = pd.Timestamp(end) if end is not None else None
        self._sell_on_end = sell_on_end
        self._fire_set: set[pd.Timestamp] = set()
        self._last_fire: pd.Timestamp | None = None
        self._ended: bool = False

    # --------------------------------------------------------------- lifecycle

    def init(self, bars: pd.DataFrame, portfolio: Portfolio) -> None:
        if not self._amounts and self._scalar_amount is not None:
            assets = _assets_from_bars(bars)
            if not assets:
                msg = "DCA needs at least one asset in `bars`"
                raise ValueError(msg)
            per_leg = self._scalar_amount / len(assets)
            self._amounts = {a: per_leg for a in assets}
        cadence = Scheduler.from_horizon(self._horizon, anchor=self._start)
        self._fire_set = set(
            cadence.triggers(
                bars.index,
                start=self._start,
                end=self._end,
            )
        )
        self._last_fire = max(self._fire_set) if self._fire_set else None

    def decide(self, ctx: Context) -> list[Order]:
        if ctx.ts not in self._fire_set:
            # Handle end-of-run sell
            if (
                self._sell_on_end
                and self._last_fire
                and ctx.ts > self._last_fire
                and not self._ended
            ):
                self._ended = True
                return _close_all(ctx)
            return []

        prices = _current_prices(ctx)
        orders: list[Order] = []
        for asset, dollars in self._amounts.items():
            if asset not in prices or prices[asset] <= 0:
                continue
            qty = dollars / prices[asset]
            if qty <= 0:
                continue
            orders.append(
                Order(
                    ts=ctx.ts,
                    asset=asset,
                    side="buy",
                    qty=qty,
                )
            )
        return orders


# -------------------------------------------------------------------- helpers


def _assets_from_bars(bars: pd.DataFrame) -> list[str]:
    """Distinct asset tickers from a Bars frame's columns."""
    if isinstance(bars.columns, pd.MultiIndex):
        return list(bars.columns.get_level_values(-1).unique())
    return [str(c) for c in bars.columns]


def _close_all(ctx: Context) -> list[Order]:
    """Produce sell orders for every currently open position."""
    orders: list[Order] = []
    # pylint: disable=protected-access
    for asset, pos in ctx.portfolio._live.positions.items():
        if pos.qty > 0:
            orders.append(Order(ts=ctx.ts, asset=asset, side="sell", qty=pos.qty))
    return orders
