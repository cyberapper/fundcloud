"""``DCA`` — dollar-cost averaging preset."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import pandas as pd

from fundcloud.portfolio import Portfolio
from fundcloud.sim.orders import Order
from fundcloud.strategies._helpers import _assets_from_bars, _current_equity
from fundcloud.strategies.base import BaseStrategy, Context
from fundcloud.strategies.hold import _current_prices
from fundcloud.strategies.scheduler import Cadence, Scheduler

__all__ = ["DCA"]


HorizonName = Literal["daily", "weekly", "monthly"]


class DCA(BaseStrategy):
    """Invest a fixed amount at a fixed cadence.

    Exactly one of ``amount`` or ``amount_pct`` must be provided —
    they're two ways to spell the same thing:

    * ``amount`` — explicit dollars per fire (scalar or per-asset).
    * ``amount_pct`` — fraction of the **current** portfolio equity
      deployed at each fire (scalar or per-asset). The dollar size is
      recomputed at every fire from ``Portfolio.equity_curve``, so the
      deposit grows or shrinks with the portfolio. Each fire is also
      clipped to currently-available cash — DCA never borrows; once
      cash is exhausted, subsequent fires emit no orders. On the very
      first fire (no equity history yet) it falls back to starting
      cash.

    Parameters
    ----------
    amount
        Dollars per fire. Either a scalar (distributed across
        ``weights``) or a mapping ``asset -> dollars``.
        Mutually exclusive with ``amount_pct``.
    amount_pct
        Equity fraction per fire. Either a scalar in ``[0, 1]``
        (distributed across ``weights``) or a mapping
        ``asset -> fraction``. Mutually exclusive with ``amount``.
    horizon
        Cadence — ``"daily"``, ``"weekly"`` (7 calendar days),
        ``"monthly"``, or a :class:`Cadence` for arbitrary steps.
    weights
        Optional. When omitted with a scalar ``amount`` / ``amount_pct``,
        DCA spreads the deposit equally across every asset in the
        ``bars`` frame at :meth:`init`. Provide an explicit mapping
        (fractions summing to 1) to weight the split unevenly. Negative
        weights are allowed and produce short-sells: e.g.
        ``weights={"A": 1.5, "B": -0.5}`` says long A by 1.5x the
        deposit and short B by 0.5x — the short proceeds fund the
        oversized long, and net deployed cash equals the deposit.
    start, end
        Optional window inside which DCA fires.
    sell_on_end
        When ``True``, close all positions on the bar after the last
        fire (after ``end``).

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

    Percentage of current equity instead of fixed dollars — deploy
    1 % of the portfolio each month, scaling automatically as equity
    grows:

    >>> DCA(amount_pct=0.01, horizon="monthly")  # doctest: +ELLIPSIS
    <fundcloud.strategies.dca.DCA object at ...>
    """

    def __init__(
        self,
        amount: float | Mapping[str, float] | None = None,
        *,
        amount_pct: float | Mapping[str, float] | None = None,
        horizon: HorizonName | Cadence | str = "monthly",
        weights: Mapping[str, float] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        sell_on_end: bool = False,
    ) -> None:
        if (amount is None) == (amount_pct is None):
            msg = "DCA: provide exactly one of `amount` or `amount_pct`."
            raise ValueError(msg)

        # Resolve the active sizing source into matching scalar / per-asset
        # storage. Per-asset mappings are stored verbatim; a scalar with no
        # weights defers the equal split to init() (where we can read the
        # bars frame); a scalar with weights is pre-multiplied here.
        self._scalar_amount: float | None = None
        self._amounts: dict[str, float] = {}
        self._scalar_amount_pct: float | None = None
        self._amount_pcts: dict[str, float] = {}

        if amount is not None:
            if isinstance(amount, Mapping):
                if weights is not None:
                    msg = "DCA: `weights` must be omitted when `amount` is a mapping."
                    raise ValueError(msg)
                self._amounts = {k: float(v) for k, v in amount.items()}
            else:
                self._scalar_amount = float(amount)
                if weights is None:
                    self._amounts = {}
                else:
                    _validate_weights_sum_to_one(weights)
                    self._amounts = {k: self._scalar_amount * float(v) for k, v in weights.items()}
        else:
            assert amount_pct is not None  # for type-checkers; guarded above
            _validate_amount_pct_values(amount_pct)
            if isinstance(amount_pct, Mapping):
                if weights is not None:
                    msg = "DCA: `weights` must be omitted when `amount_pct` is a mapping."
                    raise ValueError(msg)
                self._amount_pcts = {k: float(v) for k, v in amount_pct.items()}
            else:
                self._scalar_amount_pct = float(amount_pct)
                if weights is None:
                    self._amount_pcts = {}
                else:
                    _validate_weights_sum_to_one(weights)
                    self._amount_pcts = {
                        k: self._scalar_amount_pct * float(v) for k, v in weights.items()
                    }

        self._horizon = horizon
        self._start = pd.Timestamp(start) if start is not None else None
        self._end = pd.Timestamp(end) if end is not None else None
        self._sell_on_end = sell_on_end
        self._fire_set: set[pd.Timestamp] = set()
        self._last_fire: pd.Timestamp | None = None
        self._ended: bool = False

    # --------------------------------------------------------------- lifecycle

    def init(self, bars: pd.DataFrame, portfolio: Portfolio) -> None:
        # Deferred equal split for scalar dollars.
        if not self._amounts and self._scalar_amount is not None:
            assets = _assets_from_bars(bars)
            if not assets:
                msg = "DCA needs at least one asset in `bars`"
                raise ValueError(msg)
            per_leg = self._scalar_amount / len(assets)
            self._amounts = {a: per_leg for a in assets}
        # Deferred equal split for scalar pct.
        if not self._amount_pcts and self._scalar_amount_pct is not None:
            assets = _assets_from_bars(bars)
            if not assets:
                msg = "DCA needs at least one asset in `bars`"
                raise ValueError(msg)
            per_leg = self._scalar_amount_pct / len(assets)
            self._amount_pcts = {a: per_leg for a in assets}

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
        # Compute the per-asset dollar deposit for this fire — fixed when
        # `amount` was used, equity-scaled when `amount_pct` was used.
        if self._amount_pcts:
            equity = _current_equity(ctx.portfolio)
            deposits = {asset: pct * equity for asset, pct in self._amount_pcts.items()}
        else:
            deposits = self._amounts

        # Split legs by sign: negative dollar deposits are short-sells; the
        # proceeds add to live cash and can fund larger long legs in the
        # same fire. Process shorts first so a long-short setup like
        # ``weights={"A": 1.5, "B": -0.5}`` with a $1000 deposit gets the
        # full $1500 long on A (funded by $1000 starting cash + $500
        # from the B short) rather than clipping the long to cash on hand.
        short_legs = [(a, -float(d)) for a, d in deposits.items() if float(d) < 0]
        long_legs = [(a, float(d)) for a, d in deposits.items() if float(d) > 0]

        cash_left = float(ctx.portfolio.cash)
        orders: list[Order] = []

        # 1. Shorts first — proceeds add to the cash counter.
        for asset, dollars in short_legs:
            if asset not in prices or prices[asset] <= 0:
                continue
            qty = dollars / prices[asset]
            if qty <= 0:
                continue
            orders.append(Order(ts=ctx.ts, asset=asset, side="sell", qty=qty))
            cash_left += dollars

        # 2. Longs — clipped to live cash so DCA never deploys more *net*
        # cash than is available. Multi-leg longs share remaining cash in
        # iteration order. The next-bar fill price differs slightly from
        # the close price used here, so cash may dip marginally negative
        # for a single bar from fees / slippage; this avoids the unbounded
        # leverage that occurs without the clip.
        for asset, dollars in long_legs:
            if cash_left <= 0:
                break
            if asset not in prices or prices[asset] <= 0:
                continue
            funded = min(dollars, cash_left)
            qty = funded / prices[asset]
            if qty <= 0:
                continue
            orders.append(Order(ts=ctx.ts, asset=asset, side="buy", qty=qty))
            cash_left -= funded
        return orders


# -------------------------------------------------------------------- helpers


def _validate_weights_sum_to_one(weights: Mapping[str, float]) -> None:
    # Reject NaN / Inf per-weight before summing — those would otherwise
    # poison the total (NaN sum != 1) with a confusing message, and
    # propagate into ``Order.qty`` downstream. Negative weights are
    # **allowed**: they emit short-sells in :meth:`DCA.decide` (proceeds
    # add to cash and fund the long legs); only the net sum is bounded.
    bad = {k: v for k, v in weights.items() if not math.isfinite(float(v))}
    if bad:
        msg = f"DCA weights must be finite numbers, got {bad}"
        raise ValueError(msg)
    total_w = sum(float(v) for v in weights.values())
    if abs(total_w - 1.0) > 1e-6:
        msg = f"DCA weights must sum to 1, got {total_w}"
        raise ValueError(msg)


def _validate_amount_pct_values(amount_pct: float | Mapping[str, float]) -> None:
    """Reject ``amount_pct`` outside ``[0, 1]`` — a typo like ``1.5`` would
    silently oversize orders, and negatives degrade into skipped buys."""
    if isinstance(amount_pct, Mapping):
        bad = {k: float(v) for k, v in amount_pct.items() if not 0.0 <= float(v) <= 1.0}
        if bad:
            msg = f"DCA amount_pct values must be in [0, 1]; got {bad}"
            raise ValueError(msg)
    else:
        v = float(amount_pct)
        if not 0.0 <= v <= 1.0:
            msg = f"DCA amount_pct must be in [0, 1]; got {amount_pct!r}"
            raise ValueError(msg)


def _close_all(ctx: Context) -> list[Order]:
    """Flatten every open position — sell longs, buy-to-cover shorts."""
    orders: list[Order] = []
    # pylint: disable=protected-access
    for asset, pos in ctx.portfolio._live.positions.items():
        if pos.qty > 0:
            orders.append(Order(ts=ctx.ts, asset=asset, side="sell", qty=pos.qty))
        elif pos.qty < 0:
            orders.append(Order(ts=ctx.ts, asset=asset, side="buy", qty=abs(pos.qty)))
    return orders
