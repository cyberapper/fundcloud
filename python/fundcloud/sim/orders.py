"""``Order`` — intended transaction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = ["Order", "OrderKind", "OrderSide"]


OrderSide = Literal["buy", "sell"]
"""Direction of an :class:`Order`.

``"buy"`` opens or adds to a **long** (and closes / reduces a short).
``"sell"`` opens or adds to a **short** (and closes / reduces a long).
The simulator routes the same ``side`` through both flat→position
opens and position→opposite closes — what matters is the sign of the
resulting position-delta, not whether the trader thinks of it as
"entry" or "exit".
"""

OrderKind = Literal["market", "limit"]
"""Kind of order. ``"market"`` fills at the reference price; ``"limit"`` requires a ``limit_price``."""


@dataclass(frozen=True, slots=True)
class Order:
    """Instruction to trade. Frozen so strategies can hand one up safely.

    Exactly one of ``qty`` or ``notional`` must be set (``qty`` wins if
    both are). ``qty`` is unsigned; the :attr:`side` field carries
    direction. The :class:`~fundcloud.sim.Simulator` resolves
    notional-only orders to a quantity at fill time using the
    reference price.

    Parameters
    ----------
    ts
        Timestamp at which the strategy emitted the order.
    asset
        Asset ticker.
    side
        ``"buy"`` (long) or ``"sell"`` (short). Not the same as
        "open" / "close" — a buy on top of a short *reduces* the short
        position; a sell on top of a long *reduces* the long. See
        :data:`OrderSide` for the full convention.
    qty
        Unsigned share count. Mutually exclusive with ``notional``.
    notional
        Unsigned dollar amount; the simulator divides by the fill price
        at execution time. Mutually exclusive with ``qty``.
    kind
        ``"market"`` (default) or ``"limit"``. Limit orders also need
        ``limit_price``.
    limit_price
        Price ceiling (buy) or floor (sell). Required when
        ``kind="limit"``.
    sl_stop
        Stop-loss attached to the entry, expressed as a fraction in
        ``(0, 1)`` of the fill price (e.g. ``0.10`` = 10%). On a long
        entry the simulator records ``sl_level = fill_price * (1 - sl_stop)``
        and synthesises a forced sell when a subsequent bar's low pierces
        it. On a short entry the level is ``fill_price * (1 + sl_stop)``
        tested against bar high. Anchored to the *latest* fill — an
        accumulating second buy at a higher price tightens the stop.
        Cleared when the position closes. See
        :attr:`fundcloud.portfolio.Position.sl_level` and
        :data:`~fundcloud.sim.TradeReason`.
    tp_stop
        Take-profit attached to the entry, expressed as a fraction
        ``> 0`` of the fill price (e.g. ``0.20`` = 20%). Long entries get
        ``tp_level = fill_price * (1 + tp_stop)`` tested against bar
        high; shorts get ``fill_price * (1 - tp_stop)`` tested against
        bar low. No upper bound — but values ``>= 1`` on a *short* never
        fire because price cannot drop more than 100%. Anchor and clear
        rules mirror ``sl_stop``. May be set together with ``sl_stop``
        as a bracket order; if both could fire on the same bar, the
        stop-loss wins. See :attr:`fundcloud.portfolio.Position.tp_level`.
    tsl_stop
        Trailing stop-loss attached to the entry, expressed as a
        fraction in ``(0, 1)`` of the high-water mark (e.g. ``0.05``
        = 5%). Unlike ``sl_stop``, the trail anchor *ratchets in the
        favourable direction*: long anchors track ``max(anchor, bar.high)``
        bar by bar, never moving down; shorts track ``min(anchor, bar.low)``,
        never moving up. Effective trail level = ``anchor * (1 - tsl_stop)``
        for long, ``anchor * (1 + tsl_stop)`` for short. May coexist with
        ``sl_stop`` (the *tighter fill* wins — higher price for long,
        lower for short) and ``tp_stop`` (stops still beat take-profit
        on tied bars).

        On accumulating entries, the existing trail is **retained** —
        the high-water mark continues ratcheting from the *first*
        entry's price, regardless of the new entry's fill price or
        ``tsl_stop`` value. The trail is unconditionally cleared on
        close.

        Forced exits tag :attr:`Trade.reason` as ``"trailing_stop"``.
        See :attr:`fundcloud.portfolio.Position.tsl_pct` /
        :attr:`fundcloud.portfolio.Position.tsl_anchor`.

    Raises
    ------
    ValueError
        If neither ``qty`` nor ``notional`` is set; if ``qty`` is zero;
        if ``kind="limit"`` without a ``limit_price``; if ``sl_stop`` is
        outside ``(0, 1)``; if ``tp_stop`` is non-positive; or if
        ``tsl_stop`` is outside ``(0, 1)``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fundcloud.sim import Order
    >>> Order(ts=pd.Timestamp("2024-01-02"), asset="SPY", side="buy", qty=10.0)  # doctest: +ELLIPSIS
    Order(ts=Timestamp('2024-01-02 00:00:00'), asset='SPY', side='buy', qty=10.0, ...)

    Bracket order — long with 5% stop-loss and 10% take-profit:

    >>> Order(  # doctest: +ELLIPSIS
    ...     ts=pd.Timestamp("2024-01-02"), asset="SPY", side="buy",
    ...     qty=10.0, sl_stop=0.05, tp_stop=0.10,
    ... )
    Order(ts=..., sl_stop=0.05, tp_stop=0.1, ...)

    Full bracket — fixed stop-loss, take-profit, and trailing stop on
    the same entry. The fixed SL caps the worst-case loss at entry,
    the take-profit caps the best-case gain, and the trail rides the
    middle:

    >>> Order(  # doctest: +ELLIPSIS
    ...     ts=pd.Timestamp("2024-01-02"), asset="SPY", side="buy",
    ...     qty=10.0, sl_stop=0.10, tp_stop=0.30, tsl_stop=0.05,
    ... )
    Order(ts=..., sl_stop=0.1, tp_stop=0.3, tsl_stop=0.05)
    """

    ts: pd.Timestamp
    asset: str
    side: OrderSide
    qty: float | None = None
    notional: float | None = None
    kind: OrderKind = "market"
    limit_price: float | None = None
    sl_stop: float | None = None
    tp_stop: float | None = None
    tsl_stop: float | None = None

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            # ``Literal["buy", "sell"]`` is static-only; an arbitrary string
            # would be treated as sell-by-default in :meth:`signed_qty`,
            # silently flipping trade direction. Reject at construction.
            raise ValueError(f"Order side must be 'buy' or 'sell'; got {self.side!r}")
        if self.qty is None and self.notional is None:
            raise ValueError("Order needs qty or notional")
        if self.kind == "limit" and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        # ``qty`` and ``notional`` are unsigned magnitudes — direction
        # comes from :attr:`side`. A negative value here would silently
        # flip the trade's sign at fill time, so reject it at construction.
        # NaN/Inf would also pass past simple ``<= 0`` comparisons (since
        # all NaN comparisons are False), so we reject non-finite values
        # explicitly before the range checks.
        if self.qty is not None and (not math.isfinite(self.qty) or self.qty <= 0):
            raise ValueError(f"Order qty must be a positive finite number; got {self.qty!r}")
        if self.notional is not None and (not math.isfinite(self.notional) or self.notional <= 0):
            raise ValueError(
                f"Order notional must be a positive finite number; got {self.notional!r}"
            )
        if self.sl_stop is not None and (
            not math.isfinite(self.sl_stop) or not (0.0 < self.sl_stop < 1.0)
        ):
            raise ValueError(f"sl_stop must be a finite fraction in (0, 1); got {self.sl_stop!r}")
        if self.tp_stop is not None and (not math.isfinite(self.tp_stop) or self.tp_stop <= 0.0):
            raise ValueError(f"tp_stop must be a positive finite fraction; got {self.tp_stop!r}")
        if self.tsl_stop is not None and (
            not math.isfinite(self.tsl_stop) or not (0.0 < self.tsl_stop < 1.0)
        ):
            raise ValueError(f"tsl_stop must be a finite fraction in (0, 1); got {self.tsl_stop!r}")

    # ---------------------------------------------------------------- helpers

    def with_qty(self, qty: float) -> Order:
        """Return a new :class:`Order` with ``qty`` set and ``notional`` cleared.

        Useful when the simulator resolves a notional-only order to an
        explicit share count using the fill price.

        Parameters
        ----------
        qty
            Unsigned share count for the resolved order.

        Returns
        -------
        Order
            Copy of ``self`` with ``qty=qty`` and ``notional=None``.
        """
        return Order(
            ts=self.ts,
            asset=self.asset,
            side=self.side,
            qty=qty,
            notional=None,
            kind=self.kind,
            limit_price=self.limit_price,
            sl_stop=self.sl_stop,
            tp_stop=self.tp_stop,
            tsl_stop=self.tsl_stop,
        )

    def signed_qty(self) -> float:
        """Position-delta: positive for buys (long-bias), negative for sells (short-bias).

        Returns the change this order applies to the asset's position
        when it fills — ``+qty`` for ``"buy"``, ``-qty`` for ``"sell"``.
        A short-cover order (buy on top of a short) still has a positive
        signed quantity; the resulting net position is what indicates
        a close.

        Returns
        -------
        float
            ``+qty`` when ``side == "buy"``, ``-qty`` when ``side == "sell"``.

        Raises
        ------
        ValueError
            If ``qty`` is unset (i.e. the order is still notional-only —
            resolve it first via :meth:`with_qty`).
        """
        if self.qty is None:
            raise ValueError("signed_qty requires a resolved qty")
        return self.qty if self.side == "buy" else -self.qty
