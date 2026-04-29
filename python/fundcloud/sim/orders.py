"""``Order`` — intended transaction."""

from __future__ import annotations

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

    Raises
    ------
    ValueError
        If neither ``qty`` nor ``notional`` is set, if ``qty`` is zero,
        or if ``kind="limit"`` without a ``limit_price``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fundcloud.sim import Order
    >>> Order(ts=pd.Timestamp("2024-01-02"), asset="SPY", side="buy", qty=10.0)  # doctest: +ELLIPSIS
    Order(ts=Timestamp('2024-01-02 00:00:00'), asset='SPY', side='buy', qty=10.0, ...)
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
