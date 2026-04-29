"""Slippage models — the price-adjust side of a fill.

A slippage model nudges the raw reference price (the
:class:`~fundcloud.sim.ExecutionModel`'s open / close) into the fill
price the simulator actually books. Models implement the
:class:`SlippageModel` :class:`typing.Protocol`.

Two concrete models ship:

* :class:`NoSlippage` — no adjustment (the
  :class:`~fundcloud.sim.Simulator` default; suitable for clean
  unit-test fixtures).
* :class:`HalfSpread` — pay half the bid-ask spread on every fill,
  symmetrically buys-up / sells-down. The simplest realistic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

__all__ = ["HalfSpread", "NoSlippage", "SlippageModel"]


@runtime_checkable
class SlippageModel(Protocol):
    """Adjust the raw reference price into an achievable fill price."""

    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        """Return ``(fill_price, slippage_bps)``.

        Parameters
        ----------
        price
            Reference price from the :class:`~fundcloud.sim.ExecutionModel`
            (typically the bar open or close).
        side
            ``"buy"`` or ``"sell"``. Buys should generally fill *above*
            ``price``, sells *below*.

        Returns
        -------
        tuple[float, float]
            ``(fill_price, slippage_bps)`` — the adjusted price and the
            implied slippage in basis points (positive number, recorded
            on the :class:`~fundcloud.sim.Trade` for analytics).
        """
        ...


@dataclass(frozen=True, slots=True)
class NoSlippage:
    """Fills hit the reference price exactly. The :class:`Simulator` default.

    Examples
    --------
    >>> from fundcloud.sim import NoSlippage
    >>> NoSlippage().apply(price=100.0, side="buy")
    (100.0, 0.0)
    """

    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        return price, 0.0


@dataclass(frozen=True, slots=True)
class HalfSpread:
    """Pay half the bid-ask spread (in basis points) on every fill.

    Buys execute at ``price * (1 + half_spread_bps * 1e-4)``; sells at
    the symmetric discount. The recorded slippage is half the spread.

    Parameters
    ----------
    spread_bps
        Full bid-ask spread in basis points. Default ``2.0`` (i.e. half a
        bp paid each way) — a reasonable proxy for liquid US equities.

    Examples
    --------
    >>> from fundcloud.sim import HalfSpread
    >>> HalfSpread(spread_bps=10.0).apply(price=100.0, side="buy")
    (100.05, 5.0)
    >>> HalfSpread(spread_bps=10.0).apply(price=100.0, side="sell")
    (99.95, 5.0)
    """

    spread_bps: float = 2.0

    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        half = self.spread_bps / 2.0
        adj = 1.0 + half * 1e-4 if side == "buy" else 1.0 - half * 1e-4
        return price * adj, half
