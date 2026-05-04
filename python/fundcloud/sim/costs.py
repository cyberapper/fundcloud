"""Cost models — the commission side of a fill.

A cost model decides how much cash the simulator deducts on top of the
notional when an :class:`~fundcloud.sim.Order` fills. Models implement
the :class:`CostModel` :class:`typing.Protocol`, so any callable / class
with a matching ``fee`` method works.

Three concrete models ship:

* :class:`NoCost` — zero fees, for tests and textbook examples.
* :class:`FixedBps` — proportional-to-notional fee (default of the
  :class:`~fundcloud.sim.Simulator`, 5 bps).
* :class:`PerShare` — flat per-share commission with a minimum, the
  shape of typical US-equity broker pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["CostModel", "FixedBps", "NoCost", "PerShare"]


@runtime_checkable
class CostModel(Protocol):
    """Fee charged to cash for a fill.

    Implement ``fee(price, qty)`` returning a non-negative dollar
    amount. The simulator subtracts the result from cash on every fill.
    """

    def fee(self, *, price: float, qty: float) -> float:
        """Return the commission for a fill of ``qty`` shares at ``price``.

        Parameters
        ----------
        price
            Fill price (after slippage).
        qty
            Signed quantity. Implementations should treat the magnitude;
            buys and sells are charged symmetrically.

        Returns
        -------
        float
            Non-negative dollar fee.
        """
        ...


@dataclass(frozen=True, slots=True)
class NoCost:
    """Zero-cost model for tests and textbook examples.

    Examples
    --------
    >>> from fundcloud.sim import NoCost
    >>> NoCost().fee(price=100.0, qty=10.0)
    0.0
    """

    def fee(self, *, price: float, qty: float) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class FixedBps:
    """Proportional-to-notional fee, in basis points (1 bp = 0.01 %).

    The :class:`~fundcloud.sim.Simulator` default. Charges
    ``max(minimum, |price * qty| * bps * 1e-4)`` per fill, so a 5 bps
    model on a $10,000 trade costs $5.

    Parameters
    ----------
    bps
        Basis points charged on notional. Default ``5.0``.
    minimum
        Floor in dollars. Useful when modelling broker minimums.
        Default ``0.0``.

    Examples
    --------
    >>> from fundcloud.sim import FixedBps
    >>> FixedBps(bps=10).fee(price=100.0, qty=50.0)
    5.0
    >>> FixedBps(bps=5, minimum=1.0).fee(price=10.0, qty=1.0)  # tiny trade hits the floor
    1.0
    """

    bps: float = 5.0
    minimum: float = 0.0

    def __post_init__(self) -> None:
        # The :class:`CostModel` protocol promises non-negative fees; a
        # negative ``bps`` or ``minimum`` would credit cash on every fill
        # and silently corrupt downstream PnL.
        if self.bps < 0:
            raise ValueError(f"FixedBps bps must be non-negative; got {self.bps!r}")
        if self.minimum < 0:
            raise ValueError(f"FixedBps minimum must be non-negative; got {self.minimum!r}")

    def fee(self, *, price: float, qty: float) -> float:
        notional = abs(price * qty)
        return max(self.minimum, notional * self.bps * 1e-4)


@dataclass(frozen=True, slots=True)
class PerShare:
    """Flat per-share commission (typical US-equity broker pricing).

    Charges ``max(minimum, |qty| * rate)`` per fill — independent of
    price.

    Parameters
    ----------
    rate
        Dollars per share. Default ``0.005`` (half a cent).
    minimum
        Floor in dollars. Default ``1.0``.

    Examples
    --------
    >>> from fundcloud.sim import PerShare
    >>> PerShare(rate=0.005).fee(price=50.0, qty=400.0)
    2.0
    >>> PerShare(rate=0.005, minimum=1.0).fee(price=50.0, qty=10.0)  # below floor
    1.0
    """

    rate: float = 0.005
    minimum: float = 1.0

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError(f"PerShare rate must be non-negative; got {self.rate!r}")
        if self.minimum < 0:
            raise ValueError(f"PerShare minimum must be non-negative; got {self.minimum!r}")

    def fee(self, *, price: float, qty: float) -> float:
        return max(self.minimum, abs(qty) * self.rate)
