"""Execution models — when (and at what price) an order fills.

Two models ship, both **strictly forward-looking** so a backtest can
never see prices that come after the signal bar:

* :class:`NextBarOpen` (default) — orders submitted at bar *t* fill at
  the **open** of bar *t + 1*.
* :class:`NextBarClose` — orders submitted at bar *t* fill at the
  **close** of bar *t + 1*. Useful when you want a full bar of
  participation between signal and execution (e.g. modelling VWAP-ish
  fills or end-of-day trading desks).

Both models implement the :class:`ExecutionModel`
:class:`typing.Protocol`. Custom subclasses are welcome — the
simulator's slow path honours whatever ``fill_at`` /
``reference_price`` returns — **except** that ``fill_at`` must return
a bar index strictly later than ``signal_index`` (or ``None``).
Same-bar / earlier fills introduce look-ahead bias and the simulator
raises :class:`ValueError` if it sees one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

__all__ = ["ExecutionModel", "NextBarClose", "NextBarOpen"]


@runtime_checkable
class ExecutionModel(Protocol):
    """Decide when an order submitted at bar ``t`` fills."""

    def fill_at(
        self,
        *,
        signal_index: int,
        bars_index_size: int,
    ) -> int | None:
        """Resolve the bar index where an order fills.

        Parameters
        ----------
        signal_index
            Integer position of the bar that emitted the order.
        bars_index_size
            Total number of bars in the simulator's index.

        Returns
        -------
        int or None
            Either ``None`` (no future bar available — typically the
            last bar; the simulator records the order as
            ``filled=False`` and skips execution) or a bar index
            **strictly greater than** ``signal_index``. Same-bar or
            earlier fills are a contract violation: they introduce
            look-ahead bias and the simulator raises
            :class:`ValueError`.
        """
        ...

    def reference_price(
        self,
        *,
        bars: pd.DataFrame,
        fill_index: int,
        asset: str,
    ) -> float:
        """Return the reference price for a fill at ``fill_index``.

        The simulator hands this price to the
        :class:`~fundcloud.sim.SlippageModel`, which nudges it to the
        achievable fill price.

        Parameters
        ----------
        bars
            Bars frame (``(field, symbol)`` MultiIndex columns).
        fill_index
            Bar index returned by :meth:`fill_at`.
        asset
            Asset ticker.

        Returns
        -------
        float
            Reference price (typically open or close of the fill bar).
        """
        ...


@dataclass(frozen=True, slots=True)
class NextBarOpen:
    """Orders submitted at bar *t* fill at the **open** of bar *t + 1*.

    The honest default: a strategy that decides on bar *t*'s close
    (using bar-*t* prices and history) gets the next available
    executable price, with no look-ahead. Orders emitted on the final
    bar can't fill — the simulator records them as ``filled=False``.

    Examples
    --------
    >>> from fundcloud.sim import NextBarOpen
    >>> NextBarOpen().fill_at(signal_index=4, bars_index_size=10)
    5
    >>> NextBarOpen().fill_at(signal_index=9, bars_index_size=10) is None
    True
    """

    def fill_at(self, *, signal_index: int, bars_index_size: int) -> int | None:
        nxt = signal_index + 1
        return nxt if nxt < bars_index_size else None

    def reference_price(
        self,
        *,
        bars: pd.DataFrame,
        fill_index: int,
        asset: str,
    ) -> float:
        return _price_at(bars, fill_index, asset, side="open")


@dataclass(frozen=True, slots=True)
class NextBarClose:
    """Orders submitted at bar *t* fill at the **close** of bar *t + 1*.

    Like :class:`NextBarOpen` but uses the close of the fill bar as
    the reference price — convenient when modelling end-of-day desks
    or a full bar of participation between signal and execution.
    Look-ahead-free: the fill bar is strictly later than the signal
    bar. Orders emitted on the final bar can't fill — the simulator
    records them as ``filled=False``.

    Examples
    --------
    >>> from fundcloud.sim import NextBarClose
    >>> NextBarClose().fill_at(signal_index=4, bars_index_size=10)
    5
    >>> NextBarClose().fill_at(signal_index=9, bars_index_size=10) is None
    True
    """

    def fill_at(self, *, signal_index: int, bars_index_size: int) -> int | None:
        nxt = signal_index + 1
        return nxt if nxt < bars_index_size else None

    def reference_price(
        self,
        *,
        bars: pd.DataFrame,
        fill_index: int,
        asset: str,
    ) -> float:
        return _price_at(bars, fill_index, asset, side="close")


# -------------------------------------------------------------------- helpers


def _price_at(
    bars: pd.DataFrame,
    fill_index: int,
    asset: str,
    *,
    side: Literal["open", "close"],
) -> float:
    if isinstance(bars.columns, pd.MultiIndex):
        return float(bars[(side, asset)].iloc[fill_index])
    # Wide single-field frame: interpret it as close prices per asset.
    return float(bars[asset].iloc[fill_index])
