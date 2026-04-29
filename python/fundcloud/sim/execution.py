"""Execution models — when (and at what price) an order fills.

Two models ship: :class:`NextBarOpen` (the default, avoids look-ahead)
and :class:`SameBarClose` (convenient for quick experiments, but introduces a
subtle bias because the signal and fill share the same bar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

__all__ = ["ExecutionModel", "NextBarOpen", "SameBarClose"]


@runtime_checkable
class ExecutionModel(Protocol):
    """Decide when an order submitted at bar ``t`` fills."""

    def fill_at(
        self,
        *,
        signal_index: int,
        bars_index_size: int,
    ) -> int | None: ...

    def reference_price(
        self,
        *,
        bars: pd.DataFrame,
        fill_index: int,
        asset: str,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class NextBarOpen:
    """Orders submitted at bar *t* fill at the **open** of bar *t + 1*."""

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
class SameBarClose:
    """Orders submitted at bar *t* fill at the **close** of bar *t*.

    Note the bias: signals derived from bar *t* can inspect the close then
    trade at the close. Use :class:`NextBarOpen` for honest backtests.
    """

    def fill_at(self, *, signal_index: int, bars_index_size: int) -> int | None:
        return signal_index

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
