"""``BaseStrategy`` — decision logic that turns bars + state into orders.

A strategy is *not* a sklearn estimator; its contract is a per-bar
``decide`` rather than ``fit``/``transform``. The simulator drives the
lifecycle (``init`` once before bar 0, ``decide`` per bar, ``close`` once
after the last bar).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fundcloud.portfolio import Portfolio
from fundcloud.sim.orders import Order

__all__ = [
    "BaseStrategy",
    "Context",
    "register_strategy",
    "registered_strategies",
]


@dataclass(slots=True)
class Context:
    """Per-bar context handed to :meth:`BaseStrategy.decide`.

    Attributes
    ----------
    ts
        Current bar timestamp.
    bar
        Current bar. For a ``Bars`` MultiIndex frame this is a ``Series``
        indexed by ``(field, asset)``; for a simple price panel it's a
        single-level Series indexed by asset.
    history
        Bars up to and including ``ts``. Strategies look back into this for
        indicators.
    portfolio
        Live :class:`Portfolio` — callers should treat this as read-only.
    assets
        Convenience: asset universe ordered by column appearance.
    extras
        Optional dict for user-specified scheduled events, factor scores,
        etc. Populated by the simulator if needed, else an empty dict.
    """

    ts: pd.Timestamp
    bar: pd.Series
    history: pd.DataFrame
    portfolio: Portfolio
    assets: tuple[str, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """ABC for every concrete strategy."""

    def init(self, bars: pd.DataFrame, portfolio: Portfolio) -> None:  # noqa: B027
        """One-shot setup called before the first ``decide``.

        Default: no-op. Not abstract because most strategies don't need warm-up.
        """

    @abstractmethod
    def decide(self, ctx: Context) -> list[Order]:
        """Return the orders this strategy wants executed **next bar**."""

    def close(self, portfolio: Portfolio) -> None:  # noqa: B027
        """End-of-run hook.

        Default: no-op. Not abstract because most strategies don't need it.
        """

    # ----------------------------------------------------------------- sugar

    @property
    def name(self) -> str:
        return type(self).__name__


# -------------------------------------------------------------------- registry


_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str) -> Any:
    """Decorator: make a strategy class discoverable by name."""

    def deco(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        _REGISTRY[name] = cls
        return cls

    return deco


def registered_strategies() -> dict[str, type[BaseStrategy]]:
    return dict(_REGISTRY)
