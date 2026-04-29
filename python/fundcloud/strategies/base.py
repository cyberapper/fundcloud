"""``BaseStrategy`` — decision logic that turns bars + state into orders.

A strategy is *not* a sklearn estimator; its contract is a per-bar
``decide`` rather than ``fit``/``transform``. The
:class:`~fundcloud.sim.Simulator` drives the lifecycle: ``init`` once
before bar 0, ``decide`` per bar, ``close`` once after the last bar.

Concrete examples shipping in core: :class:`~fundcloud.strategies.Hold`
and :class:`~fundcloud.strategies.DCA`. Subclass :class:`BaseStrategy`
to add your own.
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

    Strategies are stateless on the bar boundary (they may carry their
    own state across bars, but the simulator hands them everything they
    need to decide on this bar via this object).

    Attributes
    ----------
    ts
        Current bar timestamp.
    bar
        Current bar. For a Bars MultiIndex frame this is a
        :class:`pandas.Series` indexed by ``(field, asset)``; for a
        simple price panel it's a single-level Series indexed by asset.
    history
        Bars up to and including ``ts``. Strategies look back into this
        for indicators (rolling means, momentum windows, …).
    portfolio
        Live :class:`~fundcloud.portfolio.Portfolio`. Treat as
        read-only — the simulator owns all state mutations.
    assets
        Convenience: asset universe ordered by column appearance.
    extras
        Optional dict for user-specified scheduled events, factor
        scores, etc. Populated by the simulator when configured, else
        an empty dict.
    """

    ts: pd.Timestamp
    bar: pd.Series
    history: pd.DataFrame
    portfolio: Portfolio
    assets: tuple[str, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """ABC for every concrete strategy.

    The simulator drives subclasses through a three-stage lifecycle:
    :meth:`init` once before the first bar, :meth:`decide` once per
    bar, :meth:`close` once after the last bar. Only :meth:`decide` is
    required.

    Examples
    --------
    Minimal momentum strategy that buys 1 share when the 10-day return
    is positive and sells when it's negative:

    >>> import pandas as pd
    >>> from fundcloud.strategies import BaseStrategy, Context
    >>> from fundcloud.sim import Order
    >>>
    >>> class Momentum(BaseStrategy):
    ...     def __init__(self, lookback: int = 10) -> None:
    ...         self.lookback = lookback
    ...
    ...     def decide(self, ctx: Context) -> list[Order]:
    ...         if len(ctx.history) < self.lookback:
    ...             return []
    ...         orders: list[Order] = []
    ...         close = ctx.history.xs("close", axis=1, level=0)
    ...         ret = close.iloc[-1] / close.iloc[-self.lookback] - 1.0
    ...         for asset, r in ret.items():
    ...             side = "buy" if r > 0 else "sell"
    ...             orders.append(Order(ts=ctx.ts, asset=str(asset), side=side, qty=1.0))
    ...         return orders
    """

    def init(self, bars: pd.DataFrame, portfolio: Portfolio) -> None:  # noqa: B027
        """One-shot setup called before the first :meth:`decide`.

        Default: no-op. Override to compute a warm-up window, schedule
        cadences, or stash any state derived from the full ``bars``
        frame. Not abstract because most strategies don't need it.

        Parameters
        ----------
        bars
            The full Bars frame the simulator will iterate over —
            ``(field, symbol)`` MultiIndex columns, sorted DatetimeIndex.
            Use it to precompute schedules, asset lists, etc.
        portfolio
            The live :class:`~fundcloud.portfolio.Portfolio` (already
            funded with starting cash). Treat as read-only here.
        """

    @abstractmethod
    def decide(self, ctx: Context) -> list[Order]:
        """Return the orders this strategy wants executed.

        Called once per bar. The simulator queues the returned orders
        for execution under its :class:`~fundcloud.sim.ExecutionModel`
        (default :class:`~fundcloud.sim.NextBarOpen`, so orders emitted
        on bar *t* fill at the open of bar *t + 1*).

        Parameters
        ----------
        ctx
            Per-bar :class:`Context` — current timestamp, bar slice,
            full history through ``ctx.ts``, live portfolio, asset
            universe.

        Returns
        -------
        list of Order
            Zero or more :class:`~fundcloud.sim.Order` instances. An
            empty list means "do nothing this bar".
        """

    def close(self, portfolio: Portfolio) -> None:  # noqa: B027
        """End-of-run hook.

        Default: no-op. Override to compute summary stats, persist
        diagnostics, or release external resources. Not abstract
        because most strategies don't need it.

        Parameters
        ----------
        portfolio
            Final :class:`~fundcloud.portfolio.Portfolio` state after
            the last bar.
        """

    # ----------------------------------------------------------------- sugar

    @property
    def name(self) -> str:
        """Class name — used as the default label in :class:`SimResult`."""
        return type(self).__name__


# -------------------------------------------------------------------- registry


_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str) -> Any:
    """Decorator: make a strategy class discoverable by name.

    Registered strategies can be looked up via
    :func:`registered_strategies` — useful for config-driven runners
    that pick the strategy class from a string.

    Parameters
    ----------
    name
        String key the class will be stored under. Typically a short
        snake-case identifier.

    Returns
    -------
    callable
        The decorator that registers and returns the class unchanged.

    Examples
    --------
    >>> from fundcloud.strategies import BaseStrategy, register_strategy, registered_strategies
    >>> @register_strategy("noop")
    ... class Noop(BaseStrategy):
    ...     def decide(self, ctx):
    ...         return []
    >>> "noop" in registered_strategies()
    True
    """

    def deco(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        _REGISTRY[name] = cls
        return cls

    return deco


def registered_strategies() -> dict[str, type[BaseStrategy]]:
    """Return a snapshot of the strategy registry.

    Returns
    -------
    dict[str, type[BaseStrategy]]
        Copy of the ``name -> class`` mapping populated by
        :func:`register_strategy`. Mutating the returned dict does not
        affect the registry.
    """
    return dict(_REGISTRY)
