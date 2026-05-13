"""``PatternStrategy`` — chart-pattern entry/exit on a per-asset basis.

Wraps a :class:`fundcloud.features.patterns.PatternIndicator` plus a
:class:`fundcloud.features.patterns.PatternCondition` into a
:class:`BaseStrategy`. The simulator's ``init`` runs the indicator once
and caches the events table; ``decide`` walks the per-bar context,
opens trades on event timestamps, and closes them on intraday target /
stop hits (or on the optional ``time_stop_bars`` deadline).

**Long-only by default.** The simulator's broker-style position model
does not assume naked short capability, so bearish pattern events are
*skipped* unless ``inverse=True`` is set — in which case every event's
sign is flipped (the "fade the pattern" trade) so a Double Top fires a
long entry instead of being skipped.

This is a research-grade strategy: no slippage / fees / sizing logic
beyond a fixed fraction-of-equity ``size``. For a production engine
you'd subclass and override the ``decide`` method to add execution
realism.

Known limitations
-----------------
* Only :attr:`EntryRule.ON_BREAKOUT` and :attr:`ExitRule.TARGET_OR_STOP`
  are wired into ``decide``. Other enum values raise at construction.
* Target / stop levels are anchored to the detector's pre-fill
  ``entry_price`` rather than the actual market-order fill — gaps
  between breakout bar and fill bar will misalign thresholds. Same
  bucket as the no-slippage assumption above.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fundcloud.features.patterns import (
    PatternCondition,
    PatternIndicator,
    apply_condition,
)
from fundcloud.features.patterns._enums import Direction, EntryRule, ExitRule
from fundcloud.portfolio import Portfolio
from fundcloud.sim.orders import Order
from fundcloud.strategies.base import BaseStrategy, Context, register_strategy

__all__ = ["PatternStrategy"]


@register_strategy("pattern")
class PatternStrategy(BaseStrategy):
    """Chart-pattern strategy: enter on event, exit on target/stop/time.

    Parameters
    ----------
    indicator
        A configured :class:`PatternIndicator` (e.g.
        ``HeadAndShoulders(min_quality=70)``).
    condition
        Entry / exit / target / stop rules. Defaults to the indicator's
        class-level preset.
    size
        Per-trade fraction of equity, ``0..=1``. ``0.1`` (default) means
        each open trade uses 10 % of current equity.
    inverse
        ``False`` (default) — trade in the natural direction, skip
        bearish events. ``True`` — flip every event's direction so
        bearish events fire long entries (test the fade-the-pattern
        hypothesis end-to-end).

    Examples
    --------
    >>> from fundcloud.features.patterns import DoubleBottom              # doctest: +SKIP
    >>> from fundcloud.strategies import PatternStrategy                  # doctest: +SKIP
    >>> strat = PatternStrategy(DoubleBottom(min_quality=60), size=0.1)   # doctest: +SKIP
    >>> result = bars.fc.run_strategy(strat)                              # doctest: +SKIP
    """

    def __init__(
        self,
        indicator: PatternIndicator,
        *,
        condition: PatternCondition | None = None,
        size: float = 0.1,
    ) -> None:
        if not 0.0 < size <= 1.0:
            msg = f"size must be in (0, 1]; got {size}"
            raise ValueError(msg)
        self.indicator = indicator
        self.condition = condition if condition is not None else indicator.effective_condition
        if self.condition.entry_rule is not EntryRule.ON_BREAKOUT:
            msg = (
                f"PatternStrategy currently supports only EntryRule.ON_BREAKOUT; "
                f"got {self.condition.entry_rule!r}"
            )
            raise NotImplementedError(msg)
        if self.condition.exit_rule is not ExitRule.TARGET_OR_STOP:
            msg = (
                f"PatternStrategy currently supports only ExitRule.TARGET_OR_STOP; "
                f"got {self.condition.exit_rule!r}"
            )
            raise NotImplementedError(msg)
        self.size = size
        # Filled in init():
        self._events_by_asset: dict[str, list[dict[str, Any]]] = {}
        # Open trades: asset → {sign, entry, target, stop, entry_ts, entry_pos}
        self._open: dict[str, dict[str, Any]] = {}
        # Bar index lookup so time stops are O(1) per bar.
        self._bar_index: pd.DatetimeIndex | None = None

    # --------------------------------------------------------------- lifecycle

    def init(self, bars: pd.DataFrame, portfolio: Portfolio) -> None:
        # Reset per-run state up front so reusing a strategy instance
        # across simulations doesn't leak setups or open markers from
        # the previous run — including when ``events`` comes back empty.
        self._events_by_asset = {}
        self._open = {}
        self._bar_index = bars.index
        events = self.indicator.events(bars)
        if events.empty:
            return
        events = apply_condition(events, self.condition, bars)
        # 0.6.0: direction is caller-supplied via ``condition.direction``,
        # not inferred from the detector. Long-only strategy: only trade if
        # the user asked for bullish.
        if self.condition.direction is not Direction.BULLISH:
            return
        for asset, group in events.groupby("asset"):
            recs: list[dict[str, Any]] = []
            for _, ev in group.sort_values("breakout_ts").iterrows():
                if (
                    pd.isna(ev["entry_price"])
                    or pd.isna(ev["target_price"])
                    or pd.isna(ev["stop_price"])
                ):
                    continue
                recs.append({
                    "sign": 1,  # long-only (guarded by direction check above)
                    "ts": ev["breakout_ts"],
                    "entry": float(ev["entry_price"]),
                    "target": float(ev["target_price"]),
                    "stop": float(ev["stop_price"]),
                })
            if recs:
                self._events_by_asset[str(asset)] = recs

    def decide(self, ctx: Context) -> list[Order]:
        orders: list[Order] = []
        orders.extend(self._exit_orders(ctx))
        orders.extend(self._entry_orders(ctx))
        return orders

    # --------------------------------------------------------------- internals

    def _exit_orders(self, ctx: Context) -> list[Order]:
        """Close any open trade whose target / stop / time-stop fired."""
        out: list[Order] = []
        for asset in list(self._open):
            trade = self._open[asset]
            close = self._bar_field(ctx, asset, "close")
            high = self._bar_field(ctx, asset, "high")
            low = self._bar_field(ctx, asset, "low")
            if any(v is None for v in (close, high, low)):
                continue
            sign = trade["sign"]
            # Intraday hit checks. For long: target above entry, stop below.
            target_hit = (sign > 0 and high >= trade["target"]) or (
                sign < 0 and low <= trade["target"]
            )
            stop_hit = (sign > 0 and low <= trade["stop"]) or (sign < 0 and high >= trade["stop"])
            time_stop = self._time_stop_hit(ctx, trade)
            if not (target_hit or stop_hit or time_stop):
                continue
            qty = ctx.portfolio._live.positions.get(asset, _ZERO_POSITION).qty
            if qty == 0:
                # Position already flat (e.g., closed by another rule); drop the open marker.
                del self._open[asset]
                continue
            out.append(
                Order(
                    ts=ctx.ts,
                    asset=asset,
                    side="sell" if qty > 0 else "buy",
                    qty=abs(qty),
                )
            )
            del self._open[asset]
        return out

    def _entry_orders(self, ctx: Context) -> list[Order]:
        """Open a new trade for any asset whose event timestamp is now."""
        out: list[Order] = []
        equity = self._equity(ctx)
        if equity <= 0:
            return out
        for asset, recs in self._events_by_asset.items():
            if asset in self._open:
                continue
            for rec in recs:
                if rec["ts"] != ctx.ts:
                    continue
                close = self._bar_field(ctx, asset, "close")
                if close is None or close <= 0:
                    continue
                notional = equity * self.size
                qty = notional / close
                if qty <= 0:
                    continue
                out.append(
                    Order(
                        ts=ctx.ts,
                        asset=asset,
                        side="buy" if rec["sign"] > 0 else "sell",
                        qty=qty,
                    )
                )
                self._open[asset] = {
                    "sign": rec["sign"],
                    "entry": rec["entry"],
                    "target": rec["target"],
                    "stop": rec["stop"],
                    "entry_ts": ctx.ts,
                }
                break
        return out

    def _time_stop_hit(self, ctx: Context, trade: dict[str, Any]) -> bool:
        n = self.condition.time_stop_bars
        if n is None or self._bar_index is None:
            return False
        try:
            entry_pos = self._bar_index.get_loc(trade["entry_ts"])
            now_pos = self._bar_index.get_loc(ctx.ts)
        except KeyError:
            return False
        if isinstance(entry_pos, slice):
            entry_pos = entry_pos.start
        if isinstance(now_pos, slice):
            now_pos = now_pos.start
        return (now_pos - entry_pos) >= n

    def _equity(self, ctx: Context) -> float:
        equity = float(ctx.portfolio.cash)
        for asset, pos in ctx.portfolio._live.positions.items():
            px = self._bar_field(ctx, asset, "close")
            if px is not None:
                equity += pos.qty * px
        return equity

    def _bar_field(self, ctx: Context, asset: str, field: str) -> float | None:
        try:
            value = ctx.bar[(field, asset)]
        except (KeyError, TypeError):
            return None
        if pd.isna(value):
            return None
        return float(value)


class _ZeroPosition:
    """Sentinel returned when an asset has no open position."""

    qty: float = 0.0


_ZERO_POSITION = _ZeroPosition()
