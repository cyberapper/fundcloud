"""05 — Golden-cross momentum strategy.

A textbook trading rule: go long when the 50-day SMA crosses **above** the
200-day SMA (a "golden cross"); exit when it crosses back **below** (a
"death cross"). This example demonstrates two things at once:

1. How to subclass :class:`BaseStrategy` for a custom rule.
2. Using the Rust-backed :func:`fundcloud.kernels.rolling_mean` from inside
   a strategy for speed on large histories.

Run:
    uv run python examples/05_golden_cross_momentum.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _synth import AssetProfile, generate_ohlcv
from fundcloud import kernels
from fundcloud.reports import Tearsheet
from fundcloud.sim import FixedBps, Order, Simulator
from fundcloud.strategies import BaseStrategy, Context

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


class GoldenCross(BaseStrategy):
    """Long when fast SMA > slow SMA; flat otherwise."""

    def __init__(
        self, asset: str, fast: int = 50, slow: int = 200, cash_fraction: float = 0.95
    ) -> None:
        self.asset = asset
        self.fast = fast
        self.slow = slow
        self.cash_fraction = cash_fraction
        self._holding = False

    def init(self, bars: pd.DataFrame, portfolio: object) -> None:
        close = bars[("close", self.asset)].to_numpy()
        self._fast = kernels.rolling_mean(close, self.fast)
        self._slow = kernels.rolling_mean(close, self.slow)

    def decide(self, ctx: Context) -> list[Order]:
        i = ctx.history.index.get_loc(ctx.ts)
        if i < self.slow:
            return []
        fast_now, fast_prev = self._fast[i], self._fast[i - 1]
        slow_now, slow_prev = self._slow[i], self._slow[i - 1]
        if np.isnan(slow_prev) or np.isnan(slow_now):
            return []

        crossed_above = (fast_prev <= slow_prev) and (fast_now > slow_now)
        crossed_below = (fast_prev >= slow_prev) and (fast_now < slow_now)

        orders: list[Order] = []
        if crossed_above and not self._holding:
            price = float(ctx.bar[("close", self.asset)])
            qty = (ctx.portfolio.cash * self.cash_fraction) / price
            if qty > 0:
                orders.append(Order(ts=ctx.ts, asset=self.asset, side="buy", qty=qty))
                self._holding = True
        elif crossed_below and self._holding:
            pos = ctx.portfolio._live.positions.get(self.asset)
            if pos is not None and pos.qty > 0:
                orders.append(Order(ts=ctx.ts, asset=self.asset, side="sell", qty=pos.qty))
                self._holding = False
        return orders


def main() -> None:
    bars = generate_ohlcv(
        {"SPY": AssetProfile(mu=0.08, sigma=0.16, price0=450.0)},
        start="2018-01-02",
        periods=1512,  # ~6 years to let the 200-SMA actually do work
        seed=9,
    )
    strat = GoldenCross(asset="SPY", fast=50, slow=200)
    result = Simulator(bars, cash=100_000.0, costs=FixedBps(5)).run_strategy(strat)

    print(f"Total trades:  {len(result.trades)}")
    if len(result.trades):
        buys = (result.trades["qty"] > 0).sum()
        sells = (result.trades["qty"] < 0).sum()
        print(f"Buys / sells:  {buys} / {sells}")
    print(f"End equity:    ${result.equity_curve.iloc[-1]:>12,.0f}")
    print(f"Sharpe:        {result.portfolio.sharpe():>12.2f}")
    print(f"Max drawdown:  {result.portfolio.max_drawdown() * 100:>11.1f}%")

    out = OUT / "05_golden_cross.html"
    Tearsheet(result.portfolio, title="SPY golden cross (50/200)").render_html(out)
    print(f"\nTear sheet: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
