"""07 — Live DCA into real SPY history (yfinance).

Battlefield version of example 01: same DCA strategy, but we pull actual
historical SPY data from Yahoo Finance instead of synthesising prices.

Trader question answered: "What would a $500/week DCA into SPY over the
last two years have actually returned?"

Run:
    uv add 'fundcloud[data-yf]'
    uv run python examples/07_live_spy_dca.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fundcloud.reports import Tearsheet
from fundcloud.sim import FixedBps, Simulator
from fundcloud.strategies import DCA

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    try:
        from fundcloud.data import YF
    except ImportError as e:
        print(f"this example requires yfinance: {e}", file=sys.stderr)
        return 1

    # Look back 2 years from today.
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=2)

    src = YF(symbols="SPY", interval="1d")
    try:
        bars = src.read(start=start, end=end)
    except Exception as e:
        print(f"yfinance request failed: {e}", file=sys.stderr)
        return 2

    if bars.empty:
        print("yfinance returned an empty frame — retry later", file=sys.stderr)
        return 2

    print(f"Data: SPY from {bars.index[0].date()} to {bars.index[-1].date()} ({len(bars)} bars)")

    result = Simulator(bars, cash=60_000.0, costs=FixedBps(5)).run_strategy(
        DCA(amount=500.0, horizon="weekly", weights={"SPY": 1.0})
    )

    invested = sum(
        float(q) * float(p)
        for q, p in zip(result.trades["qty"], result.trades["price"], strict=True)
    )
    end_equity = float(result.equity_curve.iloc[-1])
    print()
    print(f"Trades placed:     {len(result.trades):>4d}")
    print(f"Total invested:    ${invested:>12,.0f}")
    print(f"End equity:        ${end_equity:>12,.0f}")
    print(
        f"P&L:               ${end_equity - invested:>12,.0f}  "
        f"({(end_equity / invested - 1.0) * 100:+.1f}% on capital deployed)"
    )
    print(f"Sharpe:            {result.portfolio.sharpe():>12.2f}")
    print(f"Max drawdown:      {result.portfolio.max_drawdown() * 100:>11.1f}%")

    out = OUT / "07_live_spy_dca.html"
    Tearsheet(result.portfolio, title="Live SPY DCA $500/week").render_html(out)
    print(f"\nTear sheet: {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
