"""09 — Live market snapshot via Alpha Vantage.

A "how's my watchlist doing" view — pulls the last ~6 months of daily
data for a handful of mega-caps via Alpha Vantage, then prints a
side-by-side metrics table.

Alpha Vantage's free tier is rate-limited (~5 requests / minute). We
stagger calls with a small sleep between symbols so the script runs
cleanly end-to-end on a free key.

Run:
    uv add 'fundcloud[data-av]'
    export ALPHAVANTAGE_API_KEY=...     # or ALPHA_VANTAGE_API_KEY
    uv run python examples/09_live_market_snapshot.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from fundcloud.metrics import batch_summary

HERE = Path(__file__).parent

WATCHLIST = ["AAPL", "MSFT", "IBM"]


def main() -> int:
    if not (os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("ALPHA_VANTAGE_API_KEY")):
        print(
            "Set ALPHAVANTAGE_API_KEY (or ALPHA_VANTAGE_API_KEY) to run this example.",
            file=sys.stderr,
        )
        return 1
    try:
        from fundcloud.data import AV
    except ImportError as e:
        print(f"this example requires httpx: {e}", file=sys.stderr)
        return 1

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=6)

    closes: dict[str, pd.Series] = {}
    for i, symbol in enumerate(WATCHLIST):
        print(f"[{i + 1}/{len(WATCHLIST)}] pulling {symbol} …", file=sys.stderr)
        try:
            bars = AV(symbol, interval="1d").read(start=start, end=end)
        except Exception as e:
            print(f"  {symbol}: {e}", file=sys.stderr)
            continue
        if bars.empty:
            print(f"  {symbol}: empty frame", file=sys.stderr)
            continue
        closes[symbol] = bars[("close", symbol)]
        # Respect the 5 calls / minute free-tier ceiling.
        if i < len(WATCHLIST) - 1:
            time.sleep(13)

    if not closes:
        print("No data pulled — check your API key and rate limit.", file=sys.stderr)
        return 2

    returns = pd.DataFrame(closes).pct_change().dropna()
    strategies: dict[str, pd.Series] = {sym: returns[sym] for sym in returns.columns}
    summary = batch_summary(strategies)

    cols = ["cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    print(
        f"\nWatchlist snapshot  ({returns.index[0].date()} → {returns.index[-1].date()},"
        f" {len(returns)} trading days):\n"
    )
    print(summary[cols].to_string(float_format=lambda v: f"{v:>10.4f}"))

    best_sharpe = summary["sharpe"].idxmax()
    worst_dd = summary["max_drawdown"].idxmin()
    print(f"\n→ Best Sharpe:     {best_sharpe}   ({summary.at[best_sharpe, 'sharpe']:.2f})")
    print(f"→ Worst drawdown:  {worst_dd}  ({summary.at[worst_dd, 'max_drawdown'] * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
