"""08 — Live cross-asset portfolio: equities + bonds + crypto.

Pulls real data from three different providers and builds a diversified
portfolio:

* **SPY** (US equities) via yfinance — no key needed.
* **TLT** (long Treasuries) via FMP — needs ``FMP_API_KEY``.
* **BTC/USDT** (crypto) via Binance — no key needed.

We resample each source to daily frequency, align the overlapping range,
run HRP from skfolio (with an InverseVolatility fallback), and print the
Population summary.

Run:
    uv add 'fundcloud[pf,data-yf,data-fmp,data-bn]'
    export FMP_API_KEY=...
    uv run python examples/08_live_cross_asset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from fundcloud.optimize import EqualWeighted, InverseVolatility
from fundcloud.portfolio import Population
from fundcloud.reports import Tearsheet

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _pull_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    try:
        from fundcloud.data import YF
    except ImportError:
        print("(yfinance missing — skip SPY leg)", file=sys.stderr)
        return None
    try:
        bars = YF("SPY", interval="1d").read(start=start, end=end)
    except Exception as e:
        print(f"(YF SPY failed: {e})", file=sys.stderr)
        return None
    return bars[("close", "SPY")].rename("SPY")


def _pull_tlt(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    if not os.environ.get("FMP_API_KEY"):
        print("(FMP_API_KEY missing — skip TLT leg)", file=sys.stderr)
        return None
    try:
        from fundcloud.data import FMP
    except ImportError:
        print("(httpx missing — skip TLT leg)", file=sys.stderr)
        return None
    try:
        bars = FMP("TLT", interval="1d").read(start=start, end=end)
    except Exception as e:
        print(f"(FMP TLT failed: {e})", file=sys.stderr)
        return None
    return bars[("close", "TLT")].rename("TLT")


def _pull_btc(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    try:
        from fundcloud.data import Binance
    except ImportError:
        print("(ccxt missing — skip BTC leg)", file=sys.stderr)
        return None
    try:
        bars = Binance("BTC/USDT", interval="1d").read(start=start, end=end)
    except Exception as e:
        print(f"(Binance BTC failed: {e})", file=sys.stderr)
        return None
    return bars[("close", "BTC/USDT")].rename("BTC")


def main() -> int:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=1)

    legs: dict[str, pd.Series] = {}
    for name, pull in (("SPY", _pull_spy), ("TLT", _pull_tlt), ("BTC", _pull_btc)):
        series = pull(start, end)
        if series is not None and not series.empty:
            legs[name] = series

    if len(legs) < 2:
        print("Need at least 2 live legs to build a portfolio — exiting.", file=sys.stderr)
        return 1

    # Align on the intersection of trading days (crypto trades weekends, equity
    # ETFs don't — we take the intersection for a clean comparison).
    frame = pd.concat(legs, axis=1).dropna()
    print(
        f"Aligned range:  {frame.index[0].date()} → {frame.index[-1].date()}  "
        f"({len(frame)} bars across {frame.shape[1]} assets)"
    )
    print(f"Assets:         {', '.join(frame.columns)}")
    returns = frame.pct_change().dropna()

    # Run whichever optimisers we have. HRP / MeanRisk need [pf].
    optimisers: dict[str, object] = {
        "EqualWeighted": EqualWeighted(),
        "InverseVolatility": InverseVolatility(),
    }
    try:
        from fundcloud.optimize import HierarchicalRiskParity, MeanRisk, RiskMeasure

        optimisers["HRP"] = HierarchicalRiskParity()
        optimisers["MeanRisk_CVaR"] = MeanRisk(risk_measure=RiskMeasure.CVAR)
    except ImportError:
        print("(install fundcloud[pf] to include HRP + MeanRisk)", file=sys.stderr)

    portfolios = []
    for name, est in optimisers.items():
        est.fit(returns)  # type: ignore[attr-defined]
        p = est.predict(returns).rename(name)  # type: ignore[attr-defined]
        portfolios.append(p)

    pop = Population(portfolios)
    summary = pop.summary()
    rows = ["cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    print("\nComparison (in-sample on 1 year of live data):\n")
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))

    print("\nLatest weights:\n")
    print(pop.composition().to_string(float_format=lambda v: f"{v:>7.2%}"))

    best = str(summary.loc["sharpe"].idxmax())
    best_pf = next(p for p in portfolios if p.name == best)
    out = OUT / "08_live_cross_asset.html"
    Tearsheet(best_pf, title=f"Cross-asset — best in-sample ({best})").render_html(out)
    print(f"\nTear sheet: {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
