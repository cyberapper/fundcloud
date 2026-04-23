"""01 — Weekly DCA into an SPY-like index fund.

Retail scenario: "I want to drip $500 into SPY every week for two years.
What's my realised Sharpe, max drawdown, and end equity?"

Run:
    uv run python examples/01_dca_weekly_spy.py
"""

from __future__ import annotations

from pathlib import Path

from _synth import AssetProfile, generate_ohlcv
from fundcloud.reports import Tearsheet
from fundcloud.sim import FixedBps, Simulator
from fundcloud.strategies import DCA

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    # ~2 years of business days, 8% annualised return, 16% annualised vol.
    bars = generate_ohlcv(
        {"SPY": AssetProfile(mu=0.08, sigma=0.16, price0=450.0)},
        start="2023-01-02",
        periods=504,
        seed=7,
    )

    sim = Simulator(bars, cash=60_000.0, costs=FixedBps(5))
    result = sim.run_strategy(DCA(amount=500.0, horizon="weekly", weights={"SPY": 1.0}))

    invested = sum(
        float(t) * float(p)
        for t, p in zip(result.trades["qty"], result.trades["price"], strict=True)
    )
    print(f"Period:           {bars.index[0].date()} → {bars.index[-1].date()}")
    print(f"Total trades:     {len(result.trades)}")
    print(f"Total invested:   ${invested:>12,.0f}")
    print(f"End equity:       ${result.equity_curve.iloc[-1]:>12,.0f}")
    print(f"Sharpe:           {result.portfolio.sharpe():>12.2f}")
    print(f"Max drawdown:     {result.portfolio.max_drawdown() * 100:>11.1f}%")

    out = OUT / "01_dca_weekly_spy.html"
    Tearsheet(result.portfolio, title="DCA $500/week into SPY").render_html(out)
    print(f"\nTear sheet: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
