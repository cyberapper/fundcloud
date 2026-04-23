"""02 — DCA vs Lump-sum: which one wins?

Classic retail debate. We answer it on one synthetic market with the same
total capital deployed two ways:

* **Lump-sum**: buy $52,000 of SPY on day 1 and hold.
* **Weekly DCA**: drip $500 per week for 104 weeks (total $52,000).

Over a rising market lump-sum usually wins because you're in the market
earlier. Over a choppy or falling market DCA reduces entry-price risk.
We print a Population summary so both sides of the result are visible.

Run:
    uv run python examples/02_dca_vs_lump_sum.py
"""

from __future__ import annotations

from pathlib import Path

from _synth import AssetProfile, generate_ohlcv
from fundcloud.portfolio import Population
from fundcloud.reports import Tearsheet
from fundcloud.sim import FixedBps, Simulator
from fundcloud.strategies import DCA, Hold

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bars = generate_ohlcv(
        {"SPY": AssetProfile(mu=0.08, sigma=0.16, price0=450.0)},
        start="2023-01-02",
        periods=504,
        seed=7,
    )

    lump_sum = Simulator(bars, cash=52_000.0, costs=FixedBps(5)).run_strategy(
        Hold(weights={"SPY": 1.0})
    )
    dca = Simulator(bars, cash=60_000.0, costs=FixedBps(5)).run_strategy(
        DCA(amount=500.0, horizon="weekly", weights={"SPY": 1.0})
    )

    lump_sum.portfolio.rename("lump_sum")
    dca.portfolio.rename("dca_weekly")

    pop = Population([lump_sum.portfolio, dca.portfolio])
    summary = pop.summary()
    cols = ["lump_sum", "dca_weekly"]
    rows = ["total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    table = summary.loc[rows, cols]
    print("Comparison:\n")
    print(table.to_string(float_format=lambda v: f"{v:>10.4f}"))

    winner = (
        "lump_sum"
        if summary.at["sharpe", "lump_sum"] > summary.at["sharpe", "dca_weekly"]
        else "dca_weekly"
    )
    print(f"\n→ Better Sharpe: {winner}")

    # One report per strategy — easier than merging for a retail reader.
    for name, portfolio in (("lump_sum", lump_sum.portfolio), ("dca_weekly", dca.portfolio)):
        out = OUT / f"02_{name}.html"
        Tearsheet(portfolio, title=f"{name.replace('_', ' ').title()}").render_html(out)
    print(f"\nTear sheets: {OUT.relative_to(HERE.parent)}/02_*.html")


if __name__ == "__main__":
    main()
