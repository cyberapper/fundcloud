"""08 — DCA with ``amount_pct``: equity-relative deposits.

The ``amount`` knob fixes the deposit in dollars; ``amount_pct`` fixes
it as a fraction of *current* equity. Same cadence, same assets — but
``amount_pct`` lets the deposit scale automatically with portfolio
size, which is handy when you don't want to commit to an absolute
dollar figure up front.

We compare a fixed-dollar DCA to a 1 %-of-equity DCA on the same
two-asset universe and the same starting balance, and print the
side-by-side summary.

Run:
    uv run python examples/08_dca_amount_pct.py
"""

from __future__ import annotations

from pathlib import Path

from _synth import AssetProfile, generate_ohlcv
from fundcloud.portfolio import Population
from fundcloud.reports import Tearsheet
from fundcloud.sim import FixedBps, Simulator
from fundcloud.strategies import DCA

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bars = generate_ohlcv(
        {
            "SPY": AssetProfile(mu=0.08, sigma=0.16, price0=450.0),
            "AGG": AssetProfile(mu=0.03, sigma=0.05, price0=100.0),
        },
        start="2023-01-02",
        periods=504,
        seed=11,
    )

    # Same cash pool, monthly cadence, equal-weight default split across SPY/AGG.
    fixed = Simulator(bars, cash=100_000.0, costs=FixedBps(5)).run_strategy(
        DCA(amount=1_000.0, horizon="monthly")
    )
    pct = Simulator(bars, cash=100_000.0, costs=FixedBps(5)).run_strategy(
        DCA(amount_pct=0.01, horizon="monthly")
    )

    fixed.portfolio.rename("dca_fixed_1k")
    pct.portfolio.rename("dca_pct_1pct")

    pop = Population([fixed.portfolio, pct.portfolio])
    summary = pop.summary()
    rows = ["total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown"]
    print("Fixed dollars vs. % of equity:\n")
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))

    print(f"\nFinal equity (fixed):       ${fixed.equity_curve.iloc[-1]:>12,.0f}")
    print(f"Final equity (amount_pct):  ${pct.equity_curve.iloc[-1]:>12,.0f}")
    print(f"Trades fired (fixed):       {len(fixed.trades):>12d}")
    print(f"Trades fired (amount_pct):  {len(pct.trades):>12d}")

    for name, portfolio in (
        ("dca_fixed_1k", fixed.portfolio),
        ("dca_pct_1pct", pct.portfolio),
    ):
        out = OUT / f"08_{name}.html"
        Tearsheet(portfolio, title=name.replace("_", " ").title()).render_html(out)
    print(f"\nTear sheets: {OUT.relative_to(HERE.parent)}/08_*.html")


if __name__ == "__main__":
    main()
