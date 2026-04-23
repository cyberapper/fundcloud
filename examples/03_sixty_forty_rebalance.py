"""03 — Classic 60/40 portfolio: rebalance vs drift.

Two assets:

* ``EQUITY`` — 8% annualised return, 16% vol (SPY-like).
* ``BOND``   — 3% annualised return, 6% vol, slightly negative correlation
  with equities (classic diversifier).

We compare a **buy-and-hold** (weights drift with returns) against the
same portfolio **rebalanced quarterly**. Quarterly rebalancing lowers
vol and max drawdown at the cost of a small amount of turnover.

Run:
    uv run python examples/03_sixty_forty_rebalance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _synth import AssetProfile, generate_ohlcv
from fundcloud.portfolio import Population
from fundcloud.reports import Tearsheet
from fundcloud.sim import FixedBps, Simulator
from fundcloud.strategies import Hold, RebalanceSpec

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    profiles = {
        "EQUITY": AssetProfile(mu=0.08, sigma=0.16, price0=100.0),
        "BOND": AssetProfile(mu=0.03, sigma=0.06, price0=100.0),
    }
    # Slight negative correlation — the diversification lever.
    correlations = np.array([[1.0, -0.2], [-0.2, 1.0]])
    bars = generate_ohlcv(
        profiles, start="2022-01-03", periods=756, correlations=correlations, seed=11
    )

    weights = {"EQUITY": 0.60, "BOND": 0.40}

    drift = Simulator(bars, cash=100_000.0, costs=FixedBps(5)).run_strategy(Hold(weights=weights))
    quarterly = Simulator(bars, cash=100_000.0, costs=FixedBps(5)).run_strategy(
        # 91 calendar days ≈ one trading quarter. Pandas Timedelta doesn't
        # accept "3M" because month length is ambiguous.
        Hold(weights=weights, rebalance=RebalanceSpec(horizon="91D"))
    )
    drift.portfolio.rename("drift_only")
    quarterly.portfolio.rename("quarterly_rebal")

    pop = Population([drift.portfolio, quarterly.portfolio])
    summary = pop.summary()
    rows = ["cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    print("Comparison:\n")
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))
    print()
    print(f"Drift-only trades:      {len(drift.trades):>4d}")
    print(f"Quarterly-rebal trades: {len(quarterly.trades):>4d}")

    out = OUT / "03_sixty_forty.html"
    Tearsheet(quarterly.portfolio, title="60/40 — quarterly rebalance").render_html(out)
    print(f"\nTear sheet: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
