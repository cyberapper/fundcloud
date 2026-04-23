"""14 — Transaction costs and the rebalance decision.

Portfolios aren't free to change. A mean-variance optimiser that ignores
costs will churn the book every period. skfolio's ``MeanRisk`` takes a
``transaction_costs`` parameter and a ``previous_weights`` vector; the
optimiser then trades off "closer to the unconstrained optimum" against
"lower turnover".

This example splits 5 years of real history in two regimes — "yesterday"
(the first 3 years) and "today" (the last 2 years). It:

1. Fits MeanRisk on the OLD regime — that becomes the current book.
2. Re-fits on the NEW regime ignoring costs — the greedy rebalance.
3. Re-fits on the NEW regime with 50 bps costs + ``previous_weights`` —
   the cost-aware rebalance.

The gap between the no-cost and 50-bps new books is the value of
actually knowing your trading bill.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/14_transaction_costs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _data import pull_closes

HERE = Path(__file__).parent


def main() -> int:
    try:
        from fundcloud.optimize import MeanRisk, RiskMeasure
    except ImportError:
        print("This example requires skfolio — `uv add 'fundcloud[pf]'`", file=sys.stderr)
        return 1

    closes = pull_closes(
        {
            "US_EQ": "SPY",
            "EU_EQ": "VGK",
            "EM_EQ": "VWO",
            "BONDS_AGG": "AGG",
            "GOLD": "GLD",
        },
        years=5,
    )
    if closes is None or closes.empty:
        return 1
    returns = closes.pct_change().dropna()
    print(
        f"Live data:  {returns.index[0].date()} → {returns.index[-1].date()}  "
        f"({len(returns)} trading days)\n"
    )

    # Split ~3 years OLD vs ~2 years NEW. Real markets change regime —
    # 2020 → 2022 is a very different world from 2022 → today.
    cutoff = returns.index[int(len(returns) * 0.6)]
    cheap_hist = returns.loc[:cutoff]
    fresh_hist = returns.loc[cutoff:].iloc[1:]

    print(
        f"Old regime:  {cheap_hist.index[0].date()} → {cheap_hist.index[-1].date()}"
        f"   ({len(cheap_hist)} days)"
    )
    print(
        f"New regime:  {fresh_hist.index[0].date()} → {fresh_hist.index[-1].date()}"
        f"   ({len(fresh_hist)} days)\n"
    )

    # --- Step 1: fit on the OLD regime to get "yesterday's book" ----------
    greedy = MeanRisk(risk_measure=RiskMeasure.VARIANCE, min_return=0.06 / 252)
    greedy.fit(cheap_hist)
    prev_weights = np.asarray(greedy.weights_, dtype=float)

    # --- Step 2: re-fit on the NEW regime, two ways -----------------------
    cost_free = MeanRisk(risk_measure=RiskMeasure.VARIANCE, min_return=0.06 / 252)
    cost_free.fit(fresh_hist)

    with_costs = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        min_return=0.06 / 252,
        transaction_costs=0.0050,  # 50 bps per asset, one-way
        previous_weights=prev_weights,
    )
    with_costs.fit(fresh_hist)

    greedy_w = np.asarray(greedy.weights_, dtype=float)
    cost_free_w = np.asarray(cost_free.weights_, dtype=float)
    with_costs_w = np.asarray(with_costs.weights_, dtype=float)

    print(f"{'asset':<12}{'prev (greedy)':>16}{'new (no cost)':>16}{'new (50 bps)':>16}")
    print("-" * 60)
    for i, asset in enumerate(returns.columns):
        print(
            f"{asset:<12}"
            f"{greedy_w[i] * 100:>15.2f}%"
            f"{cost_free_w[i] * 100:>15.2f}%"
            f"{with_costs_w[i] * 100:>15.2f}%"
        )

    def one_way(new: np.ndarray) -> float:
        return float(np.abs(new - prev_weights).sum() / 2.0)

    print()
    print(f"One-way turnover to 'no cost' rebalance:  {one_way(cost_free_w) * 100:>6.2f}%")
    print(f"One-way turnover to '50 bps' rebalance:   {one_way(with_costs_w) * 100:>6.2f}%")
    print("\nTakeaway: cost-awareness anchors the new book to yesterday's,")
    print("           trading the theoretical optimum for a lower bill at the broker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
