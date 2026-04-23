"""12 — The efficient frontier, and finding the max-Sharpe portfolio.

Classic textbook exercise: for a universe of assets, sweep across target
expected returns, solve the minimum-variance problem at each target, and
inspect the resulting risk/return curve. The point that maximises Sharpe
is the tangency portfolio.

We use skfolio's ``MeanRisk`` with ``efficient_frontier_size=15`` to
generate 15 frontier portfolios in one fit call, then print them and
flag the max-Sharpe row. Assets come live from yfinance.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/12_efficient_frontier.py
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
        print("This example also requires skfolio — `uv add 'fundcloud[pf]'`", file=sys.stderr)
        return 1

    closes = pull_closes(
        {
            "US_EQ": "SPY",
            "EU_EQ": "VGK",
            "EM_EQ": "VWO",
            "BONDS_AGG": "AGG",
        },
        years=5,
    )
    if closes is None or closes.empty:
        return 1
    returns = closes.pct_change().dropna()
    print(
        f"Live data:  {returns.index[0].date()} → {returns.index[-1].date()}  "
        f"({len(returns)} trading days, {returns.shape[1]} assets)\n"
    )

    est = MeanRisk(risk_measure=RiskMeasure.VARIANCE, efficient_frontier_size=15)
    est.fit(returns)

    weights = np.asarray(est.weights_, dtype=float)
    assert weights.ndim == 2, "expected frontier weights matrix"
    n_points = weights.shape[0]

    ppy = 252.0
    mu = returns.mean().to_numpy() * ppy
    cov = returns.cov().to_numpy() * ppy
    ann_rets = weights @ mu
    ann_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpes = np.where(ann_vols > 0, ann_rets / ann_vols, np.nan)
    best = int(np.nanargmax(sharpes))

    print(f"{'point':>6}{'cagr':>12}{'ann_vol':>10}{'sharpe':>9}    {'weights'}")
    print("-" * 68)
    for i in range(n_points):
        marker = "  *" if i == best else "   "
        w_str = "/".join(f"{w:>5.2f}" for w in weights[i])
        print(
            f"{i:>6}{ann_rets[i] * 100:>11.2f}%{ann_vols[i] * 100:>9.2f}%"
            f"{sharpes[i]:>9.2f}{marker} {w_str}"
        )

    print(f"\nMax-Sharpe row:  {best}")
    print(f"  ann return:  {ann_rets[best] * 100:.2f}%")
    print(f"  ann vol:     {ann_vols[best] * 100:.2f}%")
    print(f"  sharpe:      {sharpes[best]:.2f}")
    print(
        "  weights:     "
        + ", ".join(
            f"{a}={w * 100:.1f}%" for a, w in zip(returns.columns, weights[best], strict=True)
        )
    )
    print("\nThe starred row is the tangency portfolio — max reward per unit of risk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
