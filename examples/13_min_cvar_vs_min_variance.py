"""13 — Min-CVaR vs Min-Variance: two flavours of 'safe'.

Variance measures *symmetric* risk — a 2-sigma gain contributes the same
as a 2-sigma loss. CVaR (Conditional Value-at-Risk) measures *tail*
risk — only the worst losses beyond a confidence threshold count.

Two optimisers on the same universe:

* ``MeanRisk(risk_measure=RiskMeasure.VARIANCE)`` — minimises portfolio
  variance under the fully-invested, long-only constraint.
* ``MeanRisk(risk_measure=RiskMeasure.CVAR)`` — minimises the 95%
  conditional value-at-risk.

We pull 5 years of real closes spanning the 2020 crash, the 2022
drawdown, and EM volatility — plenty of fat-tail material for CVaR to
have opinions about.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/13_min_cvar_vs_min_variance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from _data import pull_closes
from fundcloud.portfolio import Population

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

    portfolios = []
    for label, measure in (
        ("min_variance", RiskMeasure.VARIANCE),
        ("min_cvar", RiskMeasure.CVAR),
    ):
        est = MeanRisk(risk_measure=measure, min_return=None)
        est.fit(returns)
        portfolios.append(est.predict(returns).rename(label))

    pop = Population(portfolios)
    summary = pop.summary()
    rows = ["cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    print("In-sample comparison:\n")
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))

    print("\nLatest weights:\n")
    print(pop.composition().to_string(float_format=lambda v: f"{v:>7.2%}"))

    print("\nHow to read it:")
    print("  * min_variance looks at symmetric risk — treats a big rally the")
    print("    same as a big crash, so it doesn't mind a fatter left tail.")
    print("  * min_cvar minimises the mean of the worst 5% of outcomes; it will")
    print("    underweight assets whose crisis-day behaviour is bad.")
    print("  * The weight shift between the two is the 'tail-risk insurance' you")
    print("    pay in expected return to smooth out drawdowns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
