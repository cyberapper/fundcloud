"""15 — Black-Litterman: blending market-implied returns with your views.

Classical mean-variance takes historical mean returns at face value — a
shaky assumption because historical means are notoriously noisy
estimates of expected returns. Black-Litterman starts from an implied
"equilibrium" prior and then lets the user specify a handful of explicit
**views** — linear equations on expected returns, each with a confidence
level.

This example:

1. Runs a cost-free MeanRisk as the baseline ("pure history").
2. Specifies two views via skfolio's ``BlackLitterman`` prior:
   * "US equities will return 5% more per year than bonds."
   * "EU equities will return 1% more per year than EM equities."
3. Re-fits MeanRisk with that prior and compares the weight drift.

Views don't need to be directional-strong — even low-confidence views
pull the optimisation away from pathological historical artefacts.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/15_black_litterman.py
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
        from skfolio.prior import BlackLitterman
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

    # --- Baseline: plain mean-variance on history -------------------------
    baseline = MeanRisk(risk_measure=RiskMeasure.VARIANCE, min_return=0.08 / 252)
    baseline.fit(returns)

    # --- Views ------------------------------------------------------------
    # skfolio accepts equality views as plain strings, interpreted against
    # the column names of the ``X`` matrix. Numbers are **annualised**
    # expected return differentials.
    views = [
        "US_EQ - BONDS_AGG == 0.05",  # US equities beat bonds by 5% a year
        "EU_EQ - EM_EQ    == 0.01",  # EU slightly above EM
    ]
    bl_prior = BlackLitterman(views=views)
    with_views = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        min_return=0.08 / 252,
        prior_estimator=bl_prior,
    )
    with_views.fit(returns)

    base_w = np.asarray(baseline.weights_, dtype=float)
    view_w = np.asarray(with_views.weights_, dtype=float)

    print(f"{'asset':<12}{'baseline':>12}{'with views':>14}{'drift':>10}")
    print("-" * 50)
    for i, asset in enumerate(returns.columns):
        drift = view_w[i] - base_w[i]
        print(f"{asset:<12}{base_w[i] * 100:>11.2f}%{view_w[i] * 100:>13.2f}%{drift * 100:>+9.2f}%")

    # Show the historical annualised returns that the baseline is implicitly
    # trusting — makes the direction of each view's effect obvious.
    hist_ann = returns.mean() * 252.0 * 100.0
    print("\nBaseline's implicit expected return (annualised, from 5y history):")
    for asset in returns.columns:
        print(f"  {asset:<12} {hist_ann[asset]:>6.2f}%")

    turnover = float(np.abs(view_w - base_w).sum() / 2.0)
    print(f"\nOne-way turnover to honour the views:  {turnover * 100:.2f}%")

    print("\nHow to read it:")
    print("  * Each view either RAISES or LOWERS the posterior expected return")
    print("    versus the historical baseline. A 'US beats bonds by 5%' view is")
    print("    more conservative than what the last five years actually printed —")
    print("    so the posterior is *less* bullish on US, and weight drifts out")
    print("    of US and into bonds. Had we written '20%' instead, the drift")
    print("    would have flipped the other way.")
    print("  * Black-Litterman's job is to stop you blindly trusting noisy")
    print("    historical means — sometimes that means de-risking an overheated")
    print("    estimate, not just cranking up a bullish bet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
