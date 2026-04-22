"""19 — Advanced optimisers: RiskBudgeting, HERC, NestedClusters, MaxDiversification.

Examples 04, 06, 12-15 already cover `MeanRisk` and `HierarchicalRiskParity`.
This scenario fills in the rest of the ``fundcloud.optimize`` menu on a real
six-asset multi-asset universe so you can see how each constructor sits
relative to the classic MVO / HRP benchmarks:

* ``RiskBudgeting`` — each asset contributes a target share of portfolio
  variance. Defaults to *risk parity* (equal risk contribution).
* ``HierarchicalEqualRiskContribution`` (HERC) — HRP's more opinionated
  sibling: cluster the tree, then equalise risk *within* each cluster.
* ``NestedClustersOptimization`` — two-level optimisation: pick an inner
  optimiser per cluster, then an outer one to combine clusters.
* ``MaximumDiversification`` — maximise the diversification ratio
  ``(w'σ) / sqrt(w'Σw)``; tilts towards low-correlation combinations.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/19_advanced_optimisers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _data import pull_closes

from fundcloud.portfolio import Population

HERE = Path(__file__).parent


def main() -> int:
    try:
        from fundcloud.optimize import (
            HierarchicalEqualRiskContribution,
            HierarchicalRiskParity,
            MaximumDiversification,
            MeanRisk,
            NestedClustersOptimization,
            RiskBudgeting,
            RiskMeasure,
        )
    except ImportError:
        print("This example requires skfolio — `uv add 'fundcloud[pf]'`", file=sys.stderr)
        return 1

    closes = pull_closes(
        {
            "US_EQ": "SPY",
            "EU_EQ": "VGK",
            "EM_EQ": "VWO",
            "TECH": "QQQ",
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
        f"({len(returns)} trading days, {returns.shape[1]} assets)\n"
    )

    # Reference points: MeanRisk (min-variance) and HRP.
    estimators: dict[str, object] = {
        "min_variance": MeanRisk(risk_measure=RiskMeasure.VARIANCE),
        "hrp": HierarchicalRiskParity(),
        "risk_parity": RiskBudgeting(),  # equal-risk by default
        "herc": HierarchicalEqualRiskContribution(),
        "nested_clusters": NestedClustersOptimization(),
        "max_diversification": MaximumDiversification(),
    }

    portfolios = []
    for label, est in estimators.items():
        est.fit(returns)
        portfolios.append(est.predict(returns).rename(label))

    pop = Population(portfolios)
    print("In-sample summary:\n")
    summary = pop.summary()
    rows = ["cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))

    print("\nLatest weights (columns summed to 100%):\n")
    print(pop.composition().to_string(float_format=lambda v: f"{v:>7.2%}"))

    diversification = {}
    for label, est in estimators.items():
        w = np.asarray(est.weights_, dtype=float)
        sigma = returns.cov().to_numpy()
        vols = returns.std(ddof=1).to_numpy()
        numerator = float(w @ vols)
        denominator = float(np.sqrt(w @ sigma @ w))
        diversification[label] = numerator / denominator if denominator else float("nan")
    print("\nDiversification ratio (higher = more uncorrelated exposure):")
    for label, ratio in sorted(diversification.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<22}  {ratio:.3f}")

    print("\nHow to read it:")
    print("  * Min-variance funnels into the lowest-vol asset — usually bonds.")
    print("  * HRP and risk_parity both target equal risk contribution but HRP")
    print("    clusters first: you'll see chunkier allocations inside each cluster.")
    print("  * HERC tightens HRP by equalising WITHIN clusters — typically raises")
    print("    the diversification ratio.")
    print("  * NestedClusters lets you pick an optimiser per cluster; the default")
    print("    config is a drop-in for HRP and often lands close to it.")
    print("  * MaximumDiversification explicitly optimises the ratio above, so")
    print("    it should top that column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
