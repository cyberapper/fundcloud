"""04 — Hierarchical Risk Parity vs Equal-Weight vs Mean-Variance.

Five-asset universe with a mix of realistic volatility profiles:

| Asset       | mu    | sigma | Notes                         |
|-------------|-------|-------|-------------------------------|
| US_EQ       | 0.08  | 0.17  | S&P 500 analogue              |
| EU_EQ       | 0.06  | 0.19  | Stoxx 600 analogue            |
| EM_EQ       | 0.07  | 0.22  | Emerging-markets equity       |
| BONDS_AGG   | 0.03  | 0.05  | Investment-grade bonds        |
| COMMODITIES | 0.05  | 0.25  | Broad commodities basket      |

We fit three optimisers on a historical window and compare their Sharpe,
vol, and max drawdown on the same window. HRP gets its full skfolio-backed
implementation when ``fundcloud[pf]`` is installed; otherwise the pure-
Python fallback ``InverseVolatility`` steps in transparently.

Run:
    uv add 'fundcloud[pf]'
    uv run python examples/04_hrp_vs_equal_weight.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _synth import AssetProfile, close_returns, generate_ohlcv
from fundcloud.optimize import MVO, EqualWeighted
from fundcloud.portfolio import Population

HERE = Path(__file__).parent


def _corr_matrix() -> np.ndarray:
    # Plausible cross-asset correlations: equities positively correlated,
    # bonds mildly negative, commodities loose connection.
    names = ["US_EQ", "EU_EQ", "EM_EQ", "BONDS_AGG", "COMMODITIES"]
    corr = np.eye(len(names))
    corr[0, 1] = corr[1, 0] = 0.75
    corr[0, 2] = corr[2, 0] = 0.65
    corr[1, 2] = corr[2, 1] = 0.70
    corr[3, 0] = corr[0, 3] = -0.15
    corr[3, 1] = corr[1, 3] = -0.10
    corr[3, 2] = corr[2, 3] = -0.05
    corr[4, 0] = corr[0, 4] = 0.30
    corr[4, 1] = corr[1, 4] = 0.25
    corr[4, 2] = corr[2, 4] = 0.35
    corr[4, 3] = corr[3, 4] = 0.10
    return corr


def main() -> None:
    profiles = {
        "US_EQ": AssetProfile(mu=0.08, sigma=0.17),
        "EU_EQ": AssetProfile(mu=0.06, sigma=0.19),
        "EM_EQ": AssetProfile(mu=0.07, sigma=0.22),
        "BONDS_AGG": AssetProfile(mu=0.03, sigma=0.05),
        "COMMODITIES": AssetProfile(mu=0.05, sigma=0.25),
    }
    bars = generate_ohlcv(
        profiles, start="2021-01-04", periods=1008, correlations=_corr_matrix(), seed=23
    )
    returns = close_returns(bars)

    optimisers = {
        "EqualWeighted": EqualWeighted(),
        "MVO_MaxSharpe": MVO(risk_free=0.0, l2=1e-4),
    }

    try:
        from fundcloud.optimize import HierarchicalRiskParity, MeanRisk, RiskMeasure

        optimisers["HRP"] = HierarchicalRiskParity()
        optimisers["MeanRisk_CVaR"] = MeanRisk(risk_measure=RiskMeasure.CVAR)
    except ImportError:
        print("(install `fundcloud[pf]` to include HRP and MeanRisk in this run)", file=sys.stderr)

    portfolios = []
    for name, est in optimisers.items():
        est.fit(returns)
        portfolio = est.predict(returns).rename(name)
        portfolios.append(portfolio)

    pop = Population(portfolios)
    summary = pop.summary()
    rows = ["cagr", "ann_volatility", "sharpe", "max_drawdown", "cvar"]
    print("Optimiser comparison (in-sample):\n")
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))

    # Composition view — latest weights per optimiser.
    print("\nLatest weights:\n")
    comp = pop.composition()
    print(comp.to_string(float_format=lambda v: f"{v:>7.2%}"))


if __name__ == "__main__":
    main()
