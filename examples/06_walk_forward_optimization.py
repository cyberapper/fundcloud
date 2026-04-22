"""06 — Walk-forward Mean-Risk optimisation with purged CV.

The quant workflow: use a purged K-fold to carve the return history into
train/test pairs, fit a Mean-Risk optimiser on each training slice, and
apply the resulting weights to the out-of-sample test slice. The
combined OOS returns tell you how the strategy would have behaved on
data it never saw during fitting — a much more honest read than
in-sample Sharpe.

Demonstrates:

* :class:`fundcloud.validate.PurgedKFold` as a drop-in sklearn splitter.
* :class:`fundcloud.validate.EmbargoedKFold` variant for forward-leakage scenarios.
* :class:`fundcloud.optimize.MeanRisk` driving a re-fit per fold.
* Stitching fold results back together into a Fundcloud ``Portfolio``.

Run:
    uv add 'fundcloud[pf]'
    uv run python examples/06_walk_forward_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _synth import AssetProfile, close_returns, generate_ohlcv
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet
from fundcloud.validate import EmbargoedKFold, PurgedKFold

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    try:
        from fundcloud.optimize import MeanRisk, RiskMeasure
    except ImportError:
        print(
            "This example requires skfolio — install with `uv add 'fundcloud[pf]'`", file=sys.stderr
        )
        return 1

    profiles = {
        "US_EQ": AssetProfile(mu=0.08, sigma=0.17),
        "EU_EQ": AssetProfile(mu=0.06, sigma=0.19),
        "BONDS_AGG": AssetProfile(mu=0.03, sigma=0.05),
        "GOLD": AssetProfile(mu=0.04, sigma=0.12),
    }
    bars = generate_ohlcv(profiles, start="2019-01-02", periods=1260, seed=5)
    returns = close_returns(bars)

    # 5 purged folds with a 5-day embargo — defensible on daily data.
    cv = PurgedKFold(n_splits=5, purge=5)
    oos_returns = pd.Series(0.0, index=returns.index)
    weights_per_fold: dict[str, pd.Series] = {}
    in_sample_sharpes: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(returns), start=1):
        train = returns.iloc[train_idx]
        test = returns.iloc[test_idx]
        est = MeanRisk(risk_measure=RiskMeasure.CVAR)
        est.fit(train)
        in_sample_sharpes.append(float(est.predict(train).sharpe()))
        w = np.asarray(est.weights_, dtype=float)
        weights_per_fold[f"fold_{fold}"] = pd.Series(w, index=returns.columns)
        fold_r = test.to_numpy() @ w
        oos_returns.iloc[test_idx] = fold_r

    oos_returns = oos_returns.loc[oos_returns != 0.0]

    # --- EmbargoedKFold variant: also silences the next train fold ---
    cv_emb = EmbargoedKFold(n_splits=5, purge=5, embargo=5)
    oos_returns_emb = pd.Series(0.0, index=returns.index)
    for fold_e, (train_idx_e, test_idx_e) in enumerate(cv_emb.split(returns), start=1):
        train_e = returns.iloc[train_idx_e]
        test_e  = returns.iloc[test_idx_e]
        est_e = MeanRisk(risk_measure=RiskMeasure.CVAR)
        est_e.fit(train_e)
        w_e = np.asarray(est_e.weights_, dtype=float)
        oos_returns_emb.iloc[test_idx_e] = test_e.to_numpy() @ w_e
    oos_returns_emb = oos_returns_emb.loc[oos_returns_emb != 0.0]
    portfolio_emb = Portfolio(returns=oos_returns_emb, name="walk_forward_embargoed")

    portfolio = Portfolio(returns=oos_returns, name="walk_forward_meanrisk")

    print("In-sample Sharpe per fold:")
    for fold, sharpe in enumerate(in_sample_sharpes, start=1):
        print(f"  fold_{fold}: {sharpe:>6.2f}")
    print(f"Out-of-sample Sharpe:  {portfolio.sharpe():>6.2f}")
    print(f"OOS ann. return:       {portfolio.summary()['cagr'] * 100:>5.2f}%")
    print(f"OOS max drawdown:      {portfolio.max_drawdown() * 100:>5.2f}%")

    print(f"\nEmbargoedKFold comparison:")
    print(f"  OOS Sharpe (PurgedKFold):    {portfolio.sharpe():>6.2f}")
    print(f"  OOS Sharpe (EmbargoedKFold): {portfolio_emb.sharpe():>6.2f}")

    print("\nWeights per fold:\n")
    weights_df = pd.DataFrame(weights_per_fold).T
    print(weights_df.to_string(float_format=lambda v: f"{v:>7.2%}"))

    out = OUT / "06_walk_forward.html"
    Tearsheet(portfolio, title="Walk-forward MeanRisk (CVaR), OOS").render_html(out)
    print(f"\nTear sheet: {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
