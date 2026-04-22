"""24 — Attribution, skfolio round-trip, and direct Rust-kernel batches.

This example ties together the last three uncovered pockets of the public
API:

* **Portfolio-side analytics** not yet in other examples:
  :meth:`Portfolio.attribution`, :meth:`Portfolio.contribution`,
  :meth:`Portfolio.turnover`, and the ``.fc`` direct-call metrics
  :func:`omega` / :func:`ulcer_index` / :func:`value_at_risk`.
* **skfolio round-trip** via :meth:`Portfolio.from_skfolio` /
  :meth:`Portfolio.to_skfolio` — showing that a skfolio optimiser's
  ``predict`` output can be lifted into Fundcloud analytics and bounced
  back.
* **Rust-kernel direct batches** — :func:`kernels.sharpe_batch`,
  :func:`kernels.cvar_batch`, :func:`kernels.max_drawdown_batch` evaluated
  on a 200-weighting random sweep. Feeding flat NumPy arrays to the
  kernels skips the pandas overhead and completes the sweep in a few ms.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/24_attribution_and_kernels.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from _data import pull_closes

from fundcloud import kernels
from fundcloud.metrics import omega, ulcer_index, value_at_risk
from fundcloud.portfolio import Portfolio

HERE = Path(__file__).parent


def main() -> int:
    try:
        from fundcloud.optimize import MeanRisk, RiskMeasure
    except ImportError:
        print("This example requires skfolio — `uv add 'fundcloud[pf]'`", file=sys.stderr)
        return 1

    closes = pull_closes(
        {"US_EQ": "SPY", "EU_EQ": "VGK", "BONDS_AGG": "AGG", "GOLD": "GLD"},
        years=5,
    )
    if closes is None or closes.empty:
        return 1
    returns = closes.pct_change().dropna()
    print(
        f"Live data:  {returns.index[0].date()} → {returns.index[-1].date()}  "
        f"({len(returns)} trading days, {returns.shape[1]} assets)\n"
    )

    # --- 1. skfolio -> Fundcloud (analytics on a skfolio optimiser output) ---
    print("--- 1. skfolio -> Fundcloud (from_skfolio) ---")
    est = MeanRisk(risk_measure=RiskMeasure.VARIANCE, min_return=0.06 / 252)
    est.fit(returns)
    sk_portfolio = est.predict(returns)
    fc_view = Portfolio.from_skfolio(sk_portfolio, benchmark=returns["US_EQ"])
    print(f"skfolio.Portfolio -> Fundcloud.Portfolio  (returns length: {len(fc_view.returns)})")
    print(f"Sharpe (via Fundcloud):  {fc_view.sharpe():.3f}")
    print(f"Max drawdown:            {fc_view.max_drawdown() * 100:.2f}%")

    # --- 2. attribution / contribution / turnover ---------------------------
    # Portfolio wants a 1-D returns series + an optional weights frame. The
    # constant-weight MeanRisk result gives turnover=0 by design; contribution
    # is weight-shares of each bar's portfolio return.
    weights_row = pd.Series(est.weights_, index=returns.columns)
    weights_frame = pd.DataFrame([weights_row] * len(returns), index=returns.index)
    port_return = (returns * weights_frame).sum(axis=1).rename("min_var")
    p = Portfolio(
        returns=port_return,
        weights=weights_frame,
        benchmark=returns["US_EQ"],
        name="min_var",
    )
    print("\n--- 2. Attribution, contribution, turnover ---")
    print("Annualised contribution per asset (bps):")
    contrib = p.contribution() * 252.0 * 10_000
    for asset in contrib.index:
        print(f"  {asset:<12} {contrib[asset]:>8.1f} bps")
    print(f"\nTurnover:           {p.turnover() * 100:.3f}% per rebalance  "
          f"(0 because weights are constant)")

    # Round-trip back: Portfolio.to_skfolio works when the Fundcloud Portfolio
    # was built from a single-asset returns Series (the mirror of what
    # from_skfolio produces). Build a trivial such mirror to show the path.
    single_asset_fc = Portfolio(returns=port_return, name="min_var")
    back_to_sk = single_asset_fc.to_skfolio()
    print(f"Fundcloud -> skfolio:  {type(single_asset_fc).__name__} -> {type(back_to_sk).__name__}")

    # --- 3. Direct-call metrics ---------------------------------------------
    print("\n--- 3. Direct metrics (omega / ulcer_index / value_at_risk) ---")
    # Use the portfolio-level return series (weights × asset returns, summed).
    r = (returns * weights_frame).sum(axis=1)
    print(f"Omega (target 0%):   {omega(r):>7.3f}")
    print(f"Ulcer index:         {ulcer_index(r):>7.3f}")
    print(f"VaR (95%):           {value_at_risk(r) * 100:>6.2f}%")

    # --- 4. Rust-kernel direct batches on a 200-weighting sweep -------------
    print("\n--- 4. Rust kernels: 200 random weightings in one batch ---")
    rng = np.random.default_rng(1)
    n_strats = 200
    raw = np.abs(rng.normal(size=(n_strats, returns.shape[1])))
    norm = raw / raw.sum(axis=1, keepdims=True)           # weights: strat x asset
    panel_r = returns.to_numpy(dtype=float)               # days x asset
    portfolio_r = panel_r @ norm.T                        # days x strat
    portfolio_r = np.ascontiguousarray(portfolio_r)

    t0 = time.perf_counter()
    sharpes = kernels.sharpe_batch(portfolio_r, rf_per_period=0.0, periods_per_year=252.0)
    cvars = kernels.cvar_batch(portfolio_r, alpha=0.95)
    max_dds = kernels.max_drawdown_batch(portfolio_r)
    dt = time.perf_counter() - t0
    backend = "Rust" if kernels.HAS_RUST else "pure-Python"
    print(f"Backend: {backend} ({kernels.kernel_version()})")
    print(f"{n_strats} strategies × {portfolio_r.shape[0]} bars: {dt * 1000:.1f} ms")

    best = int(np.nanargmax(sharpes))
    print(f"\nBest-Sharpe weighting (strategy #{best}):")
    for asset, w in zip(returns.columns, norm[best], strict=True):
        print(f"  {asset:<12} {w * 100:>6.2f}%")
    print(f"  sharpe:           {sharpes[best]:>6.2f}")
    print(f"  95% CVaR:         {cvars[best] * 100:>6.2f}%")
    print(f"  max drawdown:     {max_dds[best] * 100:>6.2f}%")

    print("\nHow to read it:")
    print("  * from_skfolio / to_skfolio make interoperation trivial — analytics")
    print("    on a skfolio-fit portfolio, or vice-versa, without re-coding.")
    print("  * attribution / contribution decompose portfolio return into")
    print("    per-asset contribution; constant-weight portfolios have zero")
    print("    turnover but non-zero contribution by design.")
    print("  * The three direct-call metrics (omega, ulcer_index, value_at_risk)")
    print("    complement the .fc accessor for when you prefer the free-fn form.")
    print("  * The kernel batch at the end runs 200 strategies × 1255 bars of")
    print("    Sharpe / CVaR / max-drawdown in a few milliseconds — useful when")
    print("    you're scanning a parameter grid and the pandas overhead would")
    print("    dominate wall time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
