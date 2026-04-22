"""11 — Investment horizon: how holding period changes the picture.

Trader question: "I'm a long-term investor with a 5-year horizon. Should
I care about the same metrics as a swing trader with a 2-week horizon?"

Answer: absolutely not — but Fundcloud gives you two levers:

1. **Frequency resampling** — convert daily returns to weekly/monthly
   before running the analysis, so the optimisation sees the horizon
   you actually care about.
2. **Annualisation factor** — scale metrics by the ``periods_per_year``
   you expect (252 daily, 52 weekly, 12 monthly). Use the
   ``fundcloud._config.config(...)`` context manager to scope this
   locally without mutating global state.

We pull 5 years of real SPY / VGK / AGG / GLD closes via yfinance and
inspect the same strategy through three lenses, then re-run a mean-
variance optimisation at each horizon to show how allocations shift.

Run:
    uv add 'fundcloud[data-yf]'
    uv run python examples/11_investment_horizon.py
"""

from __future__ import annotations

from pathlib import Path

import fundcloud  # noqa: F401  — registers the .fc accessor
import numpy as np
from _data import pull_closes
from fundcloud._config import config
from fundcloud.metrics import returns_stats
from fundcloud.optimize import MVO

HERE = Path(__file__).parent


HORIZONS = [
    ("daily", "1D", 252),
    ("weekly", "1W", 52),
    ("monthly", "1ME", 12),
]


def main() -> int:
    closes = pull_closes(
        {
            "US_EQ": "SPY",
            "EU_EQ": "VGK",
            "BONDS_AGG": "AGG",
            "GOLD": "GLD",
        },
        years=5,
    )
    if closes is None or closes.empty:
        return 1

    print(
        f"Live data:  {closes.index[0].date()} → {closes.index[-1].date()}  "
        f"({len(closes)} trading days, {closes.shape[1]} assets)\n"
    )

    # --------------------------------------------------------------- metrics
    print(f"{'Horizon':<10}{'n_obs':>8}{'cagr':>12}{'ann_vol':>10}{'sharpe':>10}{'max_dd':>10}")
    print("-" * 60)
    for name, rule, ppy in HORIZONS:
        if rule == "1D":
            resampled = closes.pct_change().dropna()
        else:
            resampled = closes.resample(rule).last().pct_change().dropna()
        with config(periods_per_year=ppy):
            stats = returns_stats(resampled, periods_per_year=ppy)
        ann_ret = stats.loc["cagr"].mean() * 100
        ann_vol = stats.loc["ann_volatility"].mean() * 100
        sharpe = stats.loc["sharpe"].mean()
        mdd = stats.loc["max_drawdown"].mean() * 100
        print(
            f"{name:<10}{len(resampled):>8}{ann_ret:>11.2f}%{ann_vol:>9.2f}%"
            f"{sharpe:>10.2f}{mdd:>9.2f}%"
        )

    # --------------------------------------------------------------- allocations
    print("\nMVO (max-Sharpe) weights at each horizon:\n")
    header = f"{'asset':<12}" + "".join(f"{n:>14}" for n, _, _ in HORIZONS)
    print(header)
    print("-" * len(header))
    weights_by_horizon: dict[str, dict[str, float]] = {}
    for name, rule, _ppy in HORIZONS:
        if rule == "1D":
            resampled = closes.pct_change().dropna()
        else:
            resampled = closes.resample(rule).last().pct_change().dropna()
        est = MVO(l2=1e-4).fit(resampled)
        weights_by_horizon[name] = dict(
            zip(resampled.columns, np.asarray(est.weights_, dtype=float), strict=True)
        )

    for asset in closes.columns:
        row = f"{asset:<12}"
        for name, _, _ in HORIZONS:
            row += f"{weights_by_horizon[name][asset] * 100:>13.2f}%"
        print(row)

    print("\nReading the table:")
    print("  * daily → the optimiser sees the full sample — maximum statistical")
    print("    precision, but also maximum short-term noise.")
    print("  * monthly → ~60 observations; weights tend to be more extreme")
    print("    because the sample covariance is noisier.")
    print("  * long-horizon investors can legitimately prefer monthly / weekly")
    print("    resampling so the metrics they report match the decision they're making.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
