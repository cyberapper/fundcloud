"""25 — ``plots.summary``: one composed Plotly figure with every canonical panel.

Scenario: you don't want the full multi-format ``Tearsheet`` (HTML + PDF +
Excel), you just want one figure in a notebook or a standalone HTML file
that you can email to a friend. ``fundcloud.plots.summary`` composes the
cumulative curve, drawdown, rolling Sharpe, return distribution, and
monthly heatmap into a single :class:`plotly.graph_objects.Figure`; add a
``weights`` DataFrame for a fourth row showing portfolio composition.

Run:
    uv run python examples/25_plots_summary.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _synth import AssetProfile, close_returns, generate_ohlcv
from fundcloud import plots

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (returns_df, weights_df) for a two-asset 60/40-ish blend."""
    bars = generate_ohlcv(
        {
            "STOCKS": AssetProfile(mu=0.09, sigma=0.17, price0=450.0),
            "BONDS": AssetProfile(mu=0.02, sigma=0.05, price0=100.0),
        },
        periods=750,
        seed=17,
        correlations=np.array([[1.0, -0.1], [-0.1, 1.0]]),
    )
    returns = close_returns(bars)

    # Drifting equity allocation: between 50% and 75% stocks.
    rng = np.random.default_rng(17)
    n = len(returns)
    stocks_w = 0.6 + 0.05 * np.sin(np.linspace(0, 6.0, n)) + 0.02 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    stocks_w = np.clip(stocks_w, 0.5, 0.75)
    weights = pd.DataFrame(
        {"STOCKS": stocks_w, "BONDS": 1.0 - stocks_w},
        index=returns.index,
    )
    return returns, weights


def main() -> None:
    returns, weights = _synthetic()
    stocks_only = returns["STOCKS"].rename("60/40 — stocks leg")

    # 1. Single-series summary, no composition row
    simple = plots.summary(stocks_only, title="STOCKS only — 3-year summary")
    simple_path = OUT / "25_summary_series.html"
    simple.write_html(simple_path)

    # 2. Multi-asset summary with composition row
    strategy_returns = (returns * weights.shift(1).fillna(weights.iloc[0])).sum(axis=1).rename("60/40 blend")
    combined = pd.concat([strategy_returns, returns["STOCKS"].rename("pure stocks")], axis=1)
    full = plots.summary(
        combined,
        benchmark=returns["BONDS"].rename("bonds"),
        weights=weights,
        title="60/40 blend vs. pure stocks (bonds benchmark)",
    )
    full_path = OUT / "25_summary_full.html"
    full.write_html(full_path)

    # 3. Same summary in a non-default theme. set_theme is re-exported at
    # the top level so `import fundcloud as fc` can do `fc.set_theme(...)`.
    import fundcloud as fc

    fc.set_theme("dark")
    themed = plots.summary(combined, weights=weights, title="Dark theme")
    themed_path = OUT / "25_summary_dark.html"
    themed.write_html(themed_path)
    fc.set_theme("default")  # reset for subsequent scripts

    print(f"Wrote {simple_path.relative_to(HERE.parent)}  (single series)")
    print(f"Wrote {full_path.relative_to(HERE.parent)}   (multi-asset + weights + benchmark)")
    print(f"Wrote {themed_path.relative_to(HERE.parent)}  (dark theme)")


if __name__ == "__main__":
    main()
