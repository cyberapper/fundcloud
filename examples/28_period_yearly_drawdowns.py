"""28 — Period / yearly performance + drawdowns & runups.

Quantstats-style "where are we this month, this year, over the full
history?" plus the peak-to-valley / trough-to-peak episode tables that
show how the strategy has navigated its hardest moments.

All four surfaces are available as:

* stand-alone ``Portfolio`` methods — ``period_returns``,
  ``yearly_returns``, ``worst_drawdowns``, ``worst_runups``.
* rendered sections inside ``Tearsheet`` (HTML / PDF / Excel).

Run::

    uv add 'fundcloud[reports,viz]'
    uv run python examples/28_period_yearly_drawdowns.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    # Ten-year synthetic strategy + benchmark so the long-period
    # CAGRs (3Y / 5Y / 10Y) have enough data to be meaningful.
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-02", periods=2500, freq="B")
    strategy = pd.Series(rng.normal(0.0005, 0.012, 2500), index=idx, name="Strategy")
    spy = pd.Series(rng.normal(0.0003, 0.010, 2500), index=idx, name="SPY")

    pf = Portfolio(returns=strategy, benchmark=spy, name="Strategy")

    print("=== Period performance (MTD → All-time) ===")
    print(pf.period_returns().to_string(float_format=lambda x: f"{x * 100:7.2f}%"))

    print("\n=== Yearly returns ===")
    print(pf.yearly_returns().head(5).to_string(float_format=lambda x: f"{x * 100:7.2f}%"))
    print("...")

    print("\n=== Worst 5 drawdowns ===")
    print(pf.worst_drawdowns(top=5).to_string(index=False))

    print("\n=== Top 5 runups ===")
    print(pf.worst_runups(top=5).to_string(index=False))

    # Every one of these rows lands in the HTML / PDF / Excel tear sheet too.
    tear = Tearsheet(pf, benchmark=spy, title="Demo — period & episode analysis")
    html_path = OUT / "28_tear.html"
    tear.render_html(html_path)
    print(f"\nTear sheet with all new sections → {html_path.relative_to(HERE.parent)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
