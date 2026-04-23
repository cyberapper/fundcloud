"""27 — Benchmark-relative analytics, end-to-end.

Every time you build a tear sheet with ``benchmark=`` Fundcloud threads the
benchmark through four surfaces:

1. ``fundcloud.metrics.metrics(r, benchmark=b)`` — adds alpha, beta,
   correlation, R², information ratio, tracking error, up/down capture,
   capture ratio, Treynor.
2. ``plots.summary(r, benchmark=b)`` — overlays the benchmark on the
   cumulative panel and appends rolling-α / rolling-β rows.
3. ``Tearsheet(portfolio, benchmark=b).render_html`` — right-hand sidebar
   grows a ``Benchmark`` section; a rolling-α/β panel joins the chart list.
4. ``render_pdf`` / ``render_excel`` — PDF gets a dedicated benchmark page,
   Excel gets a ``Benchmark`` sheet with the aligned return series and a
   metric-by-side comparison table.

Run:
    uv run python examples/27_benchmark_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import fundcloud as fc
import numpy as np
import pandas as pd
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _synthetic() -> tuple[pd.Series, pd.Series]:
    """Return (strategy, benchmark) — benchmark is a low-vol drift series
    that the strategy's alpha component ought to beat."""
    rng = np.random.default_rng(19)
    idx = pd.bdate_range("2018-01-02", periods=1_512)
    benchmark = pd.Series(rng.normal(0.0004, 0.009, len(idx)), index=idx, name="SPY")
    # Strategy = 1.1 × benchmark + alpha-like residual.
    residual = rng.normal(0.0003, 0.006, len(idx))
    strategy = (1.1 * benchmark + residual).rename("my_strategy")
    return strategy, benchmark


def main() -> None:
    strategy, benchmark = _synthetic()

    # 1. Metrics — benchmark-relative keys appear automatically. The same
    # call can take a column name when ``strategy`` sits inside a wide
    # DataFrame — see the multi-asset block at the end.
    m = strategy.fc.metrics(benchmark=benchmark)
    rows = [
        "cagr",
        "sharpe",
        "alpha",
        "beta",
        "correlation",
        "information_ratio",
        "up_capture",
        "down_capture",
    ]
    print("Key benchmark metrics:")
    for key in rows:
        if key in m.index:
            val = m[key]
            print(f"  {key:>18}:  {val:.4f}")

    # 2. plots.summary — rolling alpha/beta rows appear because benchmark is set.
    fig = fc.plots.summary(strategy, benchmark=benchmark, title="Strategy vs SPY — synthetic demo")
    summary_path = OUT / "27_summary.html"
    fig.write_html(summary_path)

    # 3 + 4. Tearsheet — HTML / PDF / Excel all thread the benchmark.
    ts = Tearsheet(
        Portfolio(returns=strategy, benchmark=benchmark, name="my_strategy"),
        benchmark=benchmark,
        title="Strategy vs SPY — benchmarked tear sheet",
    )
    html_path = ts.render_html(OUT / "27_tearsheet.html")
    pdf_path = ts.render_pdf(OUT / "27_tearsheet.pdf")
    xlsx_path = ts.render_excel(OUT / "27_tearsheet.xlsx")

    print()
    print(
        f"Wrote {summary_path.relative_to(HERE.parent)}  (summary figure, 7 panels incl. rolling α/β)"
    )
    print(
        f"Wrote {html_path.relative_to(HERE.parent)}  (HTML tear sheet + benchmark sidebar section)"
    )
    print(f"Wrote {pdf_path.relative_to(HERE.parent)}  (A4 portrait, with rolling α/β page)")
    print(f"Wrote {xlsx_path.relative_to(HERE.parent)}  (adds a Benchmark sheet)")

    # ------------------------------------------------------------------ string
    # Benchmark can also be a column name. Fundcloud resolves the string
    # against the DataFrame and excludes that column from per-asset tabs /
    # sheets / sections so SPY doesn't appear as "SPY vs SPY".
    panel = pd.concat(
        {"strategy_a": strategy, "strategy_b": strategy * 1.05, "SPY": benchmark},
        axis=1,
    )
    panel.fc.render_html(OUT / "27_multi_string_bench.html", benchmark="SPY")
    panel.fc.plot_summary(benchmark="SPY", heatmap_asset="strategy_b").write_html(
        OUT / "27_summary_heatmap_strategy_b.html"
    )
    print(
        f"Wrote {(OUT / '27_multi_string_bench.html').relative_to(HERE.parent)}  "
        "(benchmark='SPY' resolved against DataFrame columns)"
    )
    print(
        f"Wrote {(OUT / '27_summary_heatmap_strategy_b.html').relative_to(HERE.parent)}  "
        "(heatmap_asset picks which column's monthly heatmap to show)"
    )


if __name__ == "__main__":
    main()
