---
title: Benchmark analytics
description: How to thread a benchmark through metrics, the summary figure, and the HTML / PDF / Excel tear sheets.
---

# Benchmark analytics

Pass a benchmark series (same frequency as your strategy returns) to unlock
the full benchmark-relative analytical surface in one shot.

## Metrics

```python
import pandas as pd
import fundcloud  # registers the .fc accessor

m = strategy.fc.metrics(benchmark=spy)
m.loc[["alpha", "beta", "correlation", "r_squared",
       "information_ratio", "tracking_error",
       "up_capture", "down_capture", "capture_ratio", "treynor_ratio"]]
```

!!! tip "String benchmarks"
    When your strategies sit in a wide DataFrame, you can pass the benchmark's
    column name instead of slicing the Series manually:

    ```python
    panel = pd.DataFrame({"s1": s1, "s2": s2, "SPY": spy})
    panel.fc.render_html("out.html", benchmark="SPY")
    panel.fc.plot_summary(benchmark="SPY").show()
    ```

    Fundcloud pulls SPY out of the frame and drops it from the per-asset
    rendering so you never get "SPY vs SPY" as a tab.

| Key | Meaning | Formula |
| --- | --- | --- |
| `alpha` | Jensen's annualised alpha | `ann(r − rf) − β · ann(bench − rf)` |
| `beta` | Market sensitivity | `cov(r, bench) / var(bench)` |
| `correlation` | Pearson correlation | `cov / (σ_r · σ_b)` |
| `r_squared` | Share of variance explained | `correlation²` |
| `information_ratio` | Active return / active σ | `mean(r − bench) / σ(r − bench)` |
| `tracking_error` | Annualised active σ | `σ(r − bench) · √periods_per_year` |
| `up_capture` | Participation on up days | `mean(r ∣ bench > 0) / mean(bench ∣ bench > 0)` |
| `down_capture` | Participation on down days | `mean(r ∣ bench < 0) / mean(bench ∣ bench < 0)` |
| `capture_ratio` | Morningstar single number | `up_capture / down_capture` |
| `treynor_ratio` | Excess return per unit of beta | `ann(r − rf) / β` |

Rolling variants: `fundcloud.metrics.rolling_alpha(r, bench, window=63)` and
`rolling_beta`. Both inner-align strategy and benchmark on the shared
trading calendar before the rolling covariance — essential when a
7-day/week series (crypto) is compared against a 5-day/week series
(futures, equities), otherwise NaNs from the calendar mismatch propagate
across the rolling window and empty the output.

## Summary figure

```python
from fundcloud import plots

plots.summary(strategy, benchmark=spy).show()
```

With `benchmark=` supplied, two extra full-width rows appear between the
rolling-Sharpe and the return-distribution panels:

* **Rolling alpha (annualised)** — positive = out-performing what beta alone predicts.
* **Rolling beta** — drift in market sensitivity over time.

The cumulative panel already overlays the benchmark as a dashed line, and
the stats pill automatically gains a benchmark row.

## HTML / PDF / Excel tear sheets

```python
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet

ts = Tearsheet(Portfolio(returns=strategy, name="my"),
               benchmark=spy, title="Strategy vs SPY")

ts.render_html("out.html")       # sidebar grows a "Benchmark" section
ts.render_pdf("out.pdf")         # extra rolling α/β page
ts.render_excel("out.xlsx")      # adds a "Benchmark" sheet
```

**HTML.** The right-hand sidebar renders a `Benchmark` accordion with every
benchmark-relative metric; each row shows the strategy value in bold and
the benchmark's own value as a muted second column so readers can compare
directly. Tooltips on every `?` badge explain the definition and formula.

**PDF.** An extra A4-portrait page titled "Benchmark dynamics (rolling
63-bar)" renders rolling alpha on top and rolling beta on the bottom.

**Excel.** A new `Benchmark` sheet ships the aligned strategy + benchmark
return series in columns A–C, plus a metric-by-side table in columns F–H
that puts strategy values next to the benchmark's own readings.

## Runnable example

See [`examples/27_benchmark_comparison.py`](https://github.com/cyberapper/fundcloud/blob/main/examples/27_benchmark_comparison.py)
for the full end-to-end walkthrough.
