---
title: Plots
description: Figure builders, theming, and one-call aggregated summaries — the lower-level surface beneath the tear sheet.
---

# Plots

`fundcloud.plots` is the layer beneath the [Tear sheet](../reports/tearsheets.md). Six figure builders produce standalone Plotly figures; a seventh function — [`summary`](summary.md) — composes them into a single `Figure` ready for HTML / image export. All figures are returned headless (no `fig.show()`): you choose where they go.

!!! tip "When to use which surface"
    - `fundcloud.reports.Tearsheet` — polished multi-format report (HTML / PDF / Excel) with stat cards and footer metadata.
    - `fundcloud.plots.summary` — a single composed `plotly.graph_objects.Figure` you can drop into a notebook or dashboard.
    - Individual builders (`cumulative`, `drawdown`, …) — when you want one panel and nothing else.

## At a glance

```python
import fundcloud as fc
from fundcloud import plots
from fundcloud.data import YF

# Pull bars and derive returns (one line apiece)
bars = YF("SPY").read(start="2020-01-01")
returns = bars.xs("close", axis=1, level=0).pct_change().dropna().squeeze("columns")

# Single-asset figures
plots.cumulative(returns).show()
plots.drawdown(returns, annotations=True).show()

# Multi-asset comparison — any DataFrame of returns works
multi_bars = YF(["SPY", "QQQ", "AGG"]).read(start="2020-01-01")
comparison = multi_bars.xs("close", axis=1, level=0).pct_change().dropna()
plots.cumulative(comparison, annotations=True).show()

# One composed figure
plots.summary(comparison, benchmark=returns).write_html("summary.html")

# Switch theme globally (plotly only)
fc.set_theme("dark")    # also available as plots.set_theme
```

## In this section

- **[Builders](builders.md)** — signatures, multi-asset behaviour, and the `annotations` kwarg.
- **[Themes](themes.md)** — the `set_theme` alias map and how to plug in a custom Plotly template.
- **[Summary](summary.md)** — the aggregation function, panel order, and composition-row support.

For the Python API reference see [reference/plots](../../reference/plots.md).

## Matplotlib mirror

Every builder has a matplotlib twin under `fundcloud.plots.mpl` (requires the `fundcloud[viz]` extra). It is used internally by the PDF renderer and is available to you for the same reason — PDF / notebook embedding without the Plotly runtime. The mpl builders accept the same DataFrame inputs and `annotations=` kwarg; theming (`set_theme`) is a Plotly-only concern.
