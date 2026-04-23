---
title: Aggregated summary
description: plots.summary composes the canonical panels into one headless Plotly figure.
---

# `plots.summary`

`plots.summary(returns)` returns a single `plotly.graph_objects.Figure` with the canonical panels of a strategy tear sheet. Use it when you want one aggregated figure you can drop into a notebook, a dashboard, or a standalone HTML file — without the cards/table machinery of [`Tearsheet`](../reports/tearsheets.md).

## Quick call

```python
from fundcloud import plots

fig = plots.summary(returns)                      # single strategy
fig = plots.summary(returns, benchmark=bench)     # overlay benchmark on cumulative
fig = plots.summary(returns, weights=weights_df)  # adds a composition row
fig.write_html("summary.html")
```

## Layout

Each panel is its own full-width row for readability:

| Row | Panel                            |
| --- | -------------------------------- |
| 1   | Cumulative returns               |
| 2   | Drawdown (%)                     |
| 3   | Rolling Sharpe                   |
| 4*  | Rolling alpha (annualised)       |
| 5*  | Rolling beta                     |
| 6   | Return distribution (%)          |
| 7   | Monthly returns heatmap          |
| 8†  | Portfolio composition            |

Rows marked with `*` appear only when `benchmark=` is supplied; the row marked with `†` only when `weights=` is supplied.

Every panel is rendered with `annotations=True`, so stats pills, the full-period Sharpe reference line, VaR / CVaR verticals, and annual totals appear automatically. Monthly heatmap cells now show their return percentage in-place so readers don't have to colour-match against the scale.

## Multi-asset input

Same contract as the individual builders — pass a `pandas.DataFrame` and every panel overlays per column, except the monthly heatmap which defaults to the **first** column. Pick a different asset with `heatmap_asset=`:

```python
plots.summary(returns_df, heatmap_asset="TSLA").show()
```

The subtitle shows which asset's heatmap is displayed, so readers can tell at a glance.

## Themes

`summary(..., theme="dark")` accepts the same aliases as [`set_theme`](themes.md). Without a theme kwarg, the currently active theme is applied.

## Rendering

`summary` returns a Figure — no side effects — so you drive output yourself:

```python
fig = plots.summary(returns)
fig.show()                     # inline in a notebook / browser
fig.write_html("summary.html") # self-contained, inline Plotly
fig.write_image("summary.png") # needs kaleido (uv add kaleido)
```

## Matplotlib version

`fundcloud.plots.mpl.summary(returns)` returns a `matplotlib.figure.Figure` built via `GridSpec`. It doesn't take a `theme` kwarg; otherwise the signature is identical. Use it when you want a static image without Plotly (e.g. embedding in a PDF).
