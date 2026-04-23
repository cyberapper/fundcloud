---
title: Figure builders
description: The six plot builders — signatures, multi-asset behaviour, and the annotations kwarg.
---

# Figure builders

The seven builders in `fundcloud.plots` each return a `plotly.graph_objects.Figure`. Multi-asset input (`pandas.DataFrame`, one column per strategy) is supported on every builder except `monthly_heatmap` and `yearly_returns_bars` — see below.

## Signatures

| Builder                | Accepts             | Multi-asset? | Purpose                                         |
| ---------------------- | ------------------- | ------------ | ----------------------------------------------- |
| `cumulative`           | Series / DataFrame  | ✅           | Cumulative return (%) curve, starts at 0%       |
| `drawdown`             | Series / DataFrame  | ✅           | Underwater chart (drawdown %)                   |
| `rolling_sharpe`       | Series / DataFrame  | ✅           | Rolling annualised Sharpe                       |
| `return_distribution`  | Series / DataFrame  | ✅           | Per-period return histogram                     |
| `monthly_heatmap`      | Series only         | ❌           | Year × month aggregated returns                 |
| `yearly_returns_bars`  | Series / DataFrame  | ✅           | Paired grouped bars per year vs optional bench  |
| `composition`          | DataFrame (weights) | N/A          | Stacked-area portfolio weights                  |

Every builder shares these keyword arguments:

- `theme: str | None = None` — select a Plotly template for this figure only (see [Themes](themes.md)).
- `annotations: bool = False` — enable on-figure stats pills and reference lines (see below).
- `title: str = ...` — override the default title.

## Multi-asset behaviour

Passing a `pandas.DataFrame` overlays one trace (or histogram) per column:

```python
from fundcloud.data import YF

bars = YF(["SPY", "QQQ", "AGG"]).read(start="2020-01-01")
returns = bars.xs("close", axis=1, level=0).pct_change().dropna()
plots.cumulative(returns).show()   # three lines, one per ticker
```

`return_distribution` switches to overlay mode (`barmode="overlay"`, translucent bars). `drawdown` draws lines without fills when more than one column is present, so overlapping areas don't turn into mud.

!!! warning "Monthly heatmap is single-series only"
    A heatmap is a 2-D display; overlaying two would be unreadable. Pass one column:

    ```python
    plots.monthly_heatmap(returns["SPY"]).show()
    ```

    Or call [`plots.summary`](summary.md) — it picks the first column and renders the rest of the panels multi-asset.

## The `annotations` kwarg

`annotations=True` adds stats drawn from `fundcloud.metrics.core` directly to the figure:

| Builder               | What appears                                                             |
| --------------------- | ------------------------------------------------------------------------ |
| `cumulative`          | Stats pill per series: `Total`, `CAGR`, `Vol`, `Sharpe`                  |
| `drawdown`            | Stats pill: `Max DD`, peak → trough window, average drawdown             |
| `rolling_sharpe`      | Horizontal dashed line at the full-period Sharpe                         |
| `return_distribution` | Stats pill: `μ`, `σ`, `skew`, `kurt`, VaR₅, CVaR₅ + vertical refs for VaR / CVaR |
| `monthly_heatmap`     | Annual totals on the right margin                                        |
| `composition`         | Subtitle with average L1 turnover per period                             |

The default is `annotations=False` so the builders compose cleanly into the
`Tearsheet` class, which already shows these same stats in the `stat_cards` block. Call the builders directly with `annotations=True` when you want a single stand-alone figure.

## Benchmark overlays

`cumulative` accepts a `benchmark` kwarg (a `pandas.Series`). It is always drawn last, dashed grey, regardless of the primary input shape:

```python
from fundcloud.data import YF

strategy = Tearsheet.portfolio.returns                         # Series
bench    = (YF("SPY").read(start="2020-01-01")
                     .xs("close", axis=1, level=0)
                     .pct_change().dropna().squeeze("columns"))  # Series
plots.cumulative(strategy, benchmark=bench, annotations=True).show()
```

`yearly_returns_bars` also accepts a `benchmark` kwarg. The benchmark is
drawn as the first (amber) bar in each year-group, the strategy as the
second (blue) bar; a dashed red reference line marks the strategy's mean
yearly return:

```python
plots.yearly_returns_bars(strategy, benchmark=bench).show()
```

## Matplotlib mirrors

`fundcloud.plots.mpl` has the same six functions (`mpl.cumulative`, `mpl.drawdown`, …) returning `matplotlib.figure.Figure`. They accept the same DataFrame inputs and the same `annotations=` kwarg; they don't accept `theme=` (matplotlib theming is intentionally out of scope — see [Themes](themes.md) for why). The mpl builders are what `Tearsheet.render_pdf(..., engine="matplotlib")` uses internally.
