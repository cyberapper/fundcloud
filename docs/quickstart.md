---
title: Quickstart
description: Synthetic OHLCV panel → DCA strategy → tear sheet → sklearn pipeline, in about 20 lines.
---

# Quickstart

End-to-end DCA backtest in about 20 lines. No real data required — the snippet synthesises a reproducible two-asset panel so you can paste the whole thing into a REPL.

!!! tip "What you'll have at the end"
    A `Portfolio` with Sharpe and drawdown, an HTML tear sheet, and the same pipeline running inside `GridSearchCV` across purged folds.

## Instant returns check

If you already have a returns Series — from a broker export or notebook — you're three lines from a full metric bundle:

```python
import fundcloud  # registers .fc on pandas
returns.fc.metrics()   # Sharpe, Sortino, Calmar, drawdown, CVaR, and more
returns.fc.plot_cumulative().show()
```

Not sure where to go from here? The [Returns analysis](guides/portfolio/returns-analysis.md) guide walks through the full `.fc` surface.

## Mental model

Fundcloud treats every research session as four passes over a single shared object graph:

1. **Bars** — an OHLCV panel with a `(field, asset)` column MultiIndex and a `DatetimeIndex`. This is the one canonical input every downstream path consumes.
2. **Strategy** — a small, deterministic function of the `Bars` that emits either weights, signals, or orders. `DCA` and `Hold` are the shipped presets; subclass `BaseStrategy` for anything bespoke.
3. **Simulator** — turns any of the strategy outputs into a live `Portfolio` under the cost, slippage, and fill-timing assumptions you choose.
4. **Reports** — the `Portfolio` renders to metrics, tear sheets (HTML / PDF / Excel), and any sklearn-scored pipeline.

The four numbered sections below walk through exactly one full pass; deeper treatment of each stage lives in [Guides](guides/data/backends-and-catalog.md).

## 1. Build a Bars frame

In a notebook you'd pull from yfinance; here we synthesise a tiny two-asset
OHLCV panel for reproducibility.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=252, freq="B").values)

def asset(price0: float, vol: float):
    close = price0 + np.cumsum(rng.normal(0, vol, len(idx)))
    return {
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1_000_000.0,
    }

bars = pd.concat(
    {"AAPL": pd.DataFrame(asset(180.0, 1.2), index=idx),
     "MSFT": pd.DataFrame(asset(400.0, 2.0), index=idx)},
    axis=1,
)
bars.columns = bars.columns.swaplevel(0, 1)  # (field, asset)
bars = bars.sort_index(axis=1)
```

## 2. Run a DCA strategy

```python
from fundcloud.sim import Simulator
from fundcloud.strategies import DCA

result = Simulator(bars, cash=100_000).run_strategy(
    DCA(
        amount=1_000,
        horizon="weekly",
        weights={"AAPL": 0.5, "MSFT": 0.5},
    )
)

result.portfolio.sharpe()
result.portfolio.max_drawdown()
result.equity_curve.tail()
```

## 3. Produce a tear sheet

```python
from fundcloud.reports import Tearsheet

ts = Tearsheet(result.portfolio, title="DCA weekly")
ts.render_html("dca.html")
ts.render_pdf("dca.pdf")     # needs fundcloud[reports]
ts.render_excel("dca.xlsx")  # needs fundcloud[reports]
```

Or, if you just want a single composed figure for a notebook / dashboard:

```python
import fundcloud as fc

fc.set_theme("dark")                                         # optional
fc.plots.summary(result.portfolio.returns).write_html("quick.html")
```

`plots.summary` returns one `plotly.graph_objects.Figure` with the cumulative curve, drawdown, rolling Sharpe, return distribution, and monthly heatmap already composed. See [Plots → Summary](guides/plots/summary.md) for the full layout.

## 4. Plug into sklearn

!!! note "Extras required"
    `RSI` and `SMA` need `fundcloud[ta]`; `MeanRisk` needs `fundcloud[pf]`.
    Install both: `uv add 'fundcloud[ta,pf]'`

Every Fundcloud transformer, CV splitter, and optimiser is sklearn-native:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from fundcloud.features import FeaturePipeline
from fundcloud.features.indicators import RSI, SMA
from fundcloud.optimize import MeanRisk, RiskMeasure
from fundcloud.validate import PurgedKFold

pipe = Pipeline([
    ("features", FeaturePipeline([("rsi", RSI(timeperiod=14)), ("sma", SMA(timeperiod=20))])),
    ("optim",    MeanRisk(risk_measure=RiskMeasure.CVAR)),
])
# cross-validated across purged folds — no data leakage
grid = GridSearchCV(pipe, param_grid={"optim__min_weights": [0.0, 0.02, 0.05]},
                    cv=PurgedKFold(n_splits=5, purge=3))
```

## Where to go next

<div class="fc-grid fc-grid--3" markdown>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Data</div>
<h4 class="fc-card__title">Pulling and caching market data</h4>
<p class="fc-card__body">Grab bars from Yahoo / FMP / Alpha Vantage / Binance and cache them in DuckDB or Parquet with a single <code>.sync_to(...)</code> call. Incremental refresh handled for you.</p>
[Go →](guides/data/backends-and-catalog.md){ .fc-track__cta }
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Strategies</div>
<h4 class="fc-card__title">DCA & Hold in depth</h4>
<p class="fc-card__body">Horizon semantics, rebalancing rules, and where <code>BaseStrategy</code> lets you extend.</p>
[Go →](guides/strategies/dca.md){ .fc-track__cta }
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Simulator</div>
<h4 class="fc-card__title">Costs, slippage, execution</h4>
<p class="fc-card__body">How the four <code>run_*</code> entry points compose and what assumptions they encode.</p>
[Go →](guides/sim/simulator.md){ .fc-track__cta }
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Interop</div>
<h4 class="fc-card__title">sklearn & skfolio</h4>
<p class="fc-card__body">Estimator API, <code>PurgedKFold</code>, and the skfolio adapter — with and without the <code>[pf]</code> extra.</p>
[Go →](guides/interop/sklearn.md){ .fc-track__cta }
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Optimise</div>
<h4 class="fc-card__title">Portfolio optimisation</h4>
<p class="fc-card__body">HRP quick win → risk measures → constraints and walk-forward validation. Seven optimisers, one API.</p>
[Go →](guides/portfolio/optimization.md){ .fc-track__cta }
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Speed</div>
<h4 class="fc-card__title">Rust kernels</h4>
<p class="fc-card__body">What's accelerated, by how much, and how the 1e-10 NumPy parity is verified.</p>
[Go →](guides/accelerators/rust-kernels.md){ .fc-track__cta }
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Reports</div>
<h4 class="fc-card__title">Tear sheets</h4>
<p class="fc-card__body">HTML, PDF, and Excel output — what's in each format and how to customise sections.</p>
[Go →](guides/reports/tearsheets.md){ .fc-track__cta }
</div>

</div>
