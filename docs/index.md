---
hide:
  - navigation
  - toc
title: Fundcloud — portfolio research, end-to-end
description: An open-source, portfolio-native research framework for serious self-directed investors and lean investment teams. Fluent pandas surface, sklearn-compatible estimators, Rust-accelerated kernels.
---

<div class="fc-hero" markdown>
<span class="fc-hero__eyebrow">Open-source · Python 3.10 → 3.14 · Rust core</span>

<div class="fc-hero__badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/fundcloud.svg)](https://pypi.org/project/fundcloud/)
[![Python](https://img.shields.io/pypi/pyversions/fundcloud.svg)](https://pypi.org/project/fundcloud/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/cyberapper/fundcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/cyberapper/fundcloud/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/cyberapper/fundcloud/branch/main/graph/badge.svg)](https://codecov.io/gh/cyberapper/fundcloud)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
</div>

<h1 class="fc-hero__title">
Portfolio research,<br>end-to-end, with a <span class="fc-grad">Rust core</span>.
</h1>

<p class="fc-hero__lede">
Fundcloud is a portfolio-native decision layer that takes you from market data to scored tear sheet in one coherent workflow — ingestion, strategy, simulation, optimisation, and reporting, under a single vocabulary. Newcomers get a fluent <code>.fc</code> accessor for any pandas Series or DataFrame; quants get sklearn-compatible estimators and a Rust-accelerated numeric core verified to ten-decimal parity with its NumPy reference.
</p>

<div class="fc-hero__ctas" markdown>
[Get started in 60 seconds](quickstart.md){ .fc-btn .fc-btn--primary }
[Install](install.md){ .fc-btn }
[:material-github: Repository](https://github.com/cyberapper/fundcloud){ .fc-btn }
</div>

<figure class="fc-preview">
  <img src="assets/fundcloud_lib.png" alt="Fundcloud library — portfolio research workflow from data to tear sheet" loading="lazy">
</figure>

<div class="fc-hero__meta" markdown>
<div class="fc-hero__meta-item">
<span class="fc-hero__meta-value">10–50×</span>
<span class="fc-hero__meta-label">Rust vs. NumPy, typical panels</span>
</div>
<div class="fc-hero__meta-item">
<span class="fc-hero__meta-value">1e-10</span>
<span class="fc-hero__meta-label">Rust / NumPy parity</span>
</div>
<div class="fc-hero__meta-item">
<span class="fc-hero__meta-value">1 wheel</span>
<span class="fc-hero__meta-label">abi3-py310, all supported versions</span>
</div>
<div class="fc-hero__meta-item">
<span class="fc-hero__meta-value">MIT</span>
<span class="fc-hero__meta-label">License</span>
</div>
</div>
</div>

## Two ways to start

<div class="fc-tracks" markdown>

<div class="fc-track" markdown>
<span class="fc-track__label">For investors & researchers</span>
<h3 class="fc-track__title">Research a portfolio, fast</h3>
<p class="fc-track__lede">
Pull market data, run a strategy, score the result, and export a tear sheet — without stitching together five libraries or a notebook chain.
</p>

<ul class="fc-track__list">
<li><code>returns.fc.sharpe()</code> on any pandas Series</li>
<li>Daily, weekly, and monthly DCA with adjustable horizons</li>
<li>Self-contained HTML, PDF, and Excel tear sheets — or one composed <code>plots.summary()</code> figure</li>
<li>Plotly themes (<code>white</code>, <code>dark</code>, <code>ggplot2</code>, <code>seaborn</code>) via <code>fc.set_theme</code></li>
<li>Mean-risk, HRP, and HERC optimisers behind one call</li>
</ul>

[Run the 20-line quickstart <span></span>](quickstart.md){ .fc-track__cta }
</div>

<div class="fc-track" markdown>
<span class="fc-track__label">For Python & Rust developers</span>
<h3 class="fc-track__title">Integrate the library</h3>
<p class="fc-track__lede">
A small, composable API that plugs into sklearn pipelines and exposes Rust kernels through PyO3. No global state, no framework lock-in, typed end to end.
</p>

<ul class="fc-track__list">
<li><code>PurgedKFold</code>, estimators, and scorers that plug into <code>GridSearchCV</code></li>
<li><code>Backend → Catalog → Backend</code> data layer (YF · CSV · Parquet · DuckDB · in-memory)</li>
<li>Single <code>abi3</code> wheel per platform; pure-Python fallback everywhere else</li>
<li><code>numpy</code> dtype-preserving interfaces, fully type-annotated</li>
</ul>

[Browse the API reference <span></span>](reference/data.md){ .fc-track__cta }
</div>

</div>

## What's in the box

<div class="fc-grid fc-grid--3" markdown>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Data</div>
<h4 class="fc-card__title">Pull and cache market data</h4>
<p class="fc-card__body">Grab bars from Yahoo / FMP / Alpha Vantage / Binance, cache them in DuckDB or Parquet with one line. Daily refresh dedupes automatically — no duplicate rows, ever.</p>
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Strategies</div>
<h4 class="fc-card__title">DCA, Hold, and a base class</h4>
<p class="fc-card__body">Presets with daily, weekly (7-day), and monthly horizons. Subclass <code>BaseStrategy</code> for anything bespoke.</p>
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Simulator</div>
<h4 class="fc-card__title">One Portfolio, four inputs</h4>
<p class="fc-card__body"><code>run_strategy</code>, <code>run_weights</code>, <code>run_signals</code>, <code>run_orders</code> — every path yields a live <code>Portfolio</code>.</p>
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Optimise</div>
<h4 class="fc-card__title">Mean-risk, HRP, HERC</h4>
<p class="fc-card__body">Portfolio construction through a thin adapter — swap objectives, constraints, and risk measures without rewriting the pipeline.</p>
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Reports</div>
<h4 class="fc-card__title">HTML, PDF, Excel</h4>
<p class="fc-card__body">Self-contained HTML tear sheets, WeasyPrint PDFs, and XlsxWriter workbooks with native charts. One call per format.</p>
</div>

<div class="fc-card" markdown>
<div class="fc-card__kicker">Speed</div>
<h4 class="fc-card__title">Rust kernels, NumPy fallback</h4>
<p class="fc-card__body">Rolling stats, drawdown, Sharpe, VaR, and CVaR compiled via PyO3. Transparent fallback, verified to 1e-10 parity.</p>
</div>

</div>

## 60-second check

```python
import pandas as pd
import fundcloud  # registers the .fc accessor

returns = pd.Series([0.012, -0.005, 0.008, -0.010, 0.015])
returns.fc.sharpe(periods_per_year=252)   # annualised Sharpe
returns.fc.max_drawdown()                 # worst peak-to-trough
returns.fc.drawdown_series()              # full drawdown time series
```

The full DCA-to-tearsheet walkthrough is in [Quickstart](quickstart.md).

## Design principles

- **One vocabulary.** `Data`, `Trades`, `Portfolio` mean what they mean in finance. No framework-specific renames to translate.
- **Two doors.** Fluent `.fc` one-liners for exploration, full sklearn estimators for production pipelines — same objects, same numbers.
- **Honest benchmarks.** Rust kernels run 10–50× faster than the NumPy fallback on typical panel sizes, and every kernel is tested to 1e-10 agreement with its reference implementation. See [Rust kernels](guides/accelerators/rust-kernels.md) for the methodology.
- **Cheap core install.** Core pulls only pandas, numpy, scipy, scikit-learn, and plotly. Everything heavy lives behind extras: `[pf]`, `[ta]`, `[data]`, `[reports]`.
- **Reproducible by default.** Deterministic simulation, seeded optimisers, and tear sheets that embed the exact parameters used to produce them.

<div class="fc-cta" markdown>
<div>
<h3 class="fc-cta__title">Ready to try it on your own portfolio?</h3>
<p class="fc-cta__lede">One <code>uv add fundcloud</code>, a <code>Series</code>, and you're running Sharpe and drawdown.</p>
</div>
<div class="fc-cta__actions" markdown>
[Install](install.md){ .fc-btn .fc-btn--gradient }
[Read the quickstart](quickstart.md){ .fc-btn }
</div>
</div>

---

## Acknowledgments

Fundcloud builds on the work of several excellent open-source projects. If any of them are useful to you, please consider giving them a star on GitHub.

| Project | Contribution to Fundcloud | License |
|---|---|---|
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Estimator, transformer, and CV-splitter contracts throughout | BSD-3-Clause |
| [skfolio](https://github.com/skfolio/skfolio) | Portfolio optimisation algorithms and `Portfolio`/`Population` objects | BSD-3-Clause |
| [TA-Lib](https://github.com/TA-Lib/ta-lib-python) | 170+ technical indicators via `fundcloud[ta]` | BSD-2-Clause |
| [quantstats](https://github.com/ranaroussi/quantstats) | Tear-sheet design and pandas-accessor philosophy | Apache-2.0 |
| [vectorbt](https://github.com/polakowo/vectorbt) | Vectorised simulation design | Apache-2.0 |
| [PyO3](https://github.com/PyO3/pyo3) | Rust–Python bridge for kernels | Apache-2.0 / MIT |
| [uv](https://github.com/astral-sh/uv) | Fast installation toolchain | Apache-2.0 / MIT |

Full license attribution lives in the [NOTICE](https://github.com/cyberapper/fundcloud/blob/main/NOTICE) file.
