# Fundcloud

> Portfolio research, end-to-end, with a Rust core.

[![PyPI](https://img.shields.io/pypi/v/fundcloud.svg)](https://pypi.org/project/fundcloud/)
[![Python](https://img.shields.io/pypi/pyversions/fundcloud.svg)](https://pypi.org/project/fundcloud/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/cyberapper/fundcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/cyberapper/fundcloud/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/cyberapper/fundcloud/branch/main/graph/badge.svg)](https://codecov.io/gh/cyberapper/fundcloud)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://cyberapper.github.io/fundcloud)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

![Fundcloud library overview](docs/assets/fundcloud_lib.png)

Fundcloud is a beginner-friendly, headless-for-advanced **portfolio research framework**. One install covers returns and risk analytics, drawdown analysis, portfolio optimisation, vectorised backtesting, technical indicators, purged cross-validation, multi-source market data loading, exploratory analysis, and HTML/PDF/Excel tear sheets — through a coherent `.fc` pandas surface for beginners and a full `sklearn`-compatible estimator API for advanced users. Matrix-heavy math lives in a Rust core via PyO3 and ships as a single abi3 wheel per platform.

## Install

```bash
uv add fundcloud                  # core
uv add "fundcloud[data]"          # + all network data providers (yf, fmp, av, binance)
uv add "fundcloud[pf,ta,data]"    # + skfolio + TA-Lib + data sources
uv add "fundcloud[all]"           # everything
```

| Extra | Adds |
|---|---|
| `pf` | [skfolio](https://github.com/skfolio/skfolio) — portfolio optimisation |
| `ta` | [TA-Lib](https://github.com/TA-Lib/ta-lib-python) — 170+ technical indicators |
| `data-yf` / `data-fmp` / `data-av` / `data-bn` | individual data providers |
| `data` | bundle of every data provider above |
| `viz` | matplotlib + kaleido (static plot export) |
| `reports` | WeasyPrint (PDF) + XlsxWriter (Excel) |
| `all` | everything above |

Exploratory data analysis (`fundcloud.explore.{profile, compare, quickview}`) ships in core — no extra needed.

## Quickstart (60 seconds)

```python
import pandas as pd
import fundcloud  # registers the .fc accessor on pandas

# Any returns Series gets instant analytics
returns = pd.Series([0.012, -0.005, 0.008, -0.010, 0.015], name="strategy")
returns.fc.sharpe(periods=252)           # annualised Sharpe
returns.fc.max_drawdown()
returns.fc.drawdown_series()

# Purged CV that plugs into sklearn out of the box
from fundcloud.validate import PurgedKFold
from sklearn.model_selection import cross_val_score

cv = PurgedKFold(n_splits=5, purge=3, embargo=1)
# cross_val_score(estimator, X, y, cv=cv)   # drop-in
```

The library ships DCA/Hold strategies, a simulator, skfolio-backed optimisers, native EDA, and HTML/PDF/Excel tear sheets out of the box. Prefer one composed figure over a full report? `fundcloud.plots.summary(returns)` returns a multi-panel `plotly.graph_objects.Figure` (cumulative, drawdown, rolling Sharpe, distribution, monthly heatmap) with Plotly theme support via `fc.set_theme("dark")` (re-exported at the top level for `import fundcloud as fc`); every builder also accepts multi-asset DataFrames so comparisons stay one line.

## sklearn & skfolio interop

Every Fundcloud estimator, transformer, and CV splitter passes `sklearn.utils.estimator_checks.check_estimator` and round-trips through skfolio. Example:

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
search = GridSearchCV(pipe, param_grid={"optim__min_weights": [0.0, 0.02, 0.05]},
                     cv=PurgedKFold(n_splits=5, purge=3))
search.fit(returns_panel)
```

## Architecture

```
 ┌────────────────────────────────────────────────────────────────────┐
 │  End-user surfaces:  fluent accessor .fc   |   estimator API       │
 ├───────────────┬──────────────┬───────────────┬──────────────────────┤
 │   reports     │   explore    │    plots      │    datasets          │
 ├───────────────┴──────────────┴───────────────┴──────────────────────┤
 │                       metrics │ validate │ optimize                 │
 ├──────────────────────────┬──────────────────────────────────────────┤
 │      portfolio           │                    sim                   │
 ├──────────────────────────┼──────────────────────────────────────────┤
 │       strategies         │                features                  │
 ├──────────────────────────┴──────────────────────────────────────────┤
 │                              data                                   │
 │  Backends (YF, FMP, …, Parquet, DuckDB, Memory, CSV) ─ Catalog      │
 ├─────────────────────────────────────────────────────────────────────┤
 │                   kernels  (Rust, PyO3, abi3)                       │
 └─────────────────────────────────────────────────────────────────────┘
```

## Python compatibility

Supported on Python **3.10, 3.11, 3.12, 3.13, 3.14**. Wheels are built with PyO3's `abi3-py310` feature, so one wheel per platform works across every supported version.

## Acknowledgments

Fundcloud stands on the shoulders of excellent open-source work:

- **scikit-learn** (BSD-3-Clause) — estimator, transformer, and CV-splitter contracts used throughout.
- [**skfolio**](https://github.com/skfolio/skfolio) (BSD-3-Clause) — portfolio optimisation algorithms; `Portfolio`/`Population` objects. Install with `uv add 'fundcloud[pf]'`.
- [**quantstats**](https://github.com/ranaroussi/quantstats) (Apache-2.0) — inspiration for our tear-sheet and pandas-accessor design.
- [**vectorbt**](https://github.com/polakowo/vectorbt) (Apache-2.0) and [**vectorbt.pro**](https://vectorbt.pro) — inspiration for the vectorised simulation model.
- [**TA-Lib / ta-lib-python**](https://github.com/TA-Lib/ta-lib-python) (BSD-2-Clause) — all 170+ technical indicators in `fundcloud.features.indicators`.
- [**PyO3**](https://github.com/PyO3/pyo3), [**rust-numpy**](https://github.com/PyO3/rust-numpy), [**maturin**](https://github.com/PyO3/maturin), [**uv**](https://github.com/astral-sh/uv) — the build-and-ship story.

See [`NOTICE`](NOTICE) for the full attribution.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md). TL;DR: `uv sync`, `uv run pytest`, `cargo test --workspace`, add a test, open a PR.

## License

[MIT](LICENSE).
