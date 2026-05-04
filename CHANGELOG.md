# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.4.0](https://github.com/cyberapper/fundcloud/compare/v0.3.0...v0.4.0) (2026-05-04)


### ### Added

* add Codecov configuration and update CI workflow for coverage uploads ([919b663](https://github.com/cyberapper/fundcloud/commit/919b66325de30c9441df6e5707b02cb731a47059))
* **sim,data:** intra-bar bracket orders + ClickHouse backend ([a2064fe](https://github.com/cyberapper/fundcloud/commit/a2064fe414a24edefb3981be923717ad118a3f3f))
* **strategies:** DCA executes long-short via negative weights ([89bf882](https://github.com/cyberapper/fundcloud/commit/89bf882087d9a09ab3b811ef8afb3b3a3b356b8e))
* **strategies:** improve run_hold / run_dca clarity and configurability ([8e63355](https://github.com/cyberapper/fundcloud/commit/8e63355ebab40067e6b91147cfbb4511489dd2cf))
* update dependencies and improve performance ([1ccd9df](https://github.com/cyberapper/fundcloud/commit/1ccd9df449c2631ddbb899678c24749a373e3644))


### ### Fixed

* address CodeRabbit PR [#20](https://github.com/cyberapper/fundcloud/issues/20) review ([151d90a](https://github.com/cyberapper/fundcloud/commit/151d90ab0717bbbd0d09ec40a8935857388c4091))
* align fast paths to fire same-bar fill-bar SL/TP/TSL ([9b502fd](https://github.com/cyberapper/fundcloud/commit/9b502fd8b0c787db55070cc844df3c855bb4dd43))
* **ci,sim:** exclude docker tests on Windows/macOS + PR review fixes ([5a099a2](https://github.com/cyberapper/fundcloud/commit/5a099a25df9288a3dda753eaf286c467868bda74))
* enhance validation in ClickHouse and Order classes ([4b3c995](https://github.com/cyberapper/fundcloud/commit/4b3c995ef64622b52ddc18924ecb20e3f31750b9))
* **strategies:** clip DCA `amount_pct` deposits to available cash ([2a29d2c](https://github.com/cyberapper/fundcloud/commit/2a29d2c54168bea62f0b595ce626848a67621903))
* tighten DCA tests + future-proof trade_reason rehydrate ([32a1885](https://github.com/cyberapper/fundcloud/commit/32a188502c33615bdcf10f3950757b0f1b9f2683))


### ### Changed

* **sim:** no-look-ahead execution invariant + docstring uplift ([0911031](https://github.com/cyberapper/fundcloud/commit/0911031b86fb0ff8a8af83ab02718200fdf6bd9b))


### ### Documentation

* docstring improvement ([0911031](https://github.com/cyberapper/fundcloud/commit/0911031b86fb0ff8a8af83ab02718200fdf6bd9b))

## [0.3.0](https://github.com/cyberapper/fundcloud/compare/v0.2.1...v0.3.0) (2026-04-29)


### ### Added

* **accounts:** add FundCloud platform integration for NAV and market data ([eec2d66](https://github.com/cyberapper/fundcloud/commit/eec2d6698e225c1aa2e24d18b100f493b95d1f9b))
* **accounts:** add Interactive Brokers Flex Query CSV provider ([1952dd6](https://github.com/cyberapper/fundcloud/commit/1952dd6b3469e528390f8288a0858055280a565b))
* **accounts:** FundCloud + Interactive Brokers account providers ([7735fb7](https://github.com/cyberapper/fundcloud/commit/7735fb72bab85b5a5ad2a376d3ae848edfdf9f5f))


### ### Fixed

* **accounts:** fail closed on malformed source data with typed errors ([072c487](https://github.com/cyberapper/fundcloud/commit/072c4878e5088e19c4e7e2a45be7b3f4c8ffdd05))
* address CI failure, PR review round, and pre-commit drift ([48cb057](https://github.com/cyberapper/fundcloud/commit/48cb057a64a18435066eb9015e2a95f7227acfc3))

## [0.2.1](https://github.com/cyberapper/fundcloud/compare/v0.2.0...v0.2.1) (2026-04-23)


### ### Fixed

* **build:** include LICENSE and NOTICE in sdist for PEP 639 ([047a416](https://github.com/cyberapper/fundcloud/commit/047a416011b9ee9a20a171a2d0ca90cb5d67992d))
* **build:** include LICENSE and NOTICE in sdist for PEP 639 ([c5b3dfb](https://github.com/cyberapper/fundcloud/commit/c5b3dfb1175ca1125a1afbf4f9669dc9b676a662))

## [0.2.0](https://github.com/cyberapper/fundcloud/compare/v0.1.0...v0.2.0) (2026-04-23)


### ### Added

* initial public release — fundcloud v0.1.0 ([a610800](https://github.com/cyberapper/fundcloud/commit/a6108004333c2056cf3f23ed9d0d54dc50e2f819))
* initial public release — fundcloud v0.1.0 ([d46cee5](https://github.com/cyberapper/fundcloud/commit/d46cee5d5577cba39f5fe3396489550c696c769c))


### ### Fixed

* address all CodeRabbit review findings + strengthen contributor tooling ([58092ef](https://github.com/cyberapper/fundcloud/commit/58092ef874926a1bf52f9fe9d4816ed0c4890270))
* resolve CI failures — cargo fmt, matplotlib 3.14 legend recursion ([2b8eff8](https://github.com/cyberapper/fundcloud/commit/2b8eff834994b9e635b16c9636a492218872dbc4))
* resolve CI failures — cargo fmt, matplotlib 3.14 legend recursion ([6f04d96](https://github.com/cyberapper/fundcloud/commit/6f04d960da4b38b8a237dd9b5c4968dbdb393a38))
* resolve CI failures — format, DuckDB in-operator, mypy overrides ([3ece4c4](https://github.com/cyberapper/fundcloud/commit/3ece4c45342590d2d5fee36658d1123cb1d25175))
* resolve CI lint, Python 3.14 recursion, and Windows encoding failures ([4a63943](https://github.com/cyberapper/fundcloud/commit/4a63943d02e61470477842f41f2f042894395af1))
* **rust:** resolve all cargo clippy warnings for CI ([32cedbb](https://github.com/cyberapper/fundcloud/commit/32cedbb032ec2167dcba24ae3c08296a8e96262e))

## [Unreleased]

## [0.1.0] - 2026-04-22

### Changed

- **Cumulative-return chart switched from wealth index to percent.** `fundcloud.plots.cumulative` (and the matplotlib mirror) now compute `(1 + r).cumprod() - 1` and apply a `.0%` tick format, so the y-axis reads `0%, 50%, 100%` rather than `1.0, 1.5, 2.0`. Matches the existing convention used by `drawdown`, `return_distribution`, and `monthly_heatmap`. Excel's embedded cumulative chart and PDF matplotlib panel pick up the same format.
- **`DCA` now defaults to equal weights** when `amount` is scalar and `weights` is omitted. `DCA(500, horizon="weekly")` (and `bars.fc.run_dca(500, horizon="weekly")`) now splits the deposit evenly across the assets in the bars frame, matching `Hold`'s behaviour for the analogous case. Explicit `weights=` still wins, and still has to sum to 1.
- **`SimResult.summary()` vs `SimResult.metrics()`** are now distinct: `summary()` returns the compact 11-metric view (`Portfolio.summary`) and `metrics()` returns the full ~55-metric bundle (`Portfolio.metrics`). They were mutually aliased before and both returned the compact view.
- **`fundcloud.explore.profile`** now returns a :class:`ProfileReport` object with `.stats`, `.correlations`, `.missing`, `.alerts`, `.to_html()`, `.to_dict()`, `__repr__`, and Jupyter `_repr_html_`. Passing `output=` still writes the HTML file as before — the report is returned on top.
- **`fundcloud.explore.describe`** supersedes `quickview` as a super-set of :meth:`pandas.DataFrame.describe` (count / mean / std / min / 25% / 50% / 75% / max + dtype, missing, unique, skew, kurtosis, zeros_pct, inf_pct + optional Sharpe / CAGR / volatility / max_drawdown when the index is a DatetimeIndex). `quickview` stays as a deprecated alias for one release.
- **`Tearsheet.render_pdf`** now defaults to a pure-Python matplotlib `PdfPages` engine (no Pango required); `engine="weasyprint"` keeps the CSS-styled variant for users with the system libraries.

### Added

- **Period-return table** (`fundcloud.metrics.period_returns`) — MTD / 3M / 6M / YTD / 1Y / 3Y (ann.) / 5Y (ann.) / 10Y (ann.) / All-time (ann.). Accepts a `benchmark=` kwarg for a two-column benchmark/strategy output, returns a `DataFrame` when the input is a panel. Exposed on `Portfolio.period_returns(...)` and both `.fc` accessors.
- **Run-up (rally) episode table** (`fundcloud.metrics.runup_details`) — symmetric mirror of `drawdown_details`, one row per trough→peak→retreat episode between drawdowns. Exposed on `Portfolio.runup_details()`.
- **`Portfolio.worst_drawdowns(top=10)` / `Portfolio.worst_runups(top=10)`** — display-formatted top-N episode tables with columns `Started / Recovered / Drawdown / Days` and `Started / Peaked / Runup / Days`, ready to slot into a report.
- **`Portfolio.yearly_returns(benchmark=...)`** — calendar-year returns; returns a two-column `DataFrame` when a benchmark is supplied (or when one was set at construction time), a `Series` otherwise. The same shape is available on both `.fc` accessors.
- **`fundcloud.plots.yearly_returns_bars`** — paired grouped-bar chart (benchmark amber, strategy blue, dashed mean-return reference), percent-formatted y-axis, `barmode="group"`. Wired through `.fc.plot_yearly_returns(...)`.
- **Tear sheet now renders four new sections** (HTML + matplotlib PDF + Excel): *Period performance* table, *EOY returns* paired bar chart plus matching table, *Worst 10 drawdowns* table, and *Top 10 runups* table. Excel gains four new sheets — `Period Returns`, `Yearly Returns`, `Drawdowns`, `Runups`, all percent-formatted.
- **`SimResult.pf`** — shortcut property for `SimResult.portfolio`, so `result.pf.sharpe()` is one less token to type.
- **`.fc` DataFrame accessor expanded** with six new sub-surfaces:
  - Report renderers — `render_html`, `render_pdf`, `render_excel`.
  - EDA — `describe`, `profile`, `compare`.
  - Simulator — `run_strategy`, `run_weights`, `run_signals`, `run_orders`, plus preset wrappers `run_hold(weights, ...)` and `run_dca(amount, horizon, weights, ...)`, plus a type-dispatching `simulate(what)`.
  - Plots — `plot_cumulative`, `plot_drawdown`, `plot_rolling_sharpe`, `plot_return_distribution`, `plot_monthly_heatmap`, `plot_composition`.
- **`.fc` Series accessor** gains matching `render_html` / `render_pdf` / `render_excel`, `describe`, `profile`, and the plot builders where they apply.
- **mkdocs guide** — `docs/guides/using-python-help.md` explaining the `help(...)` introspection pattern.

### Removed / cleaned

- Residual "slice-N" and `docs/progress/…` references purged from shipped Python docstrings and user-facing docs. `docs/progress/**` remains as an internal engineering log, excluded from the published site.
- Duplicate `docs/overrides/` directory deleted; mkdocs uses the repo-root `./overrides/`.

### Performance

- **Rust simulator kernel** for the three deterministic `Simulator` entry points (`run_weights`, `run_orders`, `run_signals`). The kernel lives in `crates/fundcloud-core/src/sim.rs`, is PyO3-wrapped with GIL released under `py.allow_threads`, and dispatched automatically when all three of `(costs, slippage, execution)` are built-ins (`NoCost` / `FixedBps` / `PerShare`, `NoSlippage` / `HalfSpread`, `NextBarOpen` / `SameBarClose`). Custom models transparently fall back to the pure-Python path.
- New `python/fundcloud/kernels/_sim_pyfallback.py` — a NumPy-panel loop that serves both as the pure-Python production fallback and the parity reference for the Rust kernel (the dispatcher never duplicates logic).
- Benchmark (`bench/bench_sim.py`, `run_weights`): 500×5 → 2.0×, 2000×10 → 3.4×, 5000×20 → 5.3×, 10000×30 → **6.2×** wall-time speedup vs the NumPy fallback.
- Parity tests (`tests/unit/test_sim_parity.py`, 37 new tests): Rust ↔ NumPy-fallback agree to `atol=1e-10` across every `(cost × slippage × execution)` combination for all three entry points.
- New docs guide `docs/guides/sim/performance.md` explains when the fast path engages and how to verify it.

### Added

**Data layer**
- Unified `fundcloud.data.Backend` protocol — every backend is readable, writes are gated by a `read_only` constructor flag and raise `ReadOnlyError` when locked.
- `fundcloud.data.bars` — OHLCV conversions, alignment, resampling, long/wide pivots.
- Read-only network backends (behind provider extras): `YF`, `FMP`, `AV`, `Binance` — shared HTTP client with `tenacity` retry.
- Read-write format backends (core): `Parquet` (per-key parquet files), `DuckDB` (per-key tables), `Memory` (in-process dict).
- Read-only local backend (core): `CSV` (single file or per-symbol directory).
- Explicit `WriteMode` literal — `'overwrite'`, `'upsert'` (concat + dedup on index, default for sync), `'append'` (raw), `'error'`.
- `Backend.sync_to(sink, key=..., mode='upsert')` shortcut for one-off source → sink transfers, idempotent under overlap.
- Canonical OHLCV column naming across every network backend — `open`, `high`, `low`, `close`, `volume` (lowercase snake_case, in canonical order). Helpers `OHLCV_COLUMNS`, `normalize_field`, `normalize_ohlcv_columns`, `canonicalize_ohlcv_order` are exposed from `fundcloud.data` for users who need to apply the same normalisation to their own frames.
- Adjusted equity prices by default on `YF` / `FMP` / `AV` — the canonical `close` column carries dividend/split-adjusted values. Pass `adjust=False` to get raw, as-traded prices. Fixes a latent bug where `AV` hit the `..._ADJUSTED` endpoint but parsed the raw `4. close` field, and where `FMP` ignored the `adjClose` field returned by its daily endpoint.
- 1-year default window on every network backend's `read()` — a bare `YF("SPY").read()` pulls one year, not twenty. Pass `start=` to override. Cache backends (`Parquet`/`DuckDB`/`Memory`/`CSV`) are not affected.
- `Catalog` — named datasets, incremental refresh via `sync_to(mode='upsert')`, `refresh_kwargs` with documented `start`/`end`/`lookback` contract (the `lookback` window re-pulls recently-corrected rows on refresh), describe() summary.

**Features**
- `IndicatorSpec` base class and 158 auto-generated TA-Lib wrappers across 10 groups.
- `FeaturePipeline` — sklearn `FeatureUnion`-style with a stable `pipeline_hash`.
- `FeatureStore` — cache keyed by `(dataset, pipeline_hash)`.
- `@register_indicator` extension hook for custom transformers.

**Portfolio + optimisation**
- Unified `Portfolio` (live state via `apply`/`mark_to_market`/`snapshot` plus analytics: `sharpe`, `sortino`, `calmar`, `omega`, `max_drawdown`, `cvar`, `value_at_risk`, `ulcer_index`, `turnover`, `attribution`, `contribution`, `summary`), with `from_skfolio` / `to_skfolio` round-trip.
- `Population` — comparison container for multiple portfolios.
- Pure-Python fallback optimisers (`EqualWeighted`, `InverseVolatility`, `MVO`) always available.
- `fundcloud.optimize` adapters — `MeanRisk`, `RiskBudgeting`, `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution`, `MaximumDiversification`, `NestedClustersOptimization` via the `[pf]` extra.
- `fundcloud.metrics.batch` — panel-wide comparisons (`batch_sharpe`, `batch_sortino`, `batch_max_drawdown`, `batch_cvar`, `batch_summary`).

**Strategies + simulator**
- `BaseStrategy`, `Context`, `Scheduler`, `Cadence` (daily / weekly / monthly).
- Presets: `Hold` (with `RebalanceSpec`), `DCA`.
- `@register_strategy` extension hook.
- `Simulator` with four entry points: `run_strategy`, `run_weights`, `run_signals`, `run_orders`.
- Cost models (`NoCost`, `FixedBps`, `PerShare`), slippage (`NoSlippage`, `HalfSpread`), execution (`NextBarOpen`, `SameBarClose`).

**Validation**
- Own `PurgedKFold` and `EmbargoedKFold` (sklearn `BaseCrossValidator`, always available).
- Lazy re-exports of skfolio's `CombinatorialPurgedCV` and `WalkForward`.

**Reports + exploration**
- `Tearsheet` — `render_html` (self-contained plotly), `render_pdf` (matplotlib default, WeasyPrint opt-in), `render_excel` (XlsxWriter with native charts).
- Native EDA: `explore.quickview`, `explore.profile`, `explore.compare` — plotly + jinja2 + scipy, shipped in core. Replaces external `ydata-profiling` / `sweetviz` wrappers.
- Plotly and matplotlib plot builders: `cumulative`, `drawdown`, `rolling_sharpe`, `monthly_heatmap`, `return_distribution`, `composition`.
- **Multi-asset builders.** Every Series-accepting builder now also accepts a `pandas.DataFrame` (one overlayed trace per column). `monthly_heatmap` remains single-series; it squeezes a one-column DataFrame and raises on multi-column input.
- **On-figure annotations** via `annotations=True` — stats pills drawn from `fundcloud.metrics.core` (CAGR/Vol/Sharpe on cumulative, Max DD + date range on drawdown, VaR/CVaR on distribution, annual totals on heatmap, turnover subtitle on composition) and a full-period Sharpe reference line on rolling Sharpe.
- **Plotly themes.** `plots.set_theme("white"|"dark"|"ggplot2"|"seaborn"|"default")` maps onto Plotly's builtin templates; any name registered in `plotly.io.templates` passes through transparently. Each builder also takes a `theme=` kwarg for per-figure overrides.
- **Headless aggregation.** `plots.summary(returns, benchmark=..., weights=..., theme=..., title=...)` returns one composed Plotly figure (cumulative + drawdown + rolling Sharpe + distribution + monthly heatmap + optional composition row). `fundcloud.plots.mpl.summary` does the same via `matplotlib.gridspec`.
- **Title alignment.** Plotly and matplotlib builders now share defaults — `"Drawdown (%)"`, `"Return distribution (%)"` — so swapping backends doesn't reshuffle labels.

**Speed**
- Rust kernel suite in `crates/fundcloud-core` — rolling mean/std, drawdown, Sharpe, Sortino, VaR, CVaR; Rayon-parallelised, GIL-released via `py.allow_threads`.
- `fundcloud.kernels` shim transparently picks the Rust backend when available; pure-Python references in `_pyfallback.py` agree to 1e-10 on random panels (parity tests).
- `HAS_RUST` flag exposes the active backend.

**Accessors**
- `.fc` pandas accessor on `Series` and `DataFrame` — one-liner access to every metric in `fundcloud.metrics`.

**Release**
- Tag-triggered workflow in `.github/workflows/release.yml` — staged TestPyPI → smoke-test → PyPI, both gated by GitHub environments with PyPI trusted publishing (OIDC).

[Unreleased]: https://github.com/cyberapper/fundcloud/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cyberapper/fundcloud/releases/tag/v0.1.0
