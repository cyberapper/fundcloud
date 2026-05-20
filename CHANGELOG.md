# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.7.0](https://github.com/cyberapper/fundcloud/compare/v0.6.0...v0.7.0) (2026-05-20)


### ### Added

* **patterns:** boundary-respect ratio for 2-anchor trendline component ([9c4bd03](https://github.com/cyberapper/fundcloud/commit/9c4bd030af9ee367a24626aee3203d5638544dd8))
* **patterns:** composite gates + H&S head-prominence in quality scorer ([016b7dc](https://github.com/cyberapper/fundcloud/commit/016b7dce7346892f2e0f9c2c95c7b90ff312b1c0))


### ### Fixed

* **patterns:** add triple boundary gates + sync trendline_r2 docs ([1d132a3](https://github.com/cyberapper/fundcloud/commit/1d132a331ba4a364b28fc9b876b27428efd3a5de))
* **patterns:** guard apply_condition against non-positive target/stop ([5c684c7](https://github.com/cyberapper/fundcloud/commit/5c684c74748c505f54f896f2f023c275575aa1f9))
* **patterns:** skip anchor bars in triple-detector boundary-respect gates ([18ff42d](https://github.com/cyberapper/fundcloud/commit/18ff42de9375523b357b007e94afbbc3cf0fab50))
* **patterns:** use anchor R² in trendline component of quality score ([23ac819](https://github.com/cyberapper/fundcloud/commit/23ac819bc27fc30858fd8923afde74134f241ef6))
* **patterns:** validate boundary_tolerance is non-negative in detect.rs ([29a4309](https://github.com/cyberapper/fundcloud/commit/29a43096650b9443fac8bb0e34839f35e184e038))


### ### Changed

* **patterns:** TrendLine.role removes upper/lower ambiguity from boundary scoring ([cb0fb9e](https://github.com/cyberapper/fundcloud/commit/cb0fb9e14f0230d0476f7a392eb0316eecdc4508))


### ### Documentation

* **patterns:** close docs/code gap on composite gates + Triple knob names ([7c8088a](https://github.com/cyberapper/fundcloud/commit/7c8088a4c77e00a6bd32a4c9b1e64063176be32d))
* **patterns:** refresh trendline_r2 / role docs for boundary-respect scorer ([5f866c1](https://github.com/cyberapper/fundcloud/commit/5f866c1f4e4d8d7473b8d94b29c462dc17c0b760))

## [0.6.0](https://github.com/cyberapper/fundcloud/compare/v0.5.0...v0.6.0) (2026-05-14)


### ⚠ BREAKING CHANGES

* **patterns:** The events DataFrame returned by detector.events(bars) no longer carries a `direction` column. Consumers that read it should migrate to:   * apply_condition: pass PatternCondition(direction=Direction.BULLISH|BEARISH)   * feature_quality.evaluate: pass trade_direction="long"|"short"   * PatternStrategy: pass condition=PatternCondition(direction=...)

### ### Fixed

* **accessors:** drop removed `inverse` kwarg from run_pattern ([a28b7a8](https://github.com/cyberapper/fundcloud/commit/a28b7a807e21cb7ea41272cc53b7442fd0771d30))
* **examples:** drop removed `direction` field from pattern examples ([f2a65ec](https://github.com/cyberapper/fundcloud/commit/f2a65ec184fc1c72e913a6596814e9b6b3983e91))


### ### Changed

* **patterns:** drop direction from detection; caller supplies it ([162cbdd](https://github.com/cyberapper/fundcloud/commit/162cbdd8908e61bb1fff9da9fe211e336fa21392))


### ### Documentation

* **patterns:** drop version stamps and legacy framing from comments ([e99332c](https://github.com/cyberapper/fundcloud/commit/e99332c1e08d749840fc0240b4434ca618523bca))

## [0.5.0](https://github.com/cyberapper/fundcloud/compare/v0.4.0...v0.5.0) (2026-05-08)


### ### Added

* add quarterly cadence support and enhance monthly cadence handling ([a879d80](https://github.com/cyberapper/fundcloud/commit/a879d8071605c44011786b6e58bd3641460df0b1))
* enhance GitHub Actions workflow by adding a new job to sync lockfiles ([72b5352](https://github.com/cyberapper/fundcloud/commit/72b5352ebae6c2a740d5c38f7fcf1c93a853f11d))
* introduce quarter cadence for DCA strategy ([8c56541](https://github.com/cyberapper/fundcloud/commit/8c56541d31edfb4ddfedac2848b6ed4ec82e54a2))
* **patterns:** expose per-detector tunable knobs as constructor kwargs ([dc9a2d4](https://github.com/cyberapper/fundcloud/commit/dc9a2d42de803196c0a41a63adc03b88cac06cbe))
* **patterns:** integrate chart-pattern recognition as a fundcloud feature ([2515ce0](https://github.com/cyberapper/fundcloud/commit/2515ce0f8555cc1ad97820cdd06068bb5c0844fc))
* **patterns:** v1.2.0 scorer + tiered pivot scan to surface long-window patterns ([39e5c59](https://github.com/cyberapper/fundcloud/commit/39e5c592123788d3da72214968978bdc82f9cf57))
* **scoring:** v1.1.0 — trendline_r2 now measures intermediate-bar fit ([9c158e1](https://github.com/cyberapper/fundcloud/commit/9c158e14457971e51f2b58351208f3e93f114803))
* **scoring:** version, audit-trail, and calibration scaffold for quality ([4854fc4](https://github.com/cyberapper/fundcloud/commit/4854fc4511ecb75c1381e6191b07a8bab37fb6bf))


### ### Fixed

* **patterns:** address CodeRabbit review on PR [#21](https://github.com/cyberapper/fundcloud/issues/21) ([e2ef7c5](https://github.com/cyberapper/fundcloud/commit/e2ef7c512ac06eec7f3dc24871e96734e2b952e1))
* **patterns:** address remaining CodeRabbit threads on PR [#21](https://github.com/cyberapper/fundcloud/issues/21) ([38a2895](https://github.com/cyberapper/fundcloud/commit/38a2895f24d93f260431a87d3d2647b69d6fa154))
* **patterns:** clear all CI gate failures (cargo fmt, clippy, mypy) ([51b7fdf](https://github.com/cyberapper/fundcloud/commit/51b7fdfd022905580e98d4014a0d326a49035bd4))
* **patterns:** clippy sort_by_key + raise PR diff coverage ([a1fa542](https://github.com/cyberapper/fundcloud/commit/a1fa542bfd0f742d3fa27f660ee42417b9a235d8))
* **patterns:** loosen triangle flat_threshold default 0.0005 → 0.005 ([cba73fc](https://github.com/cyberapper/fundcloud/commit/cba73fc77b71a9a561e36101a5ebbd6cc4b8b078))
* **rust:** complete pyo3 0.28 API migration in patterns bindings ([2e5bdda](https://github.com/cyberapper/fundcloud/commit/2e5bdda4bd1d2594e0cd83fd1378975330fafae3))
* **tests:** update DECAY-mode test to match decay-to-zero contract ([c7fd10a](https://github.com/cyberapper/fundcloud/commit/c7fd10ac851b343f3817765b95d3ecc28caf16f1))


### ### Changed

* **patterns:** single-version library, drop calibration scaffolding ([906c82d](https://github.com/cyberapper/fundcloud/commit/906c82de87d2e9f0822831d5d284a7e2835a78c9))


### ### Documentation

* **patterns:** add end-to-end overview, wire orphan pages, fix mkdocs strict ([c036c74](https://github.com/cyberapper/fundcloud/commit/c036c74f2fd33580f8f52ca89e07ea219a35cc36))
* **patterns:** add visualization section, fix step-4 imports, link examples ([ffea3ab](https://github.com/cyberapper/fundcloud/commit/ffea3abb0e224188e489c164202df6d33673a7c6))
* **patterns:** consolidate overview into pattern-detection guide ([5b42986](https://github.com/cyberapper/fundcloud/commit/5b42986daff1ad7c11dcd28954b205876a939835))
* **patterns:** demote calibration record + add configurable-scan example ([bf1ad45](https://github.com/cyberapper/fundcloud/commit/bf1ad4574783fd0623fb54e02532f872a492bd14))

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
