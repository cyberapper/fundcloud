# Fundcloud examples — trader scenarios

Every script in this folder is a self-contained answer to a question
a real investor or quant might ask. Run them directly with `uv` — no
notebook server needed:

```bash
uv run python examples/01_dca_weekly_spy.py
```

Each script writes its HTML tear sheet to `examples/out/` (gitignored)
and prints the quantitative summary to stdout.

## Synthetic-data scenarios (offline, reproducible)

| # | Scenario | Audience | Extras needed |
|---|---|---|---|
| 01 | [Weekly DCA into SPY](./01_dca_weekly_spy.py) | Retail | core |
| 02 | [DCA vs lump-sum](./02_dca_vs_lump_sum.py) | Retail | core |
| 03 | [60/40 with quarterly rebalance](./03_sixty_forty_rebalance.py) | Retail | core |
| 04 | [HRP vs Equal-Weight vs MVO](./04_hrp_vs_equal_weight.py) | Quant | `fundcloud[pf]` (HRP + MeanRisk) |
| 05 | [Golden-cross SMA crossover](./05_golden_cross_momentum.py) | Quant | core (uses Rust kernels) |
| 06 | [Walk-forward Mean-Risk (OOS)](./06_walk_forward_optimization.py) | Quant | `fundcloud[pf]` |
| 10 | [`.fc` pandas accessor quickstart](./10_pandas_accessor_quickstart.py) | Anyone with a Series / DataFrame | core |

### skfolio-inspired optimisation scenarios

Ported from the excellent [skfolio example gallery](https://skfolio.org/auto_examples/index.html).
Every script runs against synthetic data so results are reproducible; swap
in real returns once you've seen the mechanics.

| # | Scenario | What it teaches | Extras needed |
|---|---|---|---|
| 11 | [Investment horizon](./11_investment_horizon.py) | How daily/weekly/monthly resampling reshapes Sharpe, vol, and MVO weights | core |
| 12 | [Efficient frontier + max-Sharpe](./12_efficient_frontier.py) | Trace the risk-return curve with `efficient_frontier_size`, flag the tangency point | `fundcloud[pf]` |
| 13 | [Min-CVaR vs Min-Variance](./13_min_cvar_vs_min_variance.py) | Tail-risk vs symmetric-risk optimisation under injected crisis days | `fundcloud[pf]` |
| 14 | [Transaction costs and rebalancing](./14_transaction_costs.py) | `transaction_costs` + `previous_weights` reduce turnover under a regime shift | `fundcloud[pf]` |
| 15 | [Black-Litterman views](./15_black_litterman.py) | Blend market-implied prior with explicit investor views via `BlackLitterman` prior | `fundcloud[pf]` |

### Feature engineering

| # | Scenario | What it teaches | Extras needed |
|---|---|---|---|
| 16 | [TA-Lib feature matrix](./16_talib_feature_matrix.py) | Explore the 158 auto-wrapped TA-Lib indicators, build a curated `FeaturePipeline`, and bulk-compute the full feature matrix for ML pipelines | `fundcloud[ta,data-yf]` |
| 20 | [Signals & orders + custom indicator](./20_signals_and_orders.py) | `Simulator.run_signals` / `run_orders` + `register_indicator` for a rolling-z-score mean-reversion signal | `fundcloud[data-yf]` |

### Plumbing (data layer, reports, cross-validation, attribution)

| # | Scenario | What it teaches | Extras needed |
|---|---|---|---|
| 17 | [Local data pipeline + Catalog + FeatureStore](./17_local_data_pipeline.py) | `CSV`, `Parquet`, `DuckDB`, `Memory` backends; `Backend.sync_to(mode='upsert')`; `Catalog` + `DatasetSpec` with `refresh_kwargs` lookback; `FeatureStore`; and the `bars` helpers (`to_prices`, `to_returns`, `resample`, `align`, `as_long`, `as_wide`) | core (+ `[ta]` for the feature cache round-trip) |
| 18 | [Stakeholder report pack — HTML + PDF + Excel](./18_report_pack_pdf_excel.py) | `Tearsheet.render_html` + `render_pdf` (WeasyPrint) + `render_excel` (XlsxWriter) from one `Portfolio` | `fundcloud[reports,viz,data-yf]` |
| 19 | [Advanced optimiser menu](./19_advanced_optimisers.py) | `RiskBudgeting`, `HierarchicalEqualRiskContribution`, `NestedClustersOptimization`, `MaximumDiversification` scored against HRP / min-variance | `fundcloud[pf,data-yf]` |
| 21 | [Cost, slippage, execution lab](./21_cost_model_lab.py) | `NoCost` / `FixedBps` / `PerShare`, `NoSlippage` / `HalfSpread`, `NextBarOpen` / `SameBarClose` — same DCA, seven friction configs | core |
| 22 | [Native EDA — profile, compare, quickview](./22_eda_native.py) | The native `fundcloud.explore` reports (plotly + jinja2 + scipy, no external EDA deps): per-column stats, histograms, correlation, missing patterns, KS + Wasserstein drift, target-correlation shift | `fundcloud[data-yf]` |
| 23 | [Cross-validation zoo](./23_cv_zoo.py) | `PurgedKFold` / `EmbargoedKFold` / `WalkForward` / `CombinatorialPurgedCV` compared side-by-side with an OOS baseline | `fundcloud[pf,data-yf]` |
| 24 | [Attribution + skfolio round-trip + direct kernels](./24_attribution_and_kernels.py) | `Portfolio.attribution` / `contribution` / `turnover`, `from_skfolio` / `to_skfolio`, direct `omega` / `ulcer_index` / `value_at_risk`, and a 200-weighting `kernels.*_batch` sweep in ~1 ms | `fundcloud[pf,data-yf]` |

## Live-data ("battlefield") scenarios

These hit real APIs. They need either a key or internet access (or both).

| # | Scenario | Data source(s) | Prereqs |
|---|---|---|---|
| 07 | [Live SPY DCA](./07_live_spy_dca.py) | yfinance | `fundcloud[data-yf]` |
| 08 | [Cross-asset portfolio (equities + bonds + crypto)](./08_live_cross_asset.py) | YF + FMP + Binance | `fundcloud[pf,data-yf,data-fmp,data-bn]` + `FMP_API_KEY` |
| 09 | [Market snapshot — watchlist metrics](./09_live_market_snapshot.py) | Alpha Vantage | `fundcloud[data-av]` + `ALPHAVANTAGE_API_KEY` (or `ALPHA_VANTAGE_API_KEY`) |

Each live example graceful-fails with a clear message when an extra or
API key is missing — no scary tracebacks.

## Data

All examples synthesise correlated-GBM OHLCV panels via
[`_synth.py`](./_synth.py) so they run offline and are reproducible.

To pull real data instead, swap the three-line `generate_ohlcv(...)`
block for:

```python
from fundcloud.data import YF
bars = YF(["SPY"]).read(start="2020-01-01")
```

(`uv add 'fundcloud[data-yf]'` first.)

## Running everything

```bash
for script in examples/0*.py; do
    echo "=== $script ==="
    uv run python "$script" || exit $?
done
```

Exit code 0 means every scenario ran clean and produced a tear sheet.

## What you'll learn from each

- **01** — the basic simulator + DCA preset + HTML report pipeline.
- **02** — `Population` for comparing two strategies side-by-side.
- **03** — `Hold(rebalance=RebalanceSpec(...))` and turnover accounting.
- **04** — the `fundcloud.optimize` adapters; fallback when `[pf]` is absent.
- **05** — subclassing `BaseStrategy` + using the Rust `kernels.rolling_mean`
  inside a strategy.
- **06** — `PurgedKFold` as a sklearn-native splitter + stitching OOS
  returns back into a `Portfolio`.
- **07** — swap `generate_ohlcv(...)` for `YF(...).read(...)` — the rest
  of the pipeline is unchanged.
- **08** — combine frames from three different providers, align on the
  intersection of trading days, and compare four optimisers.
- **09** — rate-limit awareness on the Alpha Vantage free tier (13-sec
  stagger between calls), plus `metrics.batch_summary` as a one-shot
  watchlist view.
- **10** — the "pandas door" into Fundcloud: `series.fc.sharpe()`,
  `df.fc.summary()`, `prices.fc.to_returns()`. No simulator, no strategy —
  just one-liners on whatever returns you already have in memory.
- **11** — horizon-aware metrics + the `fundcloud._config.config(...)`
  context manager for scoping the annualisation factor locally.
- **12** — `efficient_frontier_size` traces 15 frontier portfolios in one
  fit call; we highlight the tangency (max-Sharpe) point.
- **13** — swap `RiskMeasure.VARIANCE` for `RiskMeasure.CVAR` to move from
  symmetric to tail-risk optimisation; inject a few "crisis days" to see
  the weight shift.
- **14** — realistic rebalance flow: today's `previous_weights` + 50 bps
  `transaction_costs` pulls the fresh fit back toward yesterday's book.
- **15** — Black-Litterman via `skfolio.prior.BlackLitterman`. Pass views
  like `"US_EQ - BONDS_AGG == 0.05"` into the prior estimator; the rest
  of the Fundcloud pipeline is unchanged.
- **16** — Fundcloud auto-wraps every TA-Lib function (158 across 10
  groups) as a sklearn-compatible `IndicatorSpec`. Section 2 composes six
  classics into a `FeaturePipeline`; section 3 runs the full catalogue in
  one pass to produce a 500 x 348 feature matrix ready for XGBoost /
  feature-importance analysis, saved as parquet.
- **17** — full local data layer end-to-end: broker-style CSVs → `CSV` →
  `Parquet` and `DuckDB` round-trip → `Backend.sync_to(mode='upsert')` for
  one-off transfers → `Catalog` with named datasets, incremental refresh
  with a `lookback` window for late-arriving corrections → the `bars`
  helpers (`to_prices`, `to_returns`, `to_log_returns`, `resample`,
  `align`, `as_long` / `as_wide`) → `FeatureStore` caching by
  `pipeline_hash`.
- **18** — one `Portfolio`, three renderers: `Tearsheet.render_html`
  (self-contained plotly), `render_pdf` (WeasyPrint), `render_excel`
  (XlsxWriter with native charts).
- **19** — the rest of `fundcloud.optimize`: `RiskBudgeting`, HERC,
  `NestedClustersOptimization`, `MaximumDiversification` scored against
  HRP / min-variance on a real six-asset universe; includes a
  diversification-ratio column that ranks each engine's de-correlation
  impact.
- **20** — the two simulator doors not covered elsewhere:
  `Simulator.run_signals` (boolean entries/exits matrix) and `run_orders`
  (explicit ts/asset/side/qty log), plus `register_indicator` wiring a
  custom rolling-z-score into the TA-Lib catalogue.
- **21** — friction lab: seven (cost × slippage × execution) configs over
  the same weekly DCA with a drag-in-bps column; shows why
  `NextBarOpen` + `FixedBps` is the realistic default.
- **22** — pre-model EDA without external deps: `quickview` (stdout
  table), `profile` (full HTML report with alerts), `compare` (KS +
  Wasserstein drift between train and holdout, optional target-correlation
  shift table).
- **23** — time-series CV gallery: `PurgedKFold` vs `EmbargoedKFold` vs
  `WalkForward` vs `CombinatorialPurgedCV`, with an OOS baseline that
  shows how CPCV's many-fold distribution shrinks the fold-to-fold spread.
- **24** — `Portfolio.attribution` / `contribution` / `turnover`,
  `from_skfolio` / `to_skfolio`, direct-call metrics `omega` /
  `ulcer_index` / `value_at_risk`, and a 200-weighting Rust-kernel sweep
  (`sharpe_batch` / `cvar_batch` / `max_drawdown_batch`) that completes in
  ~1 ms on a 5-year panel.

## Related: live integration tests

The `tests/integration/test_live_sources.py` suite exercises each provider
end-to-end and is marked `@pytest.mark.network`:

```bash
uv run pytest tests/integration -m network -q
```

Those tests and the examples above are the two complementary guarantees
that the data layer holds up against real providers.
