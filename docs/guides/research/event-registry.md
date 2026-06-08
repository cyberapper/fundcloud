# Event registry

The **single source of truth** for the event-study research engine: every discretionary
price-action concept (ICT/SMC, VCP, support/resistance, classical patterns) decomposed
into an **objective, causal, leak-free OHLCV event**. This page records each event's `id`,
name, exact formation logic, parameters, and — most importantly — its **`confirmed_ts`
rule** (when the event becomes knowable without using future bars).

This is a *catalog of definitions*, not a set of trading rules. An event is a timestamped
observation; whether it has edge is decided empirically by the event-study stack (forward
returns, MFE/MAE, edge ratio, regime/baseline-conditioned expectancy), not by doctrine.

Scope today: **US equities, daily (`1D`) bars** from the `md_ohlcv_data` feed
(`fundcloud.research.load_bars`). Crypto and intraday are deferred — the source has no deep
history for them.

## Detector contract (causal rules every event must obey)

Every event carries **four timestamps** that separate *known* from *allowed to act* from
*filled* — the discipline that keeps the research leak-free:

| timestamp | meaning |
|---|---|
| `formation_end_ts` | last bar the pattern logic reads |
| `observable_ts` (≡ `confirmed_ts`) | first bar at which **all** inputs are knowable — the leak-free anchor; path metrics are measured strictly *after* this |
| `decision_ts` | first bar a strategy may act, `= observable_ts + 1` |
| `execution_ts` | assumed fill bar, `= decision_ts` (next-bar open by default) |

1. **Causal detection.** An event at bar `t` may use information from bars `≤ t` only.
2. **Pivot neighbour-lock (the load-bearing rule).** A pivot of order `k` is
   `pivot_high(p, k) = (high[p] == max(high[p-k … p+k]))` — it uses `k` *future* bars, so it
   is only confirmable at `p + k`. Any event whose logic rests on a pivot/level/swing must set
   `observable_ts ≥ ts[pivot_index + k]`. Bar-local events (FVG, displacement) confirm at `t`;
   every pivot-derived family (S/R, sweeps, VCP swings, trendlines, cup rims) neighbour-locks.
3. **Prefix-invariance test (mandatory — the proof, not the promise).** A spec does not close a
   leak; a test does. For every detector, running it on `bars[0:T]` must yield the events with
   `observable_ts ≤ ts[T−1]` **identical** to running it on the full series. If a detector peeks
   forward, the truncated run differs and the test fails. This is the gate every detector must
   pass — it mechanically guarantees no future bar leaked, far stronger than eyeballing the logic.
4. **Measure the path from the execution price.** MFE / MAE / returns are computed against
   `open[execution_ts]` (the assumed fill), **never** the signal-bar close — otherwise you book a
   fill you could not have gotten.
5. **Sides are separate records.** Bullish and bearish variants are distinct events even when
   they share geometry.
6. **Cost.** 6 bps round-trip (3 in + 3 out) applied to any trade-like return.
7. **Versioning.** Every event carries a `logic_version` (bump on any change to formation or the
   timestamp rules) and a `params_hash` so re-definitions never pool incomparable samples.

Per-event specs below say `confirmed_ts = t` as shorthand for `observable_ts = ts[t]`.

## Status legend

| status | meaning |
|---|---|
| `implemented` | detector exists and emits the observation schema |
| `proposed` | logic specified here; detector not yet built |

## Catalog — summary

| event_id | name | family | direction | status |
|---|---|---|---|---|
| `ev_disp_bar` | Displacement bar | price_action | both | proposed |
| `ev_gap_imb_3c` | Fair Value Gap (3-candle imbalance) | price_action | both | proposed |
| `ev_sweep_fail` | Liquidity sweep / failed breakout | price_action | both | proposed |
| `ev_ob_impulse_last_opp` | Order block (last opposing candle) | price_action | both | proposed |
| `ev_vcp_contract` | Volatility contraction | volatility | long | proposed |
| `ev_pivot_break` | Pivot/base breakout | volatility | long | proposed |
| `ev_sr_touch_bounce` | Support/resistance touch + reject | structure | both | proposed |
| `ev_sr_break_retest` | Break and retest | structure | both | proposed |
| `ev_retrace_ratio` | Fibonacci/retracement-to-zone | structure | both | proposed |
| `ev_cup_u` | Cup (rounded base) | classical | long | proposed |
| `ev_cup_handle_break` | Cup and handle breakout | classical | long | proposed |
| `ev_tline_compress` | Trendline / triangle compression | structure | both | proposed |
| `ev_acc_range` / `ev_spring` / `ev_sos` | Accumulation / spring / sign-of-strength | structure | long | proposed |
| `head_and_shoulders`, `inverse_head_and_shoulders`, `double_top`, `double_bottom`, `triple_top`, `triple_bottom`, `ascending_triangle`, `descending_triangle`, `symmetrical_triangle` | Classical chart patterns | classical | both | **implemented** (`features/patterns/`) |

The 9 classical patterns already ship as Rust detectors (`fundcloud.features.patterns`); they
are listed here so this page is the one registry. Their v1 `breakout_ts = formation_end`
(`features/patterns/_events.py:85`) is **not** strictly neighbour-locked for pivot patterns —
when consumed by the event engine they must be re-anchored to `formation_end + max_pivot_order`.

## Catalog — full specifications

### Immediate targets (full rigor)

#### `ev_disp_bar` — Displacement bar
- **family / direction:** price_action / both (bullish and bearish are separate records).
- **formation_logic (bullish):** at bar `t`, `side·(close[t] − open[t]) > z_body · ATR_n(t−1)`,
  close-location `CLV = (close[t] − low[t]) / (high[t] − low[t]) ≥ clv_min`, and optionally
  `volume[t] / mean(volume, t−m … t−1) ≥ z_vol`. Bearish mirrors with `side = −1` and
  `CLV ≤ 1 − clv_min`.
- **parameters:** `atr_n ∈ {10,14,20}` (default 14); `z_body ∈ {0.8,1.0,1.25,1.5,2.0}` (1.0);
  `clv_min ∈ {0.6,0.7,0.8}` (0.7); `z_vol ∈ {none,1.0,1.25,1.5,2.0}` (none).
- **confirmed_ts:** `= t` (all inputs are bars `≤ t`; no pivots → no neighbour-lock).

#### `ev_gap_imb_3c` — Fair Value Gap (three-candle imbalance)
- **family / direction:** price_action / both.
- **formation_logic (bullish):** at bar `t`, `low[t] > high[t−2]` (strict gap), middle-bar body
  fraction `|close[t−1] − open[t−1]| / (high[t−1] − low[t−1]) ≥ body_min`, and impulse
  `(high[t−1] − low[t−1]) / ATR_n(t−1) ≥ z_imp`. Gap zone = `[high[t−2], low[t]]`,
  `gap_size = low[t] − high[t−2]`. Bearish: `high[t] < low[t−2]`, zone `[high[t], low[t−2]]`.
- **parameters:** `body_min ∈ {0.5,0.6,0.7,0.8}` (0.6); `z_imp ∈ {0.0,1.0,1.25,1.5}` (1.0);
  `atr_n ∈ {10,14,20}` (14).
- **confirmed_ts:** `= t` — all three bars `t−2, t−1, t` are closed at `t`; **zero future bars**,
  no neighbour-lock needed.
- **fill variant:** a "fill-then-go" entry waits for a later bar to trade back into the gap zone;
  that fill bar becomes `confirmed_ts` (still forward-only). Recorded as a separate `params` variant.

#### `ev_sweep_fail` — Liquidity sweep / failed breakout
- **family / direction:** price_action / both.
- **formation_logic (bullish reversal):** with a **neighbour-locked** support level `L`
  (clustered prior pivot-lows), at bar `t`: `low[t] < L − eps · ATR_n(t−1)` **and**
  `close[t] > L`. Bearish: `high[t] > R + eps · ATR_n(t−1)` and `close[t] < R`. Optional
  stronger variant requires elevated range or volume on `t`.
- **parameters:** `pivot_k ∈ {2,3,5,10}` (3); `eps ∈ {0.0,0.05,0.10,0.25,0.50}` ATR (0.10);
  `cluster_tol` for level grouping; `atr_n ∈ {10,14,20}` (14).
- **confirmed_ts:** `= t`, **provided** every pivot forming `L` satisfies `pivot_index + pivot_k ≤ t`
  (neighbour-lock). Levels not yet locked at `t` are ineligible.

### Proposed (logic + `confirmed_ts` specified; detector deferred)

- **`ev_ob_impulse_last_opp` (Order block).** Bullish: the most recent bearish candle `j` before a
  bullish displacement that, within `m` bars, closes above `max(high[j … j−r])`. Zone = body
  `[min(open[j],close[j]), max(open[j],close[j])]` (wick variant available). **confirmed_ts =**
  the impulse-close bar (not candle `j`) — causal, no future bars beyond the impulse.
- **`ev_vcp_contract` (Volatility contraction).** Over window `W`, ≥ `m_pull` successive swing
  pullbacks of decreasing amplitude `a_1 > a_2 > … > a_m`, declining `ATR/price`, flat/down volume.
  Swings are pivot-derived → **neighbour-lock**: confirmed_ts = last contraction's locked pivot
  (`pivot_index + k`). `m_pull ∈ {2,3,4}`.
- **`ev_pivot_break` (Base/pivot breakout).** `pivot = max(high over base)`; at `t`,
  `close[t] > pivot + buf · ATR`, `volume[t] ≥ z_vol · median(volume, base)`, close in top `q` of
  day range. confirmed_ts = `t`, with the base's defining pivots neighbour-locked.
- **`ev_sr_touch_bounce`.** Build horizontal levels from clustered pivots; bullish at `t` if
  `low[t] ≤ level + eps·ATR` and `close[t] > level`. confirmed_ts = `t`, levels neighbour-locked.
- **`ev_sr_break_retest`.** Break at `t0` (`close[t0] > level + buf·ATR`), then within `r` bars a
  retest `low[t] ≤ level + eps·ATR` with `close[t] > level`. confirmed_ts = the retest bar `t`.
- **`ev_retrace_ratio` (Fib / retracement-to-zone).** After an impulse `s0 → s1`,
  `r = (s1 − low[t]) / (s1 − s0)`; fires when `r` enters a band around a grid value `g` and `t`
  closes back in the impulse direction. **Test the full grid**
  `g ∈ {0.236,0.382,0.5,0.618,0.705,0.786}` (no 0.618 worship). Swings pivot-derived →
  neighbour-lock; confirmed_ts = `t` with `s0,s1` locked.
- **`ev_cup_u` / `ev_cup_handle_break`.** Rounded base (left rim `i`, trough `j`, right rim `k`,
  rims within `rim_tol`, depth in `[d_min,d_max]`, U-curvature); handle = shallow upper-half
  pullback; breakout `close[t] > handle_high + buf·ATR`. Rims are pivots → neighbour-lock;
  the cup event's confirmed_ts ≥ right-rim `pivot_index + k`; the breakout event's = `t`.
- **`ev_tline_compress`.** Robust lines fit to recent pivot highs/lows; channel width shrinks,
  `w_end/w_start ≤ rho_max`; breakout when `close[t]` exits a fitted boundary by `buf·ATR`. Lines
  pivot-derived → neighbour-lock; breakout confirmed_ts = `t`.
- **`ev_acc_range` / `ev_spring` / `ev_sos`.** Decomposed Wyckoff primitives: range after a
  downtrend (`width ≤ w_max`); spring = low pierces range support by `eps·ATR` then closes back
  inside; SOS = close exits range high by `buf·ATR`. confirmed_ts = the qualifying bar `t`, range
  bounds neighbour-locked.

## Pre-registered parameter grids

Grids are fixed **before** mining and every trial is logged — this is what makes later
multiple-testing correction (FDR, Deflated Sharpe, PBO) meaningful.

| family | grid |
|---|---|
| ATR length | {10, 14, 20} |
| Pivot order `k` | {2, 3, 5, 10} |
| Formation lookback `W` | {10, 20, 40, 60, 120, 252} |
| Displacement `z_body` | {0.8, 1.0, 1.25, 1.5, 2.0} ATR |
| FVG body fraction | {0.5, 0.6, 0.7, 0.8} |
| Volume confirmation | {none, 1.0×, 1.25×, 1.5×, 2.0×} vs rolling median |
| Level touch buffer `eps` | {0.0, 0.05, 0.10, 0.25, 0.50} ATR |
| Breakout buffer `buf` | {0.0, 0.05, 0.10, 0.25} ATR |
| VCP contractions | {2, 3, 4} |
| Cup depth | {8%, 12%, 15%, 20%, 25%, 33%, 50%} |
| Handle depth (frac of cup) | {0.2, 0.33, 0.5} |
| Fib levels | {0.236, 0.382, 0.5, 0.618, 0.705, 0.786} |
| Event horizons `h` (bars) | {1, 3, 5, 10, 20, 40, 60} |
| ATR barriers | {0.5, 1.0, 1.5, 2.0, 3.0} ATR |

## Observation schema

One tidy row per detected event. It is a **superset of `EVENTS_COLUMNS`**
(`fundcloud.features.patterns._events.EVENTS_COLUMNS`) so the existing evaluation engine
consumes it unchanged. Columns are split by leak boundary.

### Known at confirmation (the detector writes these)

| column | dtype | notes |
|---|---|---|
| `event_id` | str | catalog id (e.g. `ev_gap_imb_3c`) |
| `asset` | str | symbol — **maps to `EVENTS_COLUMNS.asset`** |
| `timeframe` | str | `"1D"` for now |
| `formation_end_ts` | datetime64[ns, UTC] | last bar the pattern reads (charting anchor) |
| `confirmed_ts` (≡ `observable_ts`) | datetime64[ns, UTC] | leak-free anchor — **maps to `EVENTS_COLUMNS.breakout_ts`** |
| `execution_ts` (≡ `decision_ts`) | datetime64[ns, UTC] | `= confirmed_ts + 1`, assumed fill bar (next-bar open) |
| `direction` | str | `bullish` / `bearish` |
| `params` | str (JSON) | resolved parameters incl. pivot order / scale |
| `logic_version` | int | bumped on any redefinition |
| `params_hash` | str | hash of `(event_id, params, logic_version)` |
| `entry_ref_price` | float | `open[execution_ts]` (the fill) — **maps to `long_entry` / `short_entry`** |
| `stop_ref_price` | float | structural stop, NaN ⇒ ATR fallback — **maps to `stop_price`** |
| `zone_lo`, `zone_hi` | float | gap/OB/level zone bounds (NaN for non-zone events) |
| `quality` | float | optional textbookness score — **maps to `EVENTS_COLUMNS.quality`** |
| `atr_at_confirm` | float | ATR at `confirmed_ts` (vol context + R unit) |
| `vol_regime`, `trend_state` | float / str | regime conditioning, computed on bars `≤ confirmed_ts` |

### Computed after confirmation (the engine writes these — never the detector)

Per horizon `h ∈ {1,3,5,10,20,40,60}`: `return_h`, `net_return_h` (− 6 bps), `mfe_atr_h`,
`mae_atr_h`, `edge_ratio_h`, `time_to_peak_h`, `triple_barrier_label_h`, `fill_frac_h`.

### Reuse mapping — feeds `feature_quality.evaluate()` with no engine change

`feature_quality._build_event_paths` (`metrics/feature_quality.py:203`) reads exactly six
fields: `asset`, `breakout_ts`, `long_entry`/`short_entry` (per direction), `stop_price`,
`quality`. It anchors at `breakout_ts` and measures the forward path from the **next** bar.
So a one-line projection (`confirmed_ts → breakout_ts`, `entry_ref_price → long_entry/short_entry`,
`stop_ref_price → stop_price`) makes the path run from the entry bar against the next-bar-open
entry. Extra columns ride along untouched (`evaluate` only `.get()`s what it needs). The
engine already returns `hit_rate`, `expectancy`, `edge_ratio`, `mfe_atr`, `mae_atr`,
`mae_p95_atr`, `ic`, `icir`, `baseline_hit`, `quality_buckets`, `per_asset`, `time_stability`;
the only genuinely new metrics to add later are `time_to_peak`, `fill_frac`, and triple-barrier
labels.

## Roadmap (each a separate, approved step)

1. **(done)** Clean daily data feed — `fundcloud.research.load_bars` / `clean_panel`.
2. **(this page)** Event catalog + observation schema.
3. Code the three immediate causal detectors (`ev_disp_bar`, `ev_gap_imb_3c`, `ev_sweep_fail`) —
   each gated by the **prefix-invariance test** (contract rule 3).
4. Wire them through `feature_quality.evaluate()` + a frozen train/validation/holdout split.
5. Add the missing inference: triple-barrier, block bootstrap, permutation, Deflated Sharpe, PBO.
6. Pivot-based families, interaction mining, then a portfolio decision layer over holdout survivors.
