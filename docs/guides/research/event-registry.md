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

Each geometric branch is its own `event_id` (its identity **is** the detection
equation, per contract rule 5 "Sides are separate records"); the detector's base
id (kept here for reference) is used only to key `params_hash` so both branches of
one detector call share one hash.

| event_id | name | family | branch | base id | status |
|---|---|---|---|---|---|
| `ev_disp_up` | Displacement bar (up) | price_action | up | `ev_disp_bar` | **implemented** (`research.detect_displacement`) |
| `ev_disp_dn` | Displacement bar (down) | price_action | down | `ev_disp_bar` | **implemented** (`research.detect_displacement`) |
| `ev_gap_up` | Fair Value Gap up (3-candle imbalance) | price_action | up | `ev_gap_imb_3c` | **implemented** (`research.detect_fvg`) |
| `ev_gap_dn` | Fair Value Gap down (3-candle imbalance) | price_action | down | `ev_gap_imb_3c` | **implemented** (`research.detect_fvg`) |
| `ev_sweep_up` | Liquidity sweep / failed breakout (support) | price_action | up | `ev_sweep_fail` | **implemented** (`research.detect_sweep_fail`) |
| `ev_sweep_dn` | Liquidity sweep / failed breakout (resistance) | price_action | down | `ev_sweep_fail` | **implemented** (`research.detect_sweep_fail`) |
| `ev_donchian_up` | Donchian N-day breakout (up) | volatility | up | `ev_donchian_break` | **implemented** (`research.detect_donchian`) |
| `ev_donchian_dn` | Donchian N-day breakout (down) | volatility | down | `ev_donchian_break` | **implemented** (`research.detect_donchian`) |
| `ev_keyrev_up` | Outside-bar key reversal (up) | price_action | up | `ev_outside_reversal` | **implemented** (`research.detect_key_reversal`) |
| `ev_keyrev_dn` | Outside-bar key reversal (down) | price_action | down | `ev_outside_reversal` | **implemented** (`research.detect_key_reversal`) |
| `ev_opengap_up` | Opening-gap continuation (up) | price_action | up | `ev_opening_gap` | **implemented** (`research.detect_opening_gap`) |
| `ev_opengap_dn` | Opening-gap continuation (down) | price_action | down | `ev_opening_gap` | **implemented** (`research.detect_opening_gap`) |
| `ev_inside_bar` | Inside bar (compression) | volatility | neutral | `ev_inside_bar` | **implemented** (`research.detect_inside_bar`) |
| `ev_nr_squeeze` | NRn range contraction | volatility | neutral | `ev_nr_squeeze` | **implemented** (`research.detect_nr_squeeze`) |
| `ev_sr_bounce_up` | Support/resistance touch + hold (up) | structure | up | `ev_sr_touch_bounce` | **implemented** (`research.detect_sr_touch_bounce`) |
| `ev_sr_bounce_dn` | Support/resistance touch + hold (down) | structure | down | `ev_sr_touch_bounce` | **implemented** (`research.detect_sr_touch_bounce`) |
| `ev_ob_up` | Order block (bullish, last opposing candle) | price_action | up | `ev_ob_impulse_last_opp` | **implemented** (`research.detect_order_block`) |
| `ev_ob_dn` | Order block (bearish, last opposing candle) | price_action | down | `ev_ob_impulse_last_opp` | **implemented** (`research.detect_order_block`) |
| `ev_vcp_contract` | Volatility contraction | volatility | long | — | proposed |
| `ev_pivot_break` | Pivot/base breakout | volatility | long | — | proposed |
| `ev_sr_break_retest` | Break and retest | structure | both | — | proposed |
| `ev_retrace_ratio` | Fibonacci/retracement-to-zone | structure | both | — | proposed |
| `ev_cup_u` | Cup (rounded base) | classical | long | — | proposed |
| `ev_cup_handle_break` | Cup and handle breakout | classical | long | — | proposed |
| `ev_tline_compress` | Trendline / triangle compression | structure | both | — | proposed |
| `ev_acc_range` / `ev_spring` / `ev_sos` | Accumulation / spring / sign-of-strength | structure | long | — | proposed |
| `head_and_shoulders`, `inverse_head_and_shoulders`, `double_top`, `double_bottom`, `triple_top`, `triple_bottom`, `ascending_triangle`, `descending_triangle`, `symmetrical_triangle` | Classical chart patterns | classical | both | — | **implemented** (`features/patterns/`) |

The 9 classical patterns already ship as Rust detectors (`fundcloud.features.patterns`); they
are listed here so this page is the one registry. Their v1 `breakout_ts = formation_end`
(`features/patterns/_events.py:85`) is **not** strictly neighbour-locked for pivot patterns —
when consumed by the event engine they must be re-anchored to `formation_end + max_pivot_order`.

The **`neutral`** branch (e.g. `ev_inside_bar`, `ev_nr_squeeze`) is for events with no inherent
geometric side — a range contraction precedes an expansion of *unknown* sign. A neutral event
carries a **single, suffix-less `event_id`**. It is first-class in the evidence layer
(`forward_paths` / `outcome_profile` read only the realised path, so `side="auto"` decides its
direction from the data); it is **excluded by design** from the legacy `to_events_frame` →
`feature_quality` bridge, which keys the long/short entry off the `_up` / `_dn` suffix. Forcing a
fake suffix would invent a phantom trade mandate the doctrine forbids.

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

### Implemented batch 2 (bar-local + reused pivot machinery)

#### `ev_donchian_break` — N-day channel breakout (`ev_donchian_up` / `ev_donchian_dn`)
- **family / direction:** volatility / up + down. **`research.detect_donchian`.**
- **formation_logic:** trailing channel over the **half-open** window `[t−N, t)` (excludes `t`):
  `hi_prior = max(high[t−N … t−1])`, `lo_prior = min(low[t−N … t−1])`. Up: `close[t] > hi_prior +
  buf · ATR_n(t−1)`; down: `close[t] < lo_prior − buf · ATR_n(t−1)`. Stop = the opposite boundary.
- **parameters:** `N ∈ {10,20,40,60,120,252}` (252 ≈ 52-week); `buf ∈ {0.0,0.05,0.10,0.25}` ATR;
  `atr_n ∈ {10,14,20}`.
- **confirmed_ts:** `= t`. A **trailing** rolling extreme is *not* a centred pivot, so there is **no
  neighbour-lock** and no future read — the key contrast with the pivot families.

#### `ev_outside_reversal` — outside-bar key reversal (`ev_keyrev_up` / `ev_keyrev_dn`)
- **family / direction:** price_action / up + down. **`research.detect_key_reversal`.**
- **formation_logic:** outside bar `high[t] > high[t−1]` **and** `low[t] < low[t−1]`; up adds
  `close[t] > high[t−1]` and `CLV ≥ clv_min`; down adds `close[t] < low[t−1]` and `CLV ≤ 1 − clv_min`.
  Stop = `low[t]` (up) / `high[t]` (down).
- **parameters:** `clv_min ∈ {0.6,0.7,0.8}`; `atr_n ∈ {10,14,20}`.
- **confirmed_ts:** `= t` — bars `t−1, t` only, no pivots.

#### `ev_opening_gap` — overnight-gap continuation (`ev_opengap_up` / `ev_opengap_dn`)
- **family / direction:** price_action / up + down. **`research.detect_opening_gap`.**
- **formation_logic:** up: `open[t] > close[t−1] + k · ATR_n(t−1)` **and** `close[t] > open[t]`; down
  mirrors. Zone = the gap void `[close[t−1], open[t]]` (up). Stop = `low[t]` / `high[t]`. Must run on
  the **adjusted / cleaned** panel (gap geometry is split/dividend-sensitive).
- **parameters:** `k ∈ {0.25,0.5,1.0,1.5}` ATR; `atr_n ∈ {10,14,20}`.
- **confirmed_ts:** `= t` — `open[t]` vs `close[t−1]`, both known at `t`'s close. No neighbour-lock.

#### `ev_inside_bar` — inside-bar compression (**neutral**)
- **family / direction:** volatility / **neutral** (single suffix-less id). **`research.detect_inside_bar`.**
- **formation_logic:** `high[t] ≤ high[t−1]` **and** `low[t] ≥ low[t−1]` (strict variant uses `<` / `>`).
  Zone = the mother-bar range `[low[t−1], high[t−1]]`.
- **parameters:** `strict ∈ {False,True}`; `atr_n ∈ {10,14,20}`.
- **confirmed_ts:** `= t`. No neighbour-lock. Side is decided by `side="auto"` (see the neutral note).

#### `ev_nr_squeeze` — NRn range contraction (**neutral**)
- **family / direction:** volatility / **neutral**. **`research.detect_nr_squeeze`.**
- **formation_logic:** `range[t] ≤ min(range[t−n+1 … t−1])` — the narrowest range of the last `n` bars
  (inclusive `≤`, so a tie fires). No zone / stop.
- **parameters:** `n ∈ {4,7,10}` (NR4 / NR7 / NR10); `atr_n ∈ {10,14,20}`.
- **confirmed_ts:** `= t`. Pure trailing window, no neighbour-lock.

#### `ev_sr_touch_bounce` — support/resistance touch + hold (`ev_sr_bounce_up` / `ev_sr_bounce_dn`)
- **family / direction:** structure / up + down. **`research.detect_sr_touch_bounce`.** Same skeleton as
  `ev_sweep_fail` (neighbour-locked levels from `confirmed_pivots`, nearest eligible level), with a
  different trigger: a **touch-and-hold** rather than a pierce-and-reclaim.
- **formation_logic:** up, support `L`: `low[t] ≤ L + eps · ATR_n(t−1)` (a touch, not necessarily a
  pierce) **and** `close[t] > L`; down mirrors with a resistance `R`. Zone = touch band
  `[L − eps · ATR, L + eps · ATR]`. Stop = `low[t]` / `high[t]`.
- **parameters:** `pivot_k ∈ {2,3,5,10}`; `eps ∈ {0.0,0.05,0.10,0.25,0.50}` ATR; `atr_n ∈ {10,14,20}`.
  v1 ships **without** level clustering (`cluster_tol` deferred).
- **confirmed_ts:** `= t`, **provided** every pivot forming the level is neighbour-locked
  (`pivot_index + pivot_k ≤ t`).

#### `ev_ob_impulse_last_opp` — order block (`ev_ob_up` / `ev_ob_dn`)
- **family / direction:** price_action / up + down. **`research.detect_order_block`.**
- **formation_logic (bullish):** at a candidate impulse-close bar `c` with a bullish displacement
  `(close[c] − open[c]) > z_body · ATR_n(c−1)`, find `j` = the **most-recent bearish candle** in
  `[c−m, c−1]`, and require `close[c] > max(high[j−r … j])`. Zone = candle `j`'s body (or its full range
  when `wick=True`); stop = `low[j]`. Bearish mirrors.
- **parameters:** `m ∈ {3,5,10}` (impulse lookback); `r ∈ {1,2,3}` (clearance window);
  `z_body ∈ {1.0,1.25,1.5}` ATR; `atr_n ∈ {10,14,20}`; `wick ∈ {False,True}`.
- **confirmed_ts:** `= c` (the impulse-close bar, **not** `j`). The detector is **impulse-bar-driven**
  (iterates over `c`, scans *backward* for `j`), so every read is `≤ c` ⇒ **no neighbour-lock** and
  prefix-invariance holds. An opposing-candle-driven loop (over `j`, scanning forward) would leak and is
  forbidden.

### Proposed (logic + `confirmed_ts` specified; detector deferred)

- **`ev_vcp_contract` (Volatility contraction).** Over window `W`, ≥ `m_pull` successive swing
  pullbacks of decreasing amplitude `a_1 > a_2 > … > a_m`, declining `ATR/price`, flat/down volume.
  Swings are pivot-derived → **neighbour-lock**: confirmed_ts = last contraction's locked pivot
  (`pivot_index + k`). `m_pull ∈ {2,3,4}`.
- **`ev_pivot_break` (Base/pivot breakout).** `pivot = max(high over base)`; at `t`,
  `close[t] > pivot + buf · ATR`, `volume[t] ≥ z_vol · median(volume, base)`, close in top `q` of
  day range. confirmed_ts = `t`, with the base's defining pivots neighbour-locked.
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
| Donchian channel `N` | {10, 20, 40, 60, 120, 252} (reuses the lookback `W` grid; 252 ≈ 52-week) |
| Gap size `k` | {0.25, 0.5, 1.0, 1.5} ATR |
| NR lookback `n` | {4, 7, 10} |
| Order-block lookback `m` | {3, 5, 10} |
| Order-block clearance `r` | {1, 2, 3} |
| Event horizons `h` (bars) | {1, 3, 5, 10, 20, 40, 60} |
| ATR barriers | {0.5, 1.0, 1.5, 2.0, 3.0} ATR |

## Observation schema

One tidy row per detected event. It is a **superset of `EVENTS_COLUMNS`**
(`fundcloud.features.patterns._events.EVENTS_COLUMNS`) so the existing evaluation engine
consumes it unchanged. Columns are split by leak boundary.

### Known at confirmation (the detector writes these)

| column | dtype | notes |
|---|---|---|
| `event_id` | str | catalog id (e.g. `ev_gap_up`); its `_up` / `_dn` suffix **is** the geometric branch (the two branches are distinct events) |
| `asset` | str | symbol — **maps to `EVENTS_COLUMNS.asset`** |
| `timeframe` | str | `"1D"` for now |
| `formation_end_ts` | datetime64[ns, UTC] | last bar the pattern reads (charting anchor) |
| `confirmed_ts` (≡ `observable_ts`) | datetime64[ns, UTC] | leak-free anchor — **maps to `EVENTS_COLUMNS.breakout_ts`** |
| `execution_ts` (≡ `decision_ts`) | datetime64[ns, UTC] | `= confirmed_ts + 1`, assumed fill bar (next-bar open) |
| `params` | str (JSON) | resolved parameters incl. pivot order / scale |
| `logic_version` | int | bumped on any redefinition |
| `params_hash` | str | hash of `(base_event_id, params, logic_version)` — both `_up` / `_dn` branches of one detector call share one hash (keyed on the detector base id, e.g. `ev_gap_imb_3c`) |
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
fields: `asset`, `breakout_ts`, `long_entry`/`short_entry`, `stop_price`,
`quality`. It anchors at `breakout_ts` and measures the forward path from the **next** bar.
So a one-line projection (`confirmed_ts → breakout_ts`, `entry_ref_price → long_entry/short_entry`,
`stop_ref_price → stop_price`) makes the path run from the entry bar against the next-bar-open
entry. `entry_ref_price` routes to `long_entry` when the `event_id` ends in `_up` and to
`short_entry` when it ends in `_dn` — a **LEGACY geometric bridge** only: the suffix is the
detection equation's branch, **not** a trade mandate (the evidence layer picks the side from
data via `side="auto"`). Extra columns ride along untouched (`evaluate` only `.get()`s what it
needs). The
engine already returns `hit_rate`, `expectancy`, `edge_ratio`, `mfe_atr`, `mae_atr`,
`mae_p95_atr`, `ic`, `icir`, `baseline_hit`, `quality_buckets`, `per_asset`, `time_stability`;
the only genuinely new metrics to add later are `time_to_peak`, `fill_frac`, and triple-barrier
labels.

## Roadmap (each a separate, approved step)

1. **(done)** Clean daily data feed — `fundcloud.research.load_bars` / `clean_panel`.
2. **(this page)** Event catalog + observation schema.
3. **(done)** Code the three immediate causal detectors (`ev_disp_bar`, `ev_gap_imb_3c`,
   `ev_sweep_fail`) in `fundcloud.research.events` — each gated by the **prefix-invariance test**
   (contract rule 3).
4. **(done)** Evidence layer — `research.scan_variants` (neutral multi-detector scan over the
   pre-registered grid, no direction assigned) feeding the data-driven views in
   `research.events.explore` (`forward_paths`, `outcome_profile`, `return_distribution`,
   `event_portfolio`, `evidence_table`). Direction is decided from the measured forward path
   (`side="auto"`), never from the geometric label. Frozen train/validation/holdout split via
   `research.frozen_split`.
5. **(done — batch 2)** Catalog expansion + legible per-variant evaluation: seven new detectors
   (`ev_donchian_break`, `ev_outside_reversal`, `ev_opening_gap`, `ev_inside_bar`, `ev_nr_squeeze`,
   `ev_sr_touch_bounce`, `ev_ob_impulse_last_opp`), each gated by the prefix-invariance test; the
   first **neutral** events; `research.variant_leaderboard` + `research.decode_params` to compare
   parameterisations side-by-side (params decoded into columns, not an opaque hash); and configurable
   grids — `DEFAULT_GRIDS` (tractable) vs `FULL_GRIDS` (registry-wide) with `count_variants` +
   a `max_variants` cap so a wide sweep is always an explicit choice.
6. Add the missing inference: triple-barrier, block bootstrap, permutation, Deflated Sharpe, PBO.
7. Remaining pivot/structure families (VCP, cup, trendline, retrace, Wyckoff, break-retest),
   interaction mining, then a portfolio decision layer over holdout survivors.
