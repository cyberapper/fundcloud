# Chart Pattern Detection

The `fundcloud.features.patterns` subpackage detects classical chart
patterns (Head & Shoulders, Double Top/Bottom, Triangles, etc.) and
exposes them as sklearn-compatible feature transformers. Detection
itself runs in pure Rust under `fundcloud._core` for speed; the Python
layer wraps it as `IndicatorSpec` subclasses so patterns compose with
the rest of the feature pipeline (TA-Lib indicators, `FeaturePipeline`,
`Simulator.run_signals`, `PurgedKFold`).

For a runnable end-to-end walkthrough, see the example scripts
`examples/31_head_and_shoulders_detection.py` (synthetic data) and
`examples/32_pattern_scan_real_data.py` (real Yahoo Finance bars).

## Status

The full v1 surface — 9 tier-1 reversal/continuation detectors — is now
shipped. Each one is a Rust module implementing `PatternDetector` plus a
thin Python subclass of `PatternIndicator` registered via
`@register_indicator`.

| Detector | Rust | Python class | Status |
|---|---|---|---|
| Head and Shoulders | ✅ | `HeadAndShoulders` | shipped |
| Inverse Head and Shoulders | ✅ | `InverseHeadAndShoulders` | shipped |
| Double Top | ✅ | `DoubleTop` | shipped |
| Double Bottom | ✅ | `DoubleBottom` | shipped |
| Triple Top | ✅ | `TripleTop` | shipped |
| Triple Bottom | ✅ | `TripleBottom` | shipped |
| Ascending Triangle | ✅ | `AscendingTriangle` | shipped |
| Descending Triangle | ✅ | `DescendingTriangle` | shipped |
| Symmetrical Triangle | ✅ | `SymmetricalTriangle` | shipped |

## Architecture at a glance

```text
┌─────────────────────────────────────────────────────────────────┐
│ Python: fundcloud.features.patterns                             │
│   PatternIndicator(IndicatorSpec)        ← sklearn fit/transform │
│   ├── HeadAndShoulders / InverseHeadAndShoulders                 │
│   ├── DoubleTop / DoubleBottom                                   │
│   ├── TripleTop / TripleBottom                                   │
│   └── AscendingTriangle / DescendingTriangle / SymmetricalTriangle│
│                                                                  │
│   PatternCondition (entry/exit descriptor + presets)             │
│   events table (canonical 14-column schema)                      │
│   Enums (Pattern, Direction, SignalMode, EntryRule, …)           │
└──────────────────┬──────────────────────────────────────────────┘
                   │  numpy zero-copy + PyO3
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Rust: crates/fundcloud-core/src/patterns/                       │
│   types.rs        Pivot / TrendLine / Pattern / Detection       │
│   pivots.rs       multi_level_pivots()                          │
│   trendline.rs    fit_trendline() + boundary helpers            │
│   scoring.rs      GeometricScorer  (0–100 quality)              │
│   detect.rs       PatternDetector trait + scan() entry point    │
│   detectors/                                                     │
│     head_shoulders.rs    HeadShoulders + Inverse                │
│     double.rs            DoubleTop + DoubleBottom                │
│     triple.rs            TripleTop + TripleBottom                │
│     triangles.rs         Asc / Desc / Symmetrical Triangle       │
└─────────────────────────────────────────────────────────────────┘
```

## Data flow — bars in, detections out

```text
1. User: bars (pd.DataFrame, MultiIndex columns (field, asset),
                              DatetimeIndex)
   │
   │ HeadAndShoulders().fit_transform(bars)
   ▼
2. PatternIndicator._compute(per_asset)
   │ pull aligned numpy arrays:
   │   ts_ns = index.view("int64")   (UTC nanoseconds)
   │   open, high, low, close, volume = numpy float64
   │
   │ py.allow_threads → one PyO3 call per (pattern × asset)
   ▼
3. _core.scan_pattern("head_and_shoulders", ts, o, h, l, c, v,
                       pivot_orders, min_quality)
   │
   │ Rust side (no GIL):
   │   a. Build OhlcvView (zero-copy borrows of numpy buffers)
   │   b. multi_level_pivots(highs, lows, ts, orders=(3,5,8))
   │   c. PatternDetector::detect(&pivots, ohlcv) for the named pattern
   │   d. GeometricScorer::score(&pattern, ohlcv) for each detection
   │   e. Filter by min_quality
   │
   │ return Vec<Detection> as Python list[dict]
   ▼
4. build_events_frame()  → canonical 14-column events DataFrame
   events_to_signal()    → per-bar 0/1 signal series (BREAKOUT mode)
   ▼
5. User: pd.DataFrame[index, asset]   (one column per asset)
```

## Pivot detection — `multi_level_pivots`

The first stage of every pattern scan. It identifies the swing
highs and swing lows (collectively, "pivots") that the detector then
walks in sequence.

### Definition: swing high / swing low at `order=N`

A bar at index `i` qualifies as a **swing high at order N** when its
price is `>=` every neighbour within `N` bars on each side:

```text
x[i] >= x[i-1], x[i-2], ..., x[i-N]
x[i] >= x[i+1], x[i+2], ..., x[i+N]
```

Out-of-bounds neighbours are clipped to the boundary value (matching
scipy's `argrelextrema(comparator, order=N, mode='clip')`). Swing lows
use `<=` against `lows[i]`.

### Multi-scale union + dedup

`multi_level_pivots(highs, lows, ts_ns, orders)` runs the detection at
every order in `orders` (smallest to largest), then deduplicates.

| Step | What |
|---|---|
| 1. Detect | For each order, find all swing highs (on `highs[]`) and swing lows (on `lows[]`) |
| 2. Union | Concatenate every order's pivot list |
| 3. Sort | By `(index, kind)` |
| 4. Dedup | For same-kind pivots within ±2 bars: keep the strictly more extreme price; on equal prices, the first-iterated (smallest order) wins |
| 5. Alternate | Walk left-to-right; collapse consecutive same-kind pivots, keeping the more extreme. Output is strictly alternating High/Low |

### Inputs / outputs

| | Type |
|---|---|
| `highs`, `lows` | `&[f64]` of equal length, same as the OHLCV panel |
| `ts_ns` | `&[i64]` UTC nanoseconds, monotonic ascending |
| `orders` | `&[usize]` lookback half-window sizes; default `[3, 5, 8]` |
| **returns** | `Vec<Pivot>`, alternating High/Low, sorted by `index` |

### Pivot record

```rust
pub struct Pivot {
    pub index: usize,    // bar offset into the OHLCV panel
    pub ts_ns: i64,      // UTC nanoseconds
    pub price: f64,      // highs[index] for High, lows[index] for Low
    pub kind: PivotKind, // High | Low
    pub order: u8,       // smallest order that detected this pivot (metadata)
}
```

### `pivot_orders` — what it actually controls

Per the math above, **the smallest order in `orders` determines the
pivot count**. Adding larger orders to the tuple does not add or remove
pivots — they are a strict subset. Concretely:

```text
pivot_orders=(3,)            → N pivots
pivot_orders=(3, 5)          → N pivots (identical bars)
pivot_orders=(3, 5, 8)       → N pivots (identical bars)
pivot_orders=(3, 5, 8, 13)   → N pivots (identical bars)

pivot_orders=(5,)            → M < N pivots (different smallest order)
pivot_orders=(8,)            → P < M pivots
```

The `order` field on each surviving pivot is **the smallest order that
detected it** — set during dedup (smaller orders iterate first; equal
prices don't replace). It is recorded as metadata; v1 detectors do not
read it. Future detectors or quality scorers may use it to prefer
"macro" pivots, but as of v1 it is purely informational.

**Practical guidance**: for v1, you may treat `pivot_orders` as a
single-element tuple holding the lookback you actually want
(`(3,)`, `(5,)`, etc.). The default `(3, 5, 8)` is kept for parity
with pattern-service.

### Tuning by bar timeframe

| Bars | Suggested `pivot_orders` |
|---|---|
| Daily equities (default) | `(3, 5, 8)` |
| Daily crypto / very volatile | `(5, 8, 13)` |
| Intraday 5-minute | `(2, 5)` |
| Weekly bars | `(2, 4)` |
| Macro setups only | `(8, 13)` |

Tune by the **bars' character** (timeframe, volatility), not by the
pattern. The same `pivot_orders` is appropriate for every detector
running on the same bars.

## Trend-line fitting — `fit_trendline`

Closed-form ordinary least squares through 2+ pivots. Used by every
detector to construct necklines, channels, and triangle sides.

### Algorithm

For pivots with `(index_i, price_i)`:

```text
x_mean = mean(index_i)
y_mean = mean(price_i)
S_xx   = sum((x_i - x_mean)^2)
S_yy   = sum((y_i - y_mean)^2)
S_xy   = sum((x_i - x_mean) * (y_i - y_mean))

slope     = S_xy / S_xx
intercept = y_mean - slope * x_mean
r_squared = 1 - (S_yy - slope * S_xy) / S_yy   (clamped to [0, 1])
```

**Rank-deficient branch**: when all pivots share the same `index`
(`S_xx = 0`), the closed form would divide by zero. The fallback
(matching numpy `lstsq`'s minimum-norm solution) is `slope=0`,
`intercept=y_mean`. `r_squared` is `1.0` if `S_yy=0` (constant `y`),
else `0.0`.

### Output

```rust
pub struct TrendLine {
    pub start_index: usize,    // index of the leftmost anchor pivot
    pub end_index: usize,      // index of the rightmost anchor pivot
    pub slope: f64,            // dPrice / dBar
    pub intercept: f64,        // y at bar 0
    pub r_squared: f64,        // [0, 1]
    pub touch_count: u8,       // number of pivots used in the fit
    pub role: Role,            // Upper / Lower — which side of the line
                               // the scorer evaluates for boundary respect
}
```

The companion helpers `validate_boundaries` (channel-width sanity
check) and `count_touches` (how many bars come within tolerance of the
line) are used by triangle / channel / rectangle detectors and are
documented inline in `crates/fundcloud-core/src/patterns/trendline.rs`.

## Geometric quality score — `GeometricScorer`

Every detection is graded on a `0..=100` composite score. The scorer
is stateless and pattern-aware (the symmetry sub-scorer dispatches by
pattern name).

### Composition

```text
raw = 30% * symmetry + 25% * volume + 25% * trendline_r2 + 20% * completeness

symmetry_gate = clamp(symmetry / 10, 0.1, 1.0)
duration_gate = clamp((bar_count − 4) / 6, 0, 1)   if bar_count < 10, else 1.0

quality = round(raw * symmetry_gate * duration_gate), clamped to [0, 100]
```

The two multiplicative gates crush the composite when a structural
prerequisite fails (near-zero symmetry, or formations shorter than the
detector minimum). See [`docs/scoring/quality.md`](../scoring/quality.md#composite-gates)
for the full rationale and motivating cases.

### Sub-scorers

#### `symmetry` (30%)

Pattern-specific. Returns `0..=100`.

| Pattern name | Formula |
|---|---|
| `head_and_shoulders` / `inverse_head_and_shoulders` | `(shoulder_score + neckline_score) / 2`, where each is `100 * (1 - pct_diff(pair) / 0.10)`, clamped at 0 |
| `double_top` / `double_bottom` | `100 * (1 - pct_diff(p1, p3) / 0.015)`, clamped at 0 |
| `ascending_triangle` / `descending_triangle` | `100 * (1 - cv(pivot_spacings))`, clamped at 0; `cv = std/mean` of the spacing-between-pivots series |
| anything else | `50.0` (neutral) |

`pct_diff(a, b) = abs(a - b) / ((|a| + |b|) / 2)`. Symmetry below
roughly 0.7 starts to look "off"; tight formations score above 90.

#### `volume` (25%)

Declining volume during the formation is a confirmation signal.

```text
ratio = mean(volumes[mid:]) / mean(volumes[:mid])
where mid = len(volumes) // 2

ratio <= 0.5  → 100  (ideal)
ratio >= 1.5  → 0
otherwise     → 100 * (1.5 - ratio)
```

If volume data is unavailable or the formation is < 4 bars, returns
`50.0` (neutral).

#### `trendline_r2` (25%)

Trend-line quality sub-score, scaled to 0–100. The metric dispatches by
touch count:

- **3+ anchor lines** (triple_top / triple_bottom and well-pivoted
  triangle sides) use the mean anchor-only `TrendLine::r_squared`. This
  varies in `[0, 1]` and reflects how cleanly the pivots line up.
- **2-anchor lines** (double_top / double_bottom, H&S necklines,
  2-touch triangle sides) use the boundary-respect ratio — the fraction
  of intermediate bars whose high/low respects the line within a 0.5%
  tolerance. This is implemented by `features.trendline_r2` in
  `crates/fundcloud-core/src/patterns/features/trendline.rs` and is a
  genuine discriminator for these patterns, not a constant.

If no trend lines are attached (e.g., a pivot-only detection), returns
`50.0`. See [`docs/scoring/quality.md`](../scoring/quality.md) for the
dispatch rules and rationale.

#### `completeness` (20%)

Combines duration and trend-line touch count.

```text
duration_score:
  bar_count < 5         → 0
  5 <= bar_count < 10   → linear ramp 0 → 50
  10 <= bar_count <= 60 → 100   (the sweet spot)
  60 < bar_count <= 120 → linear ramp 100 → 50
  bar_count > 120       → 50

touch_score (sum of TrendLine.touch_count across the detection):
  total <= 2 → 30
  total <= 4 → 60
  total > 4  → min(100, 60 + (total - 4) * 10)

completeness = (duration_score + touch_score) / 2
```

### Output

```rust
pub struct PatternScore {
    pub quality: f64,                       // 0..=100
    pub features: HashMap<String, f64>,     // {symmetry, volume, trendline_r2, completeness} as 0..=1 floats
}
```

The `features` map is preserved verbatim into the events table's
`meta["features"]` so users can inspect the breakdown.

## PatternDetector trait

Every detector implements:

```rust
pub trait PatternDetector: Send + Sync {
    fn name(&self) -> &'static str;   // matches Pattern enum value
    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern>;
}
```

`pivots` are pre-computed by `multi_level_pivots`; `ohlcv` is borrowed
zero-copy from the user's numpy arrays. The detector returns
**unscored** `Pattern`s; the geometric scorer is applied centrally by
`run_detector`.

## Detector reference

### Head and Shoulders (`HeadShouldersDetector`)

**Direction**: `Bearish`. **Stable name**: `"head_and_shoulders"`.

A reversal pattern: a peak (head) flanked by two lower peaks
(shoulders) of similar height, sitting above a neckline drawn through
the two intervening lows. The breakout below the neckline projects a
measured-move target equal to the head-to-neckline distance below the
neckline.

#### Input (per detector call)

| Argument | Description |
|---|---|
| `pivots: &[Pivot]` | Alternating swing highs and lows from `multi_level_pivots` |
| `ohlcv: OhlcvView<'_>` | Borrowed view of the bars (used for prior-trend gating) |

#### Validation rules

The detector slides a 5-pivot window over `pivots`. A window passes
when **every** condition below holds:

| # | Rule | Default |
|---|---|---|
| 1 | Sequence is `H-L-H-L-H` (kind alternation) | always |
| 2 | Head price strictly above both shoulder prices | always |
| 3 | `pct_diff(left_shoulder, right_shoulder) <= shoulder_tolerance` | `0.10` (10%) |
| 4 | `(head - avg_shoulder) / avg_shoulder >= min_head_prominence` | `0.03` (3%) |
| 5 | `right_shoulder.index - left_shoulder.index >= MIN_FORMATION_BARS` | `8` bars |
| 6 | `prior_trend_slope(closes, head_left.index, window=10) > 0` (uptrend before reversal) | always |

`prior_trend_slope` is a closed-form OLS slope over the `window` bars
immediately preceding the formation start. It returns `0.0` when there
are fewer than 3 prior bars or `mean(prior_closes) == 0`. Returning
exactly `0.0` from this helper means "no signal — reject", not "flat
is OK".

#### Output (per match)

```rust
Pattern {
    name: "head_and_shoulders",
    direction: Direction::Bearish,
    pivots: vec![h1, l1, h2, l2, h3],            // the 5 anchors
    trend_lines: vec![neckline, resistance],      // both empty if fit fails
    formation: (h1.index, h3.index),
    entry_price: Some(neckline.price_at(h3.index)),
    breakout_price: Some(neckline.price_at(h3.index)),
    variant: None,
}
```

- **Neckline** = OLS fit through `(l1, l2)`. Falls back to
  `(l1.price + l2.price) / 2` if the fit is rejected (only happens
  when the two lows share an index — extremely rare with alternating
  pivots).
- **Resistance line** = OLS fit through `(h1, h3)`. Used only as
  metadata — does not feed into `entry_price`.
- **Entry / breakout price** = the neckline value projected to the
  right-shoulder bar (`h3.index`). This is what charting platforms
  render and is what the events table uses as the breakout level.

#### Reference

`crates/fundcloud-core/src/patterns/detectors/head_shoulders.rs`,
struct `HeadShouldersDetector`.

### Inverse Head and Shoulders (`InverseHeadShouldersDetector`)

**Direction**: `Bullish`. **Stable name**: `"inverse_head_and_shoulders"`.

The mirror of the regular variant — a trough (head) flanked by two
higher troughs (shoulders), with a neckline drawn through the two
intervening highs.

#### Validation rules

Identical structure to the regular detector, with everything mirrored
(`<=` in place of `>=` and vice versa):

| # | Rule | Default |
|---|---|---|
| 1 | Sequence is `L-H-L-H-L` | always |
| 2 | Head price strictly below both shoulder prices | always |
| 3 | `pct_diff(left_shoulder, right_shoulder) <= shoulder_tolerance` | `0.10` |
| 4 | `(avg_shoulder - head) / avg_shoulder >= min_head_prominence` | `0.03` |
| 5 | `right_shoulder.index - left_shoulder.index >= MIN_FORMATION_BARS` | `8` bars |
| 6 | `prior_trend_slope(closes, head_left.index, window=10) < 0` (downtrend before reversal) | always |

#### Output

```rust
Pattern {
    name: "inverse_head_and_shoulders",
    direction: Direction::Bullish,
    pivots: vec![l1, h1, l2, h2, l3],
    trend_lines: vec![neckline, support],         // neckline through h1,h2; support through l1,l3
    formation: (l1.index, l3.index),
    entry_price: Some(neckline.price_at(l3.index)),
    breakout_price: Some(neckline.price_at(l3.index)),
    variant: None,
}
```

#### Reference

`crates/fundcloud-core/src/patterns/detectors/head_shoulders.rs`,
struct `InverseHeadShouldersDetector`.

### Double Top / Double Bottom

**Stable names**: `"double_top"` (Bearish, `H-L-H`),
`"double_bottom"` (Bullish, `L-H-L`).

Two peaks (or troughs) at approximately the same level separated by an
intervening trough (peak). Each detection carries a Bulkowski variant
in `Pattern.variant`:

* **`STRICT`** when the second extreme does not breach the first
  (resistance / support held on both tests — the textbook case);
  **`WEAK`** otherwise.
* Each pivot is then tagged **`ADAM`** (narrow 1–3 bar spike) or
  **`EVE`** (rounded reversal, ≥ 5 nearby bars) based on how many bars
  in a ±5 window sit within 1.5% of the pivot price.
* Final variant strings: `"STRICT_ADAM_ADAM"` … `"WEAK_EVE_EVE"`.

| # | Rule | Default |
|---|---|---|
| 1 | Sequence is `H-L-H` (Double Top) or `L-H-L` (Double Bottom) | always |
| 2 | `pct_diff(p1, p3) <= peak_tolerance` | `0.015` (1.5%) |
| 3 | Trough depth / peak height ≥ `min_prominence` (avg-relative) | `0.03` (3%) |
| 4 | `p3.index - p1.index >= MIN_FORMATION_BARS` | `5` bars |

**Output**: 3-pivot formation, neckline = `p2.price` (the trough for
tops, the peak for bottoms), entry / breakout = neckline, optional
resistance / support trend line through `(p1, p3)`.

Reference: `crates/fundcloud-core/src/patterns/detectors/double.rs`.

### Triple Top / Triple Bottom

**Stable names**: `"triple_top"` (Bearish, `H-L-H-L-H`),
`"triple_bottom"` (Bullish, `L-H-L-H-L`).

Three peaks (or troughs) at approximately the same level. Distinguished
from Head-and-Shoulders by all three extremes being roughly equal — H&S
has a prominent middle peak (head). The breakout level is the *worst*
intervening pivot (Bulkowski's confirmation rule):

* **Triple Top**: neckline = `min(p2, p4)` — the lowest valley.
* **Triple Bottom**: neckline = `max(p2, p4)` — the highest peak.

Using the average would trigger entries at a price the formation has
already traded through, which isn't a confirmed breakout.

| # | Rule | Default |
|---|---|---|
| 1 | Sequence alternates 5 pivots | always |
| 2 | Each peak/trough within `peak_tolerance` of the trio's mean | `0.02` (2%) |
| 3 | Pattern depth / height ≥ `min_prominence` of the mean | `0.02` (2%) |
| 4 | `p5.index - p1.index >= min_bar_count` | `10` bars |

Reference: `crates/fundcloud-core/src/patterns/detectors/triple.rs`.

### Ascending / Descending / Symmetrical Triangle

**Stable names**: `"ascending_triangle"` (Bullish, flat resistance +
rising support), `"descending_triangle"` (Bearish, falling resistance +
flat support), `"symmetrical_triangle"` (direction inferred from prior
trend, falling resistance + rising support).

Triangles are the only family that runs `validate_boundaries`: every
bar inside the formation must stay within the channel formed by the two
trend lines. Asc / Desc use a fraction-of-channel-width tolerance (2%);
Symmetric uses an absolute-price tolerance (5% of the starting gap)
because the channel collapses to zero near the apex.

Asymmetric flat-line tolerance is applied for the asc / desc detectors:
the full `flat_threshold` is allowed in the consistent direction, but
only 70% of it is allowed in the wrong direction (a strongly-rising
support contradicts the descending-triangle thesis, and vice versa).

| # | Rule | Default |
|---|---|---|
| 1 | Flat leg normalised slope within asymmetric `flat_threshold` band (asc / desc) | `0.0005` |
| 2 | Sloping leg normalised slope strictly in the right direction (sym: \|slope\| > `min_slope_threshold`) | `0.0005` |
| 3 | Lines must converge (end gap < start gap, both positive) | always |
| 4 | Every bar in formation stays within the channel under the chosen tolerance | always |
| 5 | Formation length ≥ `min_bar_count` | `8` (asc / desc), `10` (sym) |

Direction labelling for symmetric triangles uses
`prior_trend_slope(closes, formation_start, prior_window=10)`:
`Bullish` when slope > 0, `Bearish` when slope < 0, `Bullish` as a
fallback when slope == 0 (insufficient history or flat data — preserves
the reference Python's behaviour).

Overlapping detections are deduplicated, keeping the one with more
pivots; an "overlap" is > 50% of the shorter formation's length.

Reference: `crates/fundcloud-core/src/patterns/detectors/triangles.rs`.

## PyO3 bindings

The Rust core is exposed under `fundcloud._core` (a single flat module
following the existing convention; no nested submodule).

### `_core.scan_pattern`

```python
_core.scan_pattern(
    name: str,                # e.g. "head_and_shoulders"
    ts_ns: np.ndarray[int64], # UTC nanoseconds, monotonic ascending
    open:   np.ndarray[float64],
    high:   np.ndarray[float64],
    low:    np.ndarray[float64],
    close:  np.ndarray[float64],
    volume: np.ndarray[float64],
    pivot_orders: list[int],
    min_quality: float,
) -> list[dict]
```

Returns a list of detection dicts (see schema below). All numpy arrays
must have identical length. The GIL is released around the actual
scan via `py.allow_threads`, so a thread pool over multiple
`(pattern × asset)` pairs scales linearly.

### `_core.multi_level_pivots`

```python
_core.multi_level_pivots(
    highs:  np.ndarray[float64],
    lows:   np.ndarray[float64],
    ts_ns:  np.ndarray[int64],
    orders: list[int],
) -> list[dict]
```

Exposed for testing and advanced use; `scan_pattern` calls this
internally.

### `_core.list_pattern_names`

```python
_core.list_pattern_names() -> list[str]
```

Returns the registered detector names — all 9 v1 detectors:
`["head_and_shoulders", "inverse_head_and_shoulders", "double_top",
"double_bottom", "triple_top", "triple_bottom", "ascending_triangle",
"descending_triangle", "symmetrical_triangle"]`.

### Detection dict schema (PyO3 output)

| Key | Type | Description |
|---|---|---|
| `name` | `str` | Stable lowercase pattern identifier |
| `direction` | `str` | `"bullish"` / `"bearish"` / `"neutral"` |
| `pivots` | `list[dict]` | `{index, ts_ns, price, kind, order}` per pivot |
| `trend_lines` | `list[dict]` | `{start_index, end_index, slope, intercept, r_squared, touch_count, role}` — `role` is `"upper"` or `"lower"`, set by the detector and used by the scorer to pick which side of the line to evaluate. |
| `formation_start` | `int` | Bar offset of the formation start |
| `formation_end` | `int` | Bar offset of the formation end |
| `entry_price` | `float` or `None` | Where the strategy is "entered" — usually the breakout level |
| `breakout_price` | `float` or `None` | Same as entry for v1 |
| `variant` | `str` or `None` | Pattern-specific subclass label (e.g., `"STRICT_ADAM_ADAM"`) |
| `quality` | `float` | `0..=100` from `GeometricScorer` |
| `features` | `dict[str, float]` | Sub-scores: `symmetry`, `volume`, `trendline_r2`, `completeness` (each 0..=1) |

## Python feature layer

### `PatternIndicator(IndicatorSpec)`

The base class every concrete pattern indicator subclasses. It plugs
into the same `IndicatorSpec` machinery as TA-Lib indicators, so it
composes naturally with `FeaturePipeline`, `FeatureStore`, and
`PurgedKFold`.

#### Class-level attributes

| Attribute | Type | Description |
|---|---|---|
| `inputs` | `tuple[str, ...]` | `("open", "high", "low", "close", "volume")` — required Bars fields |
| `outputs` | `tuple[str, ...]` | `("signal",)` — single per-bar float column |
| `pattern_name` | `str` | Stable lowercase Rust detector key (matches `Pattern` enum value) |
| `condition` | `PatternCondition` | Default entry/exit preset; per-instance overridable |
| `default_params` | `dict` | `{"min_quality": 50.0, "pivot_orders": (3, 5, 8), "signal_mode": SignalMode.BREAKOUT, "decay_bars": 5}` |

#### Public methods

```python
indicator.fit_transform(bars) -> pd.DataFrame
    # Sklearn standard. Returns a wide signal panel:
    #   index   = bars.index
    #   columns = one per asset
    #   dtype   = float64 (1.0 on breakout, 0.0 otherwise; varies by signal_mode)

indicator.events(bars) -> pd.DataFrame
    # Rich event log with the canonical 14-column schema; one row per
    # detected pattern across all assets.

indicator.effective_condition  # property
    # Returns the active PatternCondition (per-instance override or class preset).
```

### Required input frame shape

`bars` must be a `pd.DataFrame` with:

- **MultiIndex columns** `(field, asset)` where
  `field ∈ {"open", "high", "low", "close", "volume"}`
- **`pd.DatetimeIndex`** — naive timestamps are treated as UTC; `tz`-aware
  timestamps are converted to UTC before being passed to Rust
- **Sorted ascending** by index
- **All five OHLCV fields present** for every asset (`KeyError` raised
  otherwise)

The shape `(T, 5 × n_assets)` for `T` bars and `n_assets` assets.

### Output: per-bar signal panel

```python
signals = indicator.fit_transform(bars)
```

| | Type |
|---|---|
| index | `pd.DatetimeIndex` matching `bars.index` |
| columns | one per asset (single-output indicator → no `__asset` suffix) |
| dtype | `float64` |

Cell value semantics depend on `signal_mode`:

| `SignalMode` | Cell value |
|---|---|
| `BREAKOUT` (default) | `1.0` on each `breakout_ts` bar; `0.0` elsewhere |
| `FORMATION` | `1.0` from `formation_start` to `formation_end` inclusive; `0.0` outside |
| `DECAY` | `1.0` on the breakout bar, decaying linearly to `0.0` over `decay_bars` |

This shape is exactly what `Simulator.run_signals(entries, exits)`
consumes, so backtesting is a direct plug-in.

### Output: events table

```python
events = indicator.events(bars)
```

The canonical schema is identical for every detector (so user code
generalises cleanly across patterns):

| Column | Type | Description |
|---|---|---|
| `pattern` | `Pattern` (enum) | The pattern identifier |
| `asset` | `str` | Column name from `bars` |
| `direction` | `Direction` (enum) | `BULLISH` / `BEARISH` / `NEUTRAL` |
| `formation_start` | `pd.Timestamp` | UTC, first pivot of the formation |
| `formation_end` | `pd.Timestamp` | UTC, last pivot of the formation |
| `breakout_ts` | `pd.Timestamp` or `pd.NaT` | When the breakout was confirmed (v1: equals `formation_end`) |
| `entry_price` | `float` | Where the strategy enters (v1: neckline @ right edge) |
| `breakout_price` | `float` or `NaN` | Same as `entry_price` in v1 |
| `target_price` | `float` or `NaN` | Filled by `apply_condition` (v1: `NaN` until Phase 7) |
| `stop_price` | `float` or `NaN` | Filled by `apply_condition` (v1: `NaN` until Phase 7) |
| `quality` | `float` | 0–100 from `GeometricScorer` |
| `variant` | `str` or `None` | Pattern-specific label (e.g., for double tops) |
| `pivots` | `list[dict]` | `[{ts, price, kind}]` per anchor pivot — useful for chart overlays |
| `meta` | `dict` | Pattern-specific extras: `features` (the sub-scores) and `trend_lines` |

The schema is exposed as `EVENTS_COLUMNS` from
`fundcloud.features.patterns` for assertions.

### Configuration knobs

Pass at construction or override with `set_params`:

```python
HeadAndShoulders(
    min_quality=50.0,        # quality cutoff (0–100); detections below are dropped
    pivot_orders=(3, 5, 8),  # see "pivot_orders" section above
    signal_mode=SignalMode.BREAKOUT,  # how events project to per-bar signals
    decay_bars=5,            # window for SignalMode.DECAY
    condition=...,           # PatternCondition override (see below)
)
```

### `PatternCondition`

Frozen dataclass describing the entry / exit rules a strategy applies
to detected events. Mirrors the convention used by
`fundcloud.strategies.scheduler.Cadence` — one source of truth, with
`override(...)` returning a new instance.

```python
@dataclass(frozen=True, slots=True)
class PatternCondition:
    entry_rule:    EntryRule    = EntryRule.ON_BREAKOUT
    exit_rule:     ExitRule     = ExitRule.TARGET_OR_STOP
    target_method: TargetMethod = TargetMethod.MEASURED_MOVE
    stop_method:   StopMethod   = StopMethod.BELOW_PIVOT
    time_stop_bars: int | None  = None
    atr_window:     int         = 14
    atr_multiple:   float       = 2.0

    def override(self, **kwargs) -> PatternCondition: ...
```

Each detector class ships a sensible preset on the class
(`HeadAndShoulders.condition`); per-instance overrides go via the
`condition=` constructor argument. `override` accepts both Enum values
and their `.value` strings, so:

```python
HeadAndShoulders(
    condition=PatternCondition().override(
        entry_rule="on_pullback",   # str or EntryRule.ON_PULLBACK both work
        time_stop_bars=20,
        target_method=TargetMethod.FIB_1_618,
    )
)
```

### Enums

All in `fundcloud.features.patterns`:

```python
class Pattern(str, Enum):
    HEAD_AND_SHOULDERS         = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP                 = "double_top"
    DOUBLE_BOTTOM              = "double_bottom"
    TRIPLE_TOP                 = "triple_top"
    TRIPLE_BOTTOM              = "triple_bottom"
    ASCENDING_TRIANGLE         = "ascending_triangle"
    DESCENDING_TRIANGLE        = "descending_triangle"
    SYMMETRICAL_TRIANGLE       = "symmetrical_triangle"

class Direction(str, Enum):
    BULLISH, BEARISH, NEUTRAL

class SignalMode(str, Enum):
    BREAKOUT, FORMATION, DECAY

class EntryRule(str, Enum):
    ON_BREAKOUT, ON_FORMATION_COMPLETE, ON_PULLBACK

class ExitRule(str, Enum):
    TARGET_OR_STOP, TIME_STOP, TRAILING_STOP

class TargetMethod(str, Enum):
    MEASURED_MOVE, FIB_1_618, FIXED_ATR

class StopMethod(str, Enum):
    BELOW_PIVOT, ATR_MULTIPLE, FIXED_PCT
```

All public APIs accept `EnumType | str`. Use the `.coerce(value, EnumType)`
helper to coerce at the boundary; it raises `ValueError` with a list of
valid values on miss.

## Composition examples

### With `FeaturePipeline`

```python
from fundcloud.features import FeaturePipeline
from fundcloud.features.indicators import RSI
from fundcloud.features.patterns import HeadAndShoulders

pipe = FeaturePipeline([
    ("rsi", RSI(timeperiod=14)),
    ("hns", HeadAndShoulders(min_quality=70)),
])
panel = pipe.fit_transform(bars)
```

### Backtest with `Simulator.run_signals`

```python
import fundcloud  # registers .fc accessor
from fundcloud.features.patterns import InverseHeadAndShoulders

bars = ...  # MultiIndex Bars frame
entries = InverseHeadAndShoulders().fit_transform(bars).astype(bool)
exits   = entries.shift(20).fillna(False).astype(bool)  # 20-bar holding period
result  = bars.fc.run_signals(entries, exits, size=0.1)
print(result.summary())
```

### Cross-validate pattern parameters with `PurgedKFold`

```python
from sklearn.model_selection import GridSearchCV
from fundcloud.validate import PurgedKFold

search = GridSearchCV(
    HeadAndShoulders(),
    param_grid={"min_quality": [50, 60, 70, 80]},
    cv=PurgedKFold(n_splits=5, purge=20),
)
```

## Performance characteristics

- **Per `(pattern × asset)` scan**: O(T) for pivot detection + O(P × k)
  for the detector pass, where `T` is the bar count, `P` is the
  pivot count, and `k` is the detector's window size (5 for H&S).
- **Memory**: zero-copy borrows of the user's numpy buffers; no
  intermediate float64 arrays allocated in Rust beyond the pivot
  list.
- **Concurrency**: `scan_pattern` releases the GIL; concurrent
  `ThreadPoolExecutor` over multiple assets scales linearly. (The
  Python loop in `PatternIndicator.transform` runs assets serially in
  v1; a 2D batched scan is a future optimisation that does not change
  the public API.)

Empirically, scanning 11k bars × 9 tickers × 2 detectors completes in
well under a second on commodity hardware.

## Feature-quality metrics

The `fundcloud.metrics.feature_quality` submodule grades a pattern's
predictive power on a given OHLCV universe. Not flat-reexported from
`fundcloud.metrics` (avoids collision with `metrics.win_rate`); import
as:

```python
from fundcloud.metrics import feature_quality as fq
```

### `evaluate(events, bars, *, horizons, condition=None, trade_direction='natural', baseline=True)`

Headline panel — one row per horizon. Columns:

| Column | Meaning |
|---|---|
| `n_events` | Events with sufficient lookahead at this horizon |
| `hit_rate` | Fraction of events where directional close-to-entry move > 0 at `t+h` |
| `baseline_hit` | Asset-weighted unconditional `P(close[t+h] − close[t] in direction > 0)` — the random-entry yardstick |
| `expectancy` | Mean realised R-multiple. R = signed move / stop distance |
| `edge_ratio` | `avg(MFE_atr) / avg(MAE_atr)` — payoff asymmetry |
| `mfe_atr` | Average maximum favourable excursion in ATR units (intraday max forward `high` minus entry, or mirror for bearish) |
| `mae_atr` | Average maximum adverse excursion in ATR units |
| `mae_p95_atr` | 95th-percentile MAE — the stop-sizing reference |
| `ic` | Spearman ρ between event `quality` and signed forward return |
| `icir` | Mean / std of yearly ICs — stability of the IC across periods |
| `throwback` | Fraction of events that re-touch entry within 10 bars |

`condition` (optional): if a `PatternCondition` is passed, the metric
calls `apply_condition` first to fill `target_price` / `stop_price` per
the condition's target / stop methods. R-multiples then use the real
stop distance instead of the 1×ATR fallback.

`trade_direction`: `"natural"` (default) uses each event's emitted
direction; `"inverse"` flips every event (test the fade-the-pattern
hypothesis); `"long"` / `"short"` force a fixed side. The baseline is
transformed in lockstep so the comparison stays honest.

### Stratified diagnostics

| Function | View |
|---|---|
| `quality_buckets(events, bars, *, horizon, n_buckets=5)` | Quintile events by `quality`; one row per bucket. Validates the geometric scorer — monotonic Q1→Q5 means the scorer earns its weight. |
| `per_asset(events, bars, *, horizon)` | One row per asset — discovery tool for asset-specific deployment lists. |
| `time_stability(events, bars, *, horizon, n_folds=5)` | Equal-event-count chronological folds — exposes regime sensitivity. |

### Scalar primitives

For ad-hoc use or composition into custom dashboards, the per-event
scalars expose the same logic without going through the bundle:

`hit_rate`, `expectancy`, `edge_ratio`, `avg_mfe_atr`, `avg_mae_atr`,
`mae_p95_atr`, `throwback_rate`, `information_coefficient`, `icir`.

All take a list of `_EventPath` objects (the internal aligned-path
representation) so they can be combined cheaply across stratifications.

## `apply_condition` and `PatternStrategy`

`fundcloud.features.patterns.apply_condition(events, condition, bars)`
returns a copy of the events table with `target_price` and
`stop_price` filled per the supplied `PatternCondition`. Pattern
height is derived from the events table's pivots:

* Bullish: `entry − min(low_pivot_prices)`
* Bearish: `max(high_pivot_prices) − entry`

`fundcloud.strategies.PatternStrategy(indicator, *, condition=None,
size=0.1, inverse=False)` is the long-only backtest wrapper. `init`
runs the indicator once and applies the condition; `decide` walks the
per-bar context, opens trades on event timestamps, and closes them on
intraday target hit, intraday stop hit, or `time_stop_bars`. Bearish
events are skipped unless `inverse=True`, which flips every event's
direction so the strategy long-trades the inverse hypothesis.

`bars.fc.run_pattern(pattern, *, condition, size, inverse, **params)`
is the one-liner accessor that wraps the indicator + strategy +
simulator into a `SimResult`.

## Limitations and future work
- **`breakout_ts` fires at `formation_end`**, not at a confirmed
  close-through-neckline breakout. This inflates `throwback_rate`
  systematically (the "breakout" bar is right next to the neckline
  by construction) and makes some events fire one or two bars before
  the textbook entry. A future phase will add price-action breakout
  confirmation as an opt-in mode.
- **No regime-aware scoring** — pattern-service's `ReliabilityScorer`
  blends `GeometricScorer` with empirical hit-rate per
  (direction × timeframe × regime × asset_bucket). Out of scope for v1.
- **No ML scorer** — pattern-service ships an XGBoost
  `MLScorer` predicting realised R-multiple at a 20-bar horizon. Out
  of scope for v1.
- **`pivot.order` is recorded but unused** — see the discussion under
  "Pivot detection". Future detectors or quality scorers may
  consume it.
- **No streaming / tick-by-tick path** — every scan is a batch over
  the whole bar range. The PyO3 layer has the seams to add a
  streaming wrapper later without breaking the API.
- **`PatternStrategy` is long-only** — bearish events are skipped
  unless `inverse=True` (which flips them to long). Native short-side
  trading is a follow-up (depends on simulator support for naked
  shorts).

## File reference

| Concern | File |
|---|---|
| Rust types | `crates/fundcloud-core/src/patterns/types.rs` |
| Pivot detection | `crates/fundcloud-core/src/patterns/pivots.rs` |
| Trend-line fit | `crates/fundcloud-core/src/patterns/trendline.rs` |
| Geometric scorer | `crates/fundcloud-core/src/patterns/scoring.rs` |
| Detector trait + scan entry point | `crates/fundcloud-core/src/patterns/detect.rs` |
| H&S detector pair | `crates/fundcloud-core/src/patterns/detectors/head_shoulders.rs` |
| Double Top / Bottom | `crates/fundcloud-core/src/patterns/detectors/double.rs` |
| Triple Top / Bottom | `crates/fundcloud-core/src/patterns/detectors/triple.rs` |
| Triangle trio | `crates/fundcloud-core/src/patterns/detectors/triangles.rs` |
| PyO3 bindings | `crates/fundcloud-py/src/lib.rs` (`scan_pattern`, `multi_level_pivots`, `list_pattern_names`) |
| Python `PatternIndicator` | `python/fundcloud/features/patterns/_base.py` |
| Python condition descriptor | `python/fundcloud/features/patterns/_condition.py` |
| Events frame + projection | `python/fundcloud/features/patterns/_events.py` |
| Public enums | `python/fundcloud/features/patterns/_enums.py` |
| Indicator subclasses | `python/fundcloud/features/patterns/{head_and_shoulders, inverse_head_and_shoulders, double_top, double_bottom, triple_top, triple_bottom, ascending_triangle, descending_triangle, symmetrical_triangle}.py` |
| Synthetic walkthrough | `examples/31_head_and_shoulders_detection.py` |
| Real-data scan | `examples/32_pattern_scan_real_data.py` |
