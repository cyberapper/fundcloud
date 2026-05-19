# Detector & scorer knobs

Every threshold, window, and weight in the detection + scoring pipeline.
For each, the **decision column** says whether the value is exposed as a
config knob, kept hardcoded, or planned to be auto-derived from inputs.

If a knob is currently hardcoded but you have a use case for tuning it,
open an issue — the rules-of-thumb that landed in the source are not
sacred, they're just untuned defaults.

## Conventions

- **Exposed**: settable via the Python class kwargs. Example:
  `DoubleTop(peak_tolerance=0.02, min_trough_depth=0.04).events(bars)`.
- **Keep**: hardcoded for a reason — usually a structural floor (formation
  must have at least N bars to be visually identifiable) or a
  variant-tagging detail that's cosmetic.
- **Auto-scale**: planned to derive from input length/timeframe; not
  user-tunable.

## Pipeline-level knobs (already exposed)

These live on `PatternIndicator` and apply to every detector. Set them as
kwargs on any concrete pattern class (`DoubleTop`, `HeadAndShoulders`, …).

| Knob | Type | Default | Decision | What it does |
|---|---|---|---|---|
| `min_quality` | `float` | `50.0` | exposed | Drop detections whose quality score is below this. |
| `pivot_orders` | `tuple[int, ...]` | `(3, 5, 8)` | exposed | Single-tier pivot orders. Ignored when `pivot_tiers` is set. |
| `pivot_tiers` | `tuple[tuple[int, ...], ...]` | `((3,5,8), (13,21), (34,55))` | exposed | Disjoint pivot scales — each tier runs a separate scan and the union surfaces patterns at multiple horizons. Set to `()` to fall back to single-tier `pivot_orders`. |
| `signal_mode` | `SignalMode` | `BREAKOUT` | exposed | How `transform()` projects events to a per-bar signal: `BREAKOUT` (1.0 on the breakout bar), `FORMATION` (1.0 over the whole formation), or `DECAY` (linear decay from breakout). |
| `decay_bars` | `int` | `5` | exposed | Decay window length when `signal_mode == DECAY`. |
| `condition` | `PatternCondition` | per-pattern preset | exposed | Entry/exit/target/stop rules consumed by `PatternStrategy`. Each pattern class ships a sensible default. |

## Per-detector knobs

### Double top / bottom

`DoubleTop`, `DoubleBottom`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| `peak_tolerance` | `0.015` | exposed | Maximum percentage difference between the two peaks (or troughs). 1.5% by default — tighten for stricter "matching" peaks, loosen for noisy markets. |
| `min_trough_depth` | `0.03` | exposed | Minimum trough depth (or peak height for double bottom) as a fraction of the average peak. 3% by default. |
| `MIN_FORMATION_BARS` | `5` | keep | Minimum bars between the two peaks. Below 5, the formation is too tight to be a pattern. |
| `ADAM_EVE_NEAR_PCT` | `0.015` | keep | Bulkowski Adam/Eve tag — purely a label on `events.variant`, not a detection criterion. |
| `ADAM_EVE_HALF_WINDOW` | `5` | keep | ±5 bars around the pivot for the Adam/Eve tag. Cosmetic on weekly+ bars. |
| `ADAM_MAX_BARS` | `3` | keep | Tag threshold (Adam if ≤3 near-bars, else Eve). |

### Triple top / bottom

`TripleTop`, `TripleBottom`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| `peak_tolerance` | `0.02` | exposed | Max percentage difference across the three peaks/troughs. 2%. |
| `min_trough_depth` | `0.02` | exposed | Min depth as a fraction of average peak. 2%. |
| `min_formation_bars` | `10` | exposed | Minimum bars for the formation. Higher than double because three pivots need more space. |

### Head & shoulders / inverse head & shoulders

`HeadAndShoulders`, `InverseHeadAndShoulders`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| `shoulder_tolerance` | `0.10` | exposed | Max percentage difference between the two shoulders. 10% — H&S in the wild is rarely more symmetric than this. |
| `min_head_prominence` | `0.03` | exposed | Minimum head height above shoulders as a fraction of average shoulder. 3%. |
| `prior_trend_window` | `10` | exposed | Bars before the left shoulder to check for the required prior trend (uptrend for H&S, downtrend for inverse). On daily bars this is two trading weeks — too short for multi-month formations following a flat-on-recent-bars uptrend. |
| `MIN_FORMATION_BARS` | `8` | keep | Minimum total formation length. Below 8 bars an H&S is just three pivots in close succession. |

### Ascending / descending triangle

`AscendingTriangle`, `DescendingTriangle`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| `flat_threshold` | `0.0005` | exposed | Maximum slope (per-bar fraction) for the "flat" side. 0.05% per bar ≈ tight. Loosen if your asset's volatility makes pure flatness unrealistic. |
| `min_touches` | `2` | exposed | Minimum number of pivots touching the flat side. 2 = "two highs at the same price"; raise to 3 for stricter formations. |
| `WRONG_DIR_FRACTION` | `0.7` | keep | Slope-asymmetry multiplier. The flat side may drift in the "wrong" direction by at most `flat_threshold × 0.7` (per-bar slope), so unfavourable drift is tolerated less than favourable drift. Structural filter, not a tunable. |
| `ASC_DESC_MIN_BAR_COUNT` | `8` | keep | Structural minimum. |
| `CHANNEL_TOLERANCE` | `0.02` | future | Currently used to reject formations that look more like a channel than a triangle. Could be exposed if users hit false negatives. |

### Symmetrical triangle

`SymmetricalTriangle`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| `min_touches` | `2` | exposed | Same as ascending/descending. |
| `min_slope_threshold` | `0.0005` | exposed | Minimum absolute slope (in either direction) for a side to count as sloped. Below this, the side is treated as flat — and a flat side disqualifies the pattern from being "symmetrical". |
| `prior_trend_window` | `20` | exposed | Sets the bull/bear directional label only — does not gate detection. |
| `SYM_ABS_TOLERANCE_FRACTION` | `0.05` | keep | Apex tolerance — structural. |
| `SYM_MIN_BAR_COUNT` | `10` | keep | Structural minimum. |

### Pivot detection

Lives in `crates/fundcloud-core/src/patterns/pivots.rs`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| Dedup radius | `±2 bars` | keep | When merging pivots across orders, two same-kind pivots within ±2 bars collapse into the more extreme one. Below 2 it would over-merge; above 2 it would lose distinct nearby pivots. |

The user-facing pivot knobs (`pivot_tiers`, `pivot_orders`) are at the
pipeline level, above.

## Scorer knobs (`GeometricScorer`)

Lives in `crates/fundcloud-core/src/patterns/scoring.rs`.

| Knob | Default | Decision | What it does |
|---|---|---|---|
| `symmetry` weight | `0.30` | keep | Composite weight on the symmetry sub-score. |
| `volume` weight | `0.25` | keep | Volume sub-score weight. |
| `trendline_r2` weight | `0.25` | keep | Anchor-only R² of attached trend lines; informative for 3+ anchor lines (triple_top / triple_bottom / well-pivoted triangle sides), trivially 1.0 for 2-anchor lines. See `docs/scoring/quality.md`. |

### Calibrated per-pattern `min_quality` defaults

Subclasses override `min_quality` to preserve the top-X% selectivity the
old `min_quality=50` floor gave on the prior scorer. Recalibrated against
a real-data corpus (~50 US large/mid-caps + sector ETFs + commodity/FX
proxies, 2018-2026 dailies) after the boundary-respect + role-aware fix;
the synthetic-GBM column is the prior recommendation kept for reference.
Override per instance if your asset class needs a tighter / looser cutoff.

| Pattern | Real-data (default) | Synthetic-GBM | Δ |
|---|---|---|---|
| `double_top`, `double_bottom` | `75.0` | `75.0` | 0 |
| `triple_top`, `triple_bottom` | `71.0` | `66.0` | +5 |
| `head_and_shoulders` | `67.0` | `73.0` | -6 |
| `inverse_head_and_shoulders` | `68.0` | `73.0` | -5 |
| `ascending_triangle`, `descending_triangle` | `74.0` | `74.0` | 0 |
| `symmetrical_triangle` | `73.0` | `73.0` | 0 |

Real-data corpus snapshot: `/tmp/calibration-real/events_{pre,post}.parquet`
(~50 tickers, 8866 detections post-fix vs 11145 pre-fix). Differences of
≤3 points were treated as sampling noise and not promoted.
| `completeness` weight | `0.20` | keep | Completeness sub-score weight. |
| Duration floor | `5 bars` | keep | Below 5 bars, duration score is 0. |
| Duration saturation | `10 bars` | keep | At ≥10 bars, duration score saturates at 100 (no long-pattern penalty). |
| Touch-count thresholds | `≤2 → 30, ≤4 → 60, …` | keep | Trendline touch contribution to completeness. |

If you want to deploy a custom scorer, the supported route is:

1. Subclass / replace `GeometricScorer` in your application code.
2. Pair it with a `ReliabilityScorer` (statistical) or `MLScorer`
   (learned) for outcome-based confidence — the geometric scorer is
   deliberately blind to forward returns. See
   [`docs/scoring/quality.md`](../../scoring/quality.md) for the
   philosophy.

## What's not configurable, on purpose

- **The set of detector names.** Adding a new detector is a code change
  (subclass `PatternDetector`, register in `detector_for`). The library
  is opinionated about the v1 catalogue.
- **The events-table schema.** `EVENTS_COLUMNS` is a stable contract.
- **The scorer's sub-score formulas.** They're geometric primitives.
  Changing one is a code change, not a knob — see the
  [scorer spec](../../scoring/quality.md) for what each sub-score
  measures and the rationale.

## Examples

```python
from fundcloud.features.patterns import DoubleTop, HeadAndShoulders

# Looser double top (e.g. for crypto)
loose = DoubleTop(peak_tolerance=0.03, min_trough_depth=0.05)
events = loose.events(bars)

# Stricter H&S, only the strongest formations
strict = HeadAndShoulders(
    min_quality=70.0,
    shoulder_tolerance=0.05,
    min_head_prominence=0.05,
    prior_trend_window=20,
)
events = strict.events(bars)

# Long-window only — skip the short-formation tier
long_only = DoubleTop(pivot_tiers=((13, 21), (34, 55)))
events = long_only.events(bars)
```
