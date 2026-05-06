# Geometric Quality Score

> **Scope.** This document is the contract for `quality`, the 0–100
> score attached to every detected formation. Quality measures one thing
> only: **how textbook the formation is, geometrically**. It is not a
> prediction of outcome, regime fit, or expected return — those live in
> `ReliabilityScorer` and `MLScorer`.

## Source of truth

- Implementation: [`crates/fundcloud-core/src/patterns/scoring.rs`](https://github.com/cyberapper/fundcloud/blob/main/crates/fundcloud-core/src/patterns/scoring.rs)
- Public API: `GeometricScorer.score(pattern, ohlcv) -> PatternScore { quality, features }`
- Per-event surface: each event in the events frame carries
  - `quality` (the composite score, `f64` in `0..=100`),
  - `meta["features"]` (each sub-score in `0..=1` for inspection).

If you change a formula, threshold, or weight, **update this file in
the same commit**.

## Composite

```
quality = round(
      0.30 × symmetry
    + 0.25 × volume
    + 0.25 × trendline_r²
    + 0.20 × completeness
), clamped to [0, 100]
```

| Weight | Sub-score | Range |
|--:|---|---|
| 0.30 | `symmetry` | 0–100 |
| 0.25 | `volume` | 0–100 |
| 0.25 | `trendline_r²` | 0–100 |
| 0.20 | `completeness` | 0–100 |

The four sub-scores are independent geometric measurements; the weights
encode an editorial judgment about which dimensions matter most.

## Sub-scores

All sub-scorers are **stateless** and read **only** from
`(pattern, ohlcv[formation_start..=formation_end])`. They never read
future bars or aggregated outcome statistics. This is a structural
property — see [Anti-patterns](#anti-patterns).

### `symmetry` (30%)

Pattern-specific. The formula dispatches on `pattern.name`. All return
`0..=100`; out-of-tolerance differences clamp to 0.

#### Double top / Double bottom

```
diff = pct_diff(pivot[0].price, pivot[2].price)
score = max(0, 100 × (1 − diff / 0.015))
```

`pct_diff(a, b) = |a − b| / ((|a| + |b|) / 2)`.

Returns `0.0` if fewer than 3 pivots — defensive guard.

#### Triple top / Triple bottom

```
trio = (pivot[0].price, pivot[2].price, pivot[4].price)
mean = avg(trio)
worst_diff = max(pct_diff(p, mean) for p in trio)
score = max(0, 100 × (1 − worst_diff / 0.02))
```

Returns `0.0` if fewer than 5 pivots.

#### Head & shoulders / Inverse head & shoulders

```
shoulder_diff = pct_diff(pivot[0].price, pivot[4].price)   # left vs right shoulder
neckline_diff = pct_diff(pivot[1].price, pivot[3].price)   # neckline left vs right
shoulder_score = max(0, 100 × (1 − shoulder_diff / 0.10))
neckline_score = max(0, 100 × (1 − neckline_diff / 0.10))
score = (shoulder_score + neckline_score) / 2
```

Returns `0.0` if fewer than 5 pivots. Note the **head height is not
scored** — only shoulder symmetry and neckline symmetry. A head only 1%
above the shoulders still gets full marks here; gating happens earlier
in the detector.

#### Ascending / Descending / Symmetrical triangle

```
spacings = [pivot[i+1].index − pivot[i].index for i in 0..n-1]
mean_spacing = avg(spacings)
cv = std(spacings) / mean_spacing      # coefficient of variation
score = max(0, 100 × (1 − cv))
```

Triangle symmetry is **temporal**, not price — measures how regularly
the alternating pivots are spaced in time. A perfectly regular triangle
scores 100; high variance in inter-pivot gaps drives the score down.

Floors at `50.0` for fewer than 4 pivots and when `mean_spacing == 0`.

#### Unknown patterns

Any pattern name not handled above returns `50.0` (neutral). Adding a
detector without adding a `symmetry` branch silently caps it at neutral.

### `volume` (25%)

```
volumes = ohlcv.volume[formation_start..=formation_end]
mid = len(volumes) / 2          # integer division
front = mean(volumes[..mid])
back  = mean(volumes[mid..])
ratio = back / front

if ratio <= 0.5:  score = 100
elif ratio >= 1.5: score = 0
else:              score = 100 × (1.5 − ratio)
```

Floors at `50.0` when the formation has ≤ 3 bars or when `front == 0`.

The intuition is Bulkowski's "volume should decline during the
formation, then expand on the breakout". This sub-score only measures
the in-formation decline; breakout-bar volume is not part of `quality`.

### `trendline_r²` (25%)

```
score = mean(
    max(
        trendline_fit_r2(ohlcv.close, tl),
        trendline_fit_r2(ohlcv.high,  tl),
        trendline_fit_r2(ohlcv.low,   tl),
    )
    for tl in pattern.trend_lines
) × 100
```

If no trend lines are attached to the pattern, returns `50.0` (neutral).

**Important — this is *not* the average of `TrendLine::r_squared`.**
That field is the R² of the least-squares fit through the line's anchor
pivots only — and since the line is constructed *to fit* those anchors,
that R² is essentially always ~1.0 by construction (1.0 exactly with two
anchors). It measures "did we draw the line through the points we said
we'd draw it through?", which is trivially true.

`trendline_fit_r2(prices, line)` measures the fit against the
**intermediate bars** between the anchors — the structural question of
whether price actually behaved as if the line were meaningful (acting
as support / resistance) over the formation window. Cleanly-respected
trendlines score near 1.0; cherry-picked anchors with chaotic
intermediate behaviour score near 0.0.

The max-of-three over `(close, high, low)` auto-selects the natural
price series for the line: a low-anchored support line naturally fits
the lows; a high-anchored resistance line fits the highs; a midline-ish
line fits closes. A spurious line that fits *none* of the three remains
correctly scored low.

### `completeness` (20%)

```
completeness = (duration_score + touch_score) / 2
```

#### Duration score (bars in formation)

| `bar_count` | Score |
|---|---|
| `< 5` | 0 |
| `5..10` | linear ramp 0 → 50 |
| `>= 10` | 100 |

Duration is a quality *floor* (need enough bars for the formation to be
visually identifiable), not a quality *ceiling*. A textbook 6-month
double top is no less geometrically clean than a 30-bar one — they're
just different timeframes. Anything past the 10-bar floor scores 100.

#### Touch score (cumulative trend-line touch count)

```
total = sum(tl.touch_count for tl in pattern.trend_lines)

if total <= 2:  score = 30
elif total <= 4: score = 60
else:            score = min(100, 60 + (total − 4) × 10)
```

Step function. A pattern with only the two anchor pivots touching gets
30; cleanly retested formations with 5+ touches saturate at 100.

## Anti-patterns

What `quality` is **not** allowed to do. **If a sub-scorer ever needs
one of these inputs, it does not belong in `GeometricScorer`** —
propose a new scorer (`ReliabilityScorer`, `MLScorer`, etc.).

1. **Read future bars.** Any score that depends on
   `bars[formation_end+1:]` is fitting outcome, not geometry. The Rust
   `score()` signature only takes `OhlcvView<'_>` and a `Pattern` whose
   `formation` slice is bounded.

2. **Read realised outcome statistics** (per-asset historical hit rate;
   per-regime expectancy; an aggregate from the analytics DB). Same
   reason — leaks outcome into geometry. `GeometricScorer` is `Default`
   and stateless.

3. **Be tuned to maximise IC against future returns.** Conflates
   geometry with predictive value; destroys IC's utility as a
   diagnostic. Outcome-based confidence belongs in a separate scorer
   that the user composes downstream.

4. **Vary across runs given identical inputs.** Real-money decisions
   need reproducibility. The scorer is pure (no RNG, no clock, no I/O).

## Canonical fixture set

The fixture set in
[`crates/fundcloud-core/tests/canonical_quality.rs`](https://github.com/cyberapper/fundcloud/blob/main/crates/fundcloud-core/tests/canonical_quality.rs)
is the executable contract for what `quality` should produce on
hand-crafted formations across the documented bands.

- `every_canonical_fixture_lands_in_its_band` — runs by default; passes
  today.
- `calibration_targets_describe_known_gaps` — ignored by default; lists
  fixtures whose desired band the current scorer doesn't satisfy. Run
  with:
  ```bash
  cargo test -p fundcloud-core --test canonical_quality -- --ignored --nocapture
  ```

## Monotonicity tests

The monotonicity tests in
[`scoring.rs#tests`](https://github.com/cyberapper/fundcloud/blob/main/crates/fundcloud-core/src/patterns/scoring.rs)
(prefix `*_monotonic_*`) lock in the *shape* of the scorer's response
on each axis: perturbing one geometric attribute toward "more textbook"
must never decrease the relevant score.
