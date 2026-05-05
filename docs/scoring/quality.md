# Geometric Quality Score — Specification

> **Scope.** This document is the contract for `quality`, the 0–100 score
> attached to every detected formation. Quality measures one thing only:
> **how textbook the formation is, geometrically**. It is not a prediction
> of outcome, regime fit, or expected return — those live in
> `ReliabilityScorer` and `MLScorer`.
>
> See `docs/DECISIONS.md#quality-is-geometry-only` for the design rationale.

## Source of truth

- Implementation: [`crates/fundcloud-core/src/patterns/scoring.rs`](../../crates/fundcloud-core/src/patterns/scoring.rs)
- Public API: `GeometricScorer.score(pattern, ohlcv) -> PatternScore { quality, features }`
- Per-event surface: each event in the events frame carries
  - `quality` (the composite score, `f64` in `0..=100`),
  - `meta["features"]` (each sub-score in `0..=1` for inspection),
  - `meta["scorer_version"]` (semver string — see [Versioning](#versioning)).

Anything in this document is a claim about what the implementation does. If
you change a formula, threshold, or weight, **update this file in the same
commit**. If a number here lacks a citation, it carries a
`TODO(no-source)` marker — every such marker is a calibration target.

## Composite

```
quality = round(
      0.30 × symmetry
    + 0.25 × volume
    + 0.25 × trendline_r²
    + 0.20 × completeness
), clamped to [0, 100]
```

| Weight | Sub-score | Range | Documented source |
|--:|---|---|---|
| 0.30 | `symmetry` | 0–100 | TODO(no-source). Inherited from reference Python. Calibration target. |
| 0.25 | `volume` | 0–100 | TODO(no-source). Inherited from reference Python. Calibration target. |
| 0.25 | `trendline_r²` | 0–100 | TODO(no-source). Inherited from reference Python. Calibration target. |
| 0.20 | `completeness` | 0–100 | TODO(no-source). Inherited from reference Python. Calibration target. |

The four sub-scores are independent geometric measurements; the weights
encode an editorial judgment about which dimensions matter most. Until
calibration (see [Calibration record](#calibration-record)) the weights
are an inherited prior, not a fitted parameter. Treat as such when
interpreting absolute scores.

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

| Knob | Value | Source |
|---|---|---|
| Tolerance | 0.015 (1.5%) | TODO(no-source). Inherited from reference Python. Likely Bulkowski-derived but unverified. |
| Pivot indices | 0 (first peak/trough), 2 (second peak/trough) | Detector output contract. |

Returns `0.0` if fewer than 3 pivots — defensive guard, should never
happen for a detected double pattern.

#### Triple top / Triple bottom

```
trio = (pivot[0].price, pivot[2].price, pivot[4].price)
mean = avg(trio)
worst_diff = max(pct_diff(p, mean) for p in trio)
score = max(0, 100 × (1 − worst_diff / 0.02))
```

| Knob | Value | Source |
|---|---|---|
| Tolerance | 0.02 (2%) | TODO(no-source). Wider than double's 1.5%; rationale undocumented. |

Returns `0.0` if fewer than 5 pivots.

#### Head & shoulders / Inverse head & shoulders

```
shoulder_diff = pct_diff(pivot[0].price, pivot[4].price)   # left vs right shoulder
neckline_diff = pct_diff(pivot[1].price, pivot[3].price)   # neckline left vs right
shoulder_score = max(0, 100 × (1 − shoulder_diff / 0.10))
neckline_score = max(0, 100 × (1 − neckline_diff / 0.10))
score = (shoulder_score + neckline_score) / 2
```

| Knob | Value | Source |
|---|---|---|
| Shoulder tolerance | 0.10 (10%) | TODO(no-source). Bulkowski 2nd ed. p.317 cites "8% within each other" — possible discrepancy worth verifying. |
| Neckline tolerance | 0.10 (10%) | TODO(no-source). Same. |
| Equal weight (50/50) | — | TODO(no-source). Equal split is defensible but not derived. |

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

| Knob | Value | Source |
|---|---|---|
| Floor for < 4 pivots | 50.0 | TODO(no-source). Neutral fallback, undocumented. |
| Floor when `mean_spacing == 0` | 50.0 | Numerical guard. |

Triangle symmetry is **temporal**, not price — measures how regularly
the alternating pivots are spaced in time. A perfectly regular triangle
scores 100; high variance in inter-pivot gaps drives the score down.

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

| Knob | Value | Source |
|---|---|---|
| Lower bound (full credit) | ratio ≤ 0.5 | TODO(no-source). |
| Upper bound (zero credit) | ratio ≥ 1.5 | TODO(no-source). |
| Floor when formation ≤ 3 bars | 50.0 | Defensive — too few bars to estimate halves. |
| Floor when `front == 0` | 50.0 | Division guard. |

The intuition is Bulkowski's "volume should decline during the
formation, then expand on the breakout". This sub-score only measures
the in-formation decline; breakout-bar volume is not part of `quality`.

### `trendline_r²` (25%)

```
score = mean(trendline_fit_r2(ohlcv.close, tl) for tl in pattern.trend_lines) × 100
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

| Knob | Value | Source |
|---|---|---|
| Price series evaluated against | `ohlcv.close` | v1.1 convention. Refining to per-line price series (highs for resistance lines, lows for support lines) is a v1.2 candidate. |

Introduced in `SCORER_VERSION = 1.1.0`. Prior versions read the
anchor-only `r_squared` field directly, which produced a near-constant
score of ~95–100 across all detections and contributed no
discriminative signal.

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

| Knob | Value | Source |
|---|---|---|
| Min viable formation | 5 bars | TODO(no-source). |
| Saturation point | 10 bars | `SCORER_VERSION = 1.2.0`. |

**`SCORER_VERSION 1.1.0 → 1.2.0` change** — duration is a quality
*floor* (need enough bars for the formation to be visually
identifiable), not a quality *ceiling*. A textbook 6-month double top
is not less geometrically clean than a 30-bar one — they're just
different timeframes. The previous penalty (decay 100 → 50 over
60..120 bars, floor 50 past 120 bars) actively biased the system
against long-window patterns and combined with `min_quality=50` to
filter them out. Now any formation past the 10-bar floor scores 100
on the duration component, regardless of length.

Short-formation handling (under 10 bars) is unchanged.

#### Touch score (cumulative trend-line touch count)

```
total = sum(tl.touch_count for tl in pattern.trend_lines)

if total <= 2:  score = 30
elif total <= 4: score = 60
else:            score = min(100, 60 + (total − 4) × 10)
```

| Knob | Value | Source |
|---|---|---|
| Step thresholds (2, 4) | TODO(no-source). |
| Step values (30, 60) | TODO(no-source). |
| Increment above 4 | +10 per touch | TODO(no-source). |

Step function. A pattern with only the two anchor pivots touching gets
30; cleanly retested formations with 5+ touches saturate at 100.

## Versioning

The scorer is a **stable contract**. Every event written to the events
frame carries `meta["scorer_version"]` so that any downstream record
(detection, trade, evaluation) is traceable to the exact scorer build
that produced it.

- `SCORER_VERSION` is defined in [`scoring.rs`](../../crates/fundcloud-core/src/patterns/scoring.rs)
  and propagated through the PyO3 binding into Python.
- Bumping rules (semver-like, where each level corresponds to the
  granularity of the change):
  - **Patch** (`x.y.Z`): bug fix that does not change scores by more
    than `\|Δquality\| ≤ 1` on the canonical fixture set.
  - **Minor** (`x.Y.0`): adds a sub-score, adds a pattern, or any change
    that moves canonical scores by `\|Δquality\| > 1` but is monotonic
    in geometric quality (i.e., better-looking patterns still score
    higher).
  - **Major** (`X.0.0`): redefinition of what `quality` measures (e.g.,
    repurposing as outcome-aware). Reserved — not anticipated.
- Every version bump must:
  1. Update `SCORER_VERSION` in `scoring.rs`.
  2. Update this document.
  3. Update / regenerate canonical fixture scores under
     `tests/fixtures/scoring/canonical/` if expected.
  4. Note the change in `docs/CHANGELOG.md`.

## Calibration record

The scorer's relationship to "what a domain expert would call textbook"
is currently **uncalibrated**. The calibration target is Spearman rank
correlation `ρ` between scorer output and a hand-rated sample of real
detections.

| Date | Sample (n) | Rater | Held-out ρ | 95% CI | Scorer version |
|---|---|---|---|---|---|
| 2026-05-05 | 60 | claude (AI) | +0.289 ± 0.154 | weights wide-open | 1.0.0 |
| 2026-05-05 | (skipped) | — | — | — | 1.1.0 |
| 2026-05-05 | 69 | claude (AI) | +0.335 ± 0.126 | weights wide-open | 1.2.0 |

**2026-05-05 run notes** (`scripts/scoring/calibration_interim_n60.json`):
- This run is a *baseline*, not a production calibration — the rater
  is Claude using vision-based judgment of the rendered charts. A
  human-rater run is still required before deploying any scorer
  change. Treat the ρ as indicative, not authoritative.
- **Structural findings that don't depend on rater identity**:
  1. `trendline_r2` is **near-constant at 0.98 ± 0.08** across all 60
     detections — trendlines forced through anchor pivots always
     fit r² ≈ 1.0. The 25% weight is contributing no discriminative
     signal. Either rework the formula (e.g., measure fit on
     intermediate bars between anchors, not the anchors themselves)
     or drop the component.
  2. Production weights (30/25/25/25/20) and uniform (25/25/25/25)
     produce **identical Spearman** (0.492 vs 0.496). The current
     weighting is not load-bearing. The fitted optimizer collapses
     to uniform too.
  3. CV ρ = 0.289 (below the 0.5 threshold) — the four-component
     decomposition is **structurally incomplete**. Pushing samples
     larger won't fix this; new sub-scores will.
- **Recommendations before next calibration run**:
  - ✅ Fix or replace `trendline_r2` so it actually varies across
    detections. — **Done in `SCORER_VERSION = 1.1.0`** (see notes
    below).
  - Consider pattern-family-specific symmetry sub-scores
    (double-top symmetry ≠ H&S symmetry ≠ triangle symmetry).
  - Add candidate sub-scores: pivot prominence, inter-pivot return
    magnitude, formation tightness vs surrounding volatility.

**`SCORER_VERSION 1.0.0 → 1.1.0` change** (2026-05-05):
- `score_trendline` now reads `trendline_fit_r2(prices, line)` —
  the R² of the line against the **intermediate bars** between
  anchors — rather than the anchor-only `TrendLine::r_squared`.
  See [`crates/.../trendline.rs`](../../crates/fundcloud-core/src/patterns/trendline.rs).
- Each line is evaluated against the maximum of `close`, `high`, and
  `low` to auto-select the line's natural price series (a low-anchored
  neckline naturally fits lows; a high-anchored resistance line fits
  highs). Refining `TrendLine` to carry the anchor kind explicitly is
  a v1.2 candidate that would let us drop the max-of-three.
- Empirical impact across the full Mag7 + SPY + QQQ universe (4999
  detections):
  - `trendline_r2` distribution shifts from
    **mean=0.98 std=0.08** (near-constant under v1.0) to
    **mean=0.06 std=0.14, range [0.00, 0.84]** under v1.1 — i.e., the
    component now carries real signal.
  - Quality bands redistribute: from **74 good / 68 marginal / 36
    poor** under v1.0 to **1 good / 91 marginal / 79 poor** under v1.1.
    The earlier "good" tier was largely a free 25 points from the
    constant-1.0 trendline component; v1.1 surfaces this honestly.
- `calibration_interim_n60.json` is preserved as the v1.0 baseline.
  A fresh v1.1 calibration requires re-rating — the prior 60 ratings
  only overlap 19 detections with the new sample, and within that
  overlap `trendline_r2` is constant zero so individual-component ρ is
  undefined.

**`SCORER_VERSION 1.1.0 → 1.2.0` change** (2026-05-05):
- `score_completeness` removes the long-formation duration penalty.
  Previously `bar_count > 60` decayed 100 → 50 over 60..120 and
  floored at 50 past 120. v1.2 saturates at 100 once `bar_count >= 10`,
  consistent with the philosophy that quality measures geometric
  textbookness only — not a preferred timeframe (see
  `feedback_quality_geometric_only`). The 5-bar minimum and 5..10 ramp
  are unchanged.
- Companion change in the **Python detection layer** (not in the
  scorer): `PatternIndicator` now defaults to **multi-tier pivot
  scanning** (`pivot_tiers=((3,5,8), (13,21), (34,55))`). The detector
  windows over the merged alternating pivot sequence; with all orders
  in one sweep, small-order pivots clutter that sequence and hide
  major swings. Running disjoint tiers exposes patterns at three
  scales — short, intermediate, multi-month. Empirical impact on AAPL
  full history (1980–2026, daily, 11.4k bars):

  | min_quality | OLD detections | NEW detections | NEW long-window (>60 bars) | NEW multi-quarter (>120 bars) |
  |---|---|---|---|---|
  | 50 | 182 | 206 (+13%) | 22 | 8 |
  | 30 | 648 | 751 (+16%) | 99 | 53 |

  The largest H&S formation under min_quality=30 spans 604 bars
  (~2.4 years on dailies). Whether it's a *good* H&S is a quality
  question — the detector now sees it instead of fragmenting it.

The calibration workflow lives in
[`scripts/scoring/calibrate.py`](../../scripts/scoring/calibrate.py).
Add a row to this table on every new calibration run.

## Anti-patterns

What `quality` is **not** allowed to do, with the rationale and the
test that protects against it. **If a sub-scorer ever needs one of
these inputs, it does not belong in `GeometricScorer`** — propose a new
scorer (`ReliabilityScorer`, `MLScorer`, etc.).

1. **Read future bars.**
   _Why._ Lookahead. Any score that depends on `bars[formation_end+1:]`
   is fitting outcome, not geometry.
   _Protection._ The Rust `score()` signature only takes
   `OhlcvView<'_>` and a `Pattern` whose `formation` slice is bounded.
   `tests/scoring/test_no_lookahead.py` (TODO) asserts that score is
   invariant to changes in `ohlcv[formation_end+1:]`.

2. **Read realised outcome statistics** (e.g., per-asset historical
   hit rate; per-regime expectancy; an aggregate from the analytics
   DB).
   _Why._ Same as above — leaks outcome into geometry.
   _Protection._ `GeometricScorer` is `Default` and stateless. Do not
   add cached parameters that are loaded from outcome aggregates.

3. **Be tuned to maximise IC against future returns.**
   _Why._ Conflates geometry with predictive value; destroys IC's
   utility as a diagnostic.
   _Protection._ The calibration target (above) is rank-agreement with
   a **hand-rated** sample, not with realised returns.

4. **Vary across runs given identical inputs.**
   _Why._ Real-money decisions need reproducibility.
   _Protection._ The scorer is pure (no RNG, no clock, no I/O).
   `tests/scoring/test_determinism.py` (TODO) asserts bitwise-equal
   scores across repeated calls.

## Canonical fixture set

The fixture set in [`crates/fundcloud-core/tests/canonical_quality.rs`](../../crates/fundcloud-core/tests/canonical_quality.rs)
is the executable contract for what `quality` should produce on
hand-crafted formations across the documented bands.

- `every_canonical_fixture_lands_in_its_band` — runs by default; passes
  today.
- `calibration_targets_describe_known_gaps` — ignored by default; lists
  fixtures whose desired band the current scorer doesn't satisfy. Each
  one maps to a calibration TODO. Run with:
  ```bash
  cargo test -p fundcloud-core --test canonical_quality -- --ignored --nocapture
  ```

Promoting a calibration target into the contract requires bumping
`SCORER_VERSION`.

## Monotonicity tests

The monotonicity tests in
[`scoring.rs#tests`](../../crates/fundcloud-core/src/patterns/scoring.rs)
(prefix `*_monotonic_*`) lock in the *shape* of the scorer's response
on each axis: perturbing one geometric attribute toward "more textbook"
must never decrease the relevant score. These tests are independent of
the absolute weighting; they survive any future calibration that keeps
the scorer monotonic.

## Open calibration TODOs

Each item is a known reason the scorer is not yet defensible at scale:

- Replace every `TODO(no-source)` threshold above with either a
  citation (Bulkowski, Edwards & Magee, …) or a value derived from
  data via the calibration script.
- Address the three `CALIBRATION_TARGETS` in
  `crates/fundcloud-core/tests/canonical_quality.rs`:
  composite-vs-symmetry weighting, short-duration penalty, head
  prominence in H&S symmetry.
- Hand-rate ≥ 200 real detections; fit weights via
  `scripts/scoring/calibrate.py`; record `ρ` in the calibration table
  above.
- Decide whether timeframe-aware completeness ranges are needed.
