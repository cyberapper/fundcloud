# Scorer calibration history

> Audit trail of `GeometricScorer` calibration runs and the
> version bumps that came out of them. This is a **history doc**, not
> a workflow library users need to follow. The geometric scorer ships
> with the weights here as defaults; if you want to deploy a custom
> scorer, see [`docs/scoring/quality.md`](../scoring/quality.md) for
> the philosophy and the supported route (subclass / pair with
> `ReliabilityScorer` or `MLScorer`).

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

The hand-rating + calibration scaffolding (`scripts/scoring/sample_for_rating.py`,
`rate.py`, `calibrate.py`, `ratings*.csv`) was built during the v1.0–v1.2
audit. It served that purpose; library users do not need to run it.

## SCORER_VERSION 1.0.0 → 1.1.0 (2026-05-05)

**Change.** `score_trendline` now reads `trendline_fit_r2(prices, line)` —
the R² of the line against the **intermediate bars** between
anchors — rather than the anchor-only `TrendLine::r_squared`.
See `crates/fundcloud-core/src/patterns/trendline.rs`.

Each line is evaluated against the maximum of `close`, `high`, and
`low` to auto-select the line's natural price series (a low-anchored
neckline naturally fits lows; a high-anchored resistance line fits
highs).

**Why.** Under v1.0, `trendline_r2` was near-constant at **0.98 ± 0.08**
across all 60 detections — trendlines forced through anchor pivots
always fit r² ≈ 1.0. The 25% weight contributed no discriminative
signal. Production weights (30/25/25/20) and uniform (25/25/25/25)
produced identical Spearman.

**Empirical impact** across Mag7 + SPY + QQQ (4999 detections):

- `trendline_r2` distribution shifted from **mean=0.98 std=0.08**
  (near-constant) to **mean=0.06 std=0.14, range [0.00, 0.84]**.
- Quality bands redistributed: from **74 good / 68 marginal / 36
  poor** under v1.0 to **1 good / 91 marginal / 79 poor** under v1.1.
  The earlier "good" tier was largely a free 25 points from the
  constant-1.0 trendline component.

`calibration_interim_n60.json` is preserved as the v1.0 baseline.

## SCORER_VERSION 1.1.0 → 1.2.0 (2026-05-05)

**Change.** `score_completeness` removes the long-formation duration
penalty. Previously `bar_count > 60` decayed 100 → 50 over 60..120 and
floored at 50 past 120. v1.2 saturates at 100 once `bar_count >= 10`.
The 5-bar minimum and 5..10 ramp are unchanged.

**Why.** Quality measures geometric textbookness, not a preferred
timeframe — a textbook 6-month double top is no less clean than a
30-bar one. The previous penalty actively biased the system against
long-window patterns and combined with `min_quality=50` to filter them
out. See `feedback_quality_geometric_only` in the project memory.

**Companion change** in the Python detection layer (not in the scorer):
`PatternIndicator` now defaults to multi-tier pivot scanning
(`pivot_tiers=((3,5,8), (13,21), (34,55))`). With all orders in one
sweep, small-order pivots clutter the alternating sequence and hide
major swings. Running disjoint tiers exposes patterns at three
scales — short, intermediate, multi-month. Empirical impact on AAPL
full history (1980–2026, daily, 11.4k bars):

| min_quality | OLD detections | NEW detections | NEW long-window (>60 bars) | NEW multi-quarter (>120 bars) |
|---|---|---|---|---|
| 50 | 182 | 206 (+13%) | 22 | 8 |
| 30 | 648 | 751 (+16%) | 99 | 53 |

**v1.2 calibration result** (n=69, claude rater): CV Spearman
ρ = 0.335 ± 0.126. Optimizer collapses to the existing 30/25/25/20
production weights, so no weight change. The bottleneck is no longer
weight tuning — the four sub-scores don't capture pattern-name
conformance, which would be the next axis to add if the geometric
scorer ever moves past v1.2.

## Where things go from here

We're explicitly **not** making the calibration loop a library-user
deliverable. The library ships:

- The `GeometricScorer` with the defaults committed at v1.2.0.
- All knobs that survived the audit, exposed as constructor kwargs (see
  [`docs/guides/patterns/knobs.md`](../guides/patterns/knobs.md)).
- The `quality` column in the events table — one input among many for
  user strategies.

Outcome-based confidence (does this detection actually work in your
universe?) is the user's responsibility. Pair `quality` with your own
forward-return analysis, ML model, or `ReliabilityScorer` /
`MLScorer` extension.
