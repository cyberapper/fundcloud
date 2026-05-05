# Quality scorer calibration workflow

End-to-end recipe for turning the geometric quality scorer's inherited
weights into a calibrated, defensible measurement instrument. The
output is a single number — Spearman ρ between scorer output and
hand-rated textbook quality on a held-out sample — that gets recorded
in [`docs/scoring/quality.md#calibration-record`](../../docs/scoring/quality.md#calibration-record).

## Prerequisites

- Bars cached at `examples/out/pattern_scan_bars.parquet`. Generate it
  by running `examples/32_pattern_scan_real_data.py` if missing.
- Working `fundcloud._core` (run `unset CONDA_PREFIX && uv run maturin
  develop --release` if the binding is stale).

## Workflow

```
┌────────────────────────────────────┐
│ 1. Sample detections for rating    │   sample_for_rating.py
│    Stratified by pattern × quality │   → rating_set.parquet
└─────────────────┬──────────────────┘   → charts/ (one HTML per detection)
                  │
                  ▼
┌────────────────────────────────────┐
│ 2. Hand-rate                       │   rate.py
│    Interactive CLI; resumable      │   → ratings.csv
└─────────────────┬──────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ 3. Calibrate weights               │   calibrate.py
│    Fit weights → maximise          │   → calibration_<date>.json
│    Spearman ρ on ratings           │   prints: ρ, 95% CI, weights
└────────────────────────────────────┘
```

## 1. Sample

```bash
uv run python scripts/scoring/sample_for_rating.py \
    --bars examples/out/pattern_scan_bars.parquet \
    --out scripts/scoring/rating_set.parquet \
    --charts-dir scripts/scoring/charts \
    --n 200 \
    --min-quality 0
```

Pulls events across every pattern, stratifies by (pattern × current
quality band) so the rating set spans the full quality distribution,
then renders one interactive HTML chart per detection.

## 2. Rate

```bash
uv run python scripts/scoring/rate.py \
    --rating-set scripts/scoring/rating_set.parquet \
    --charts-dir scripts/scoring/charts \
    --ratings scripts/scoring/ratings.csv \
    --rater-id peter
```

Interactive CLI:
- Opens each chart in the default browser.
- Prompts for a 0–100 rating (or `s` to skip, `q` to quit).
- Optional one-line note (constraint, suspicion, special case).
- Resumable — already-rated detections are skipped on rerun.
- Crashed-process safe — every rating is fsynced immediately.

The CLI **does not show the current scorer output**. Rating is
performed against your judgment of "how textbook does this look",
independent of how the v1.0.0 scorer happened to rate it.

## 3. Calibrate

```bash
uv run python scripts/scoring/calibrate.py \
    --ratings scripts/scoring/ratings.csv \
    --rating-set scripts/scoring/rating_set.parquet \
    --out scripts/scoring/calibration_$(date +%Y-%m-%d).json
```

Fits non-negative weights summing to 1 that maximise Spearman ρ
between scorer output and hand-rated grade. Reports:

- Mean ρ ± std across 5-fold cross-validation.
- 95% CI on each weight via 1000-bootstrap.
- Comparison: current weights vs fitted weights vs uniform weights.
- Sample size, scorer version, calibration date.

Once you're satisfied, copy the row of fitted weights into
`crates/fundcloud-core/src/patterns/scoring.rs`, bump
`SCORER_VERSION`, and add a row to the calibration record table in
`docs/scoring/quality.md`.

## Discipline

- Never rate against future returns — if you find yourself thinking
  "this one worked, so I'll rate it higher", you've crossed back into
  outcome-tuning territory. See `docs/scoring/quality.md#anti-patterns`.
- Get a second rater if the calibration is going into production.
  Spearman ρ between two raters is itself an upper bound on what the
  scorer can achieve; below ρ ≈ 0.7 inter-rater, "textbook quality" is
  not crisp enough to deploy as a real-money input.
- Hold out a regime. Rate detections from one date range; calibrate on
  it; verify ρ holds on a held-out date range. Drift between the two
  is direct evidence that "textbook" is not regime-invariant for your
  rater — adjust your rating discipline before tuning weights further.
