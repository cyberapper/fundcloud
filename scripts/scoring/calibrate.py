"""Fit geometric-quality weights against hand-rated detections.

Inputs:
  --ratings      ratings.csv produced by rate.py
  --rating-set   rating_set.parquet produced by sample_for_rating.py

What it does:
  1. Joins ratings with the per-component scorer outputs.
  2. Fits non-negative weights summing to 1 that maximise Spearman ρ
     between the weighted composite and the human ratings.
  3. Reports k-fold CV ρ (mean ± std) and per-weight 95% bootstrap CIs.
  4. Compares the fitted weights against the current
     30/25/25/20 production weights and against uniform 25/25/25/25.
  5. Persists the calibration record as JSON.

The calibration objective is **Spearman**, not MSE — we care about
rank order matching, not absolute scale agreement. The rating's
absolute scale is only meaningful within a single rater anyway.

Run:
    uv run python scripts/scoring/calibrate.py \\
        --ratings scripts/scoring/ratings.csv \\
        --rating-set scripts/scoring/rating_set.parquet \\
        --out scripts/scoring/calibration_$(date +%Y-%m-%d).json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

# Production weights at the time of writing — keep in sync with
# `crates/fundcloud-core/src/patterns/scoring.rs::GeometricScorer::score`.
# If you bump SCORER_VERSION you must update these to match.
_PROD_WEIGHTS: dict[str, float] = {
    "symmetry": 0.30,
    "volume": 0.25,
    "trendline_r2": 0.25,
    "completeness": 0.20,
}
_COMPONENTS: tuple[str, ...] = ("symmetry", "volume", "trendline_r2", "completeness")


def _composite(weights: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Weighted-sum composite, expecting weights ∈ R⁴ and components shape (n, 4)."""
    return components @ weights


def _neg_spearman(weights: np.ndarray, components: np.ndarray, ratings: np.ndarray) -> float:
    composite = _composite(weights, components)
    rho, _ = spearmanr(composite, ratings)
    if not np.isfinite(rho):
        return 1.0  # worst possible — degenerate inputs
    return -float(rho)


def _fit_weights(components: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Solve for non-negative weights summing to 1 maximising Spearman ρ.

    Spearman is non-smooth so we use SLSQP with a numerical jacobian.
    We restart from a few seeds (uniform, prod, and four corner one-hot
    weight vectors) to dodge poor local optima — Spearman's discrete
    rank surface has many of them.
    """
    bounds = [(0.0, 1.0)] * len(_COMPONENTS)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    seeds: list[np.ndarray] = [
        np.full(len(_COMPONENTS), 1.0 / len(_COMPONENTS)),
        np.array([_PROD_WEIGHTS[c] for c in _COMPONENTS]),
    ]
    for i in range(len(_COMPONENTS)):
        v = np.zeros(len(_COMPONENTS))
        v[i] = 1.0
        seeds.append(v)

    best_w: np.ndarray | None = None
    best_obj = np.inf
    for seed in seeds:
        res = minimize(
            _neg_spearman,
            seed,
            args=(components, ratings),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 200},
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_w = res.x
    assert best_w is not None
    # Normalize to handle small slack on the equality constraint.
    return best_w / best_w.sum()


def _spearman_rho(weights: np.ndarray, components: np.ndarray, ratings: np.ndarray) -> float:
    rho, _ = spearmanr(_composite(weights, components), ratings)
    return float(rho) if np.isfinite(rho) else float("nan")


def _kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        out.append((train_idx, val_idx))
    return out


def _bootstrap_weights(
    components: np.ndarray, ratings: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(ratings)
    out = np.empty((n_boot, len(_COMPONENTS)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = _fit_weights(components[idx], ratings[idx])
    return out


def _format_weights(weights: dict[str, float]) -> str:
    return "  " + "  ".join(f"{c}={weights[c]:.3f}" for c in _COMPONENTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--rating-set", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rater-id",
        type=str,
        default=None,
        help="If set, restrict to ratings from this rater. Otherwise uses every rater present.",
    )
    args = parser.parse_args()

    if not args.ratings.exists():
        sys.stderr.write(f"ERROR: ratings file {args.ratings} not found\n")
        return 2
    if not args.rating_set.exists():
        sys.stderr.write(f"ERROR: rating set {args.rating_set} not found\n")
        return 2

    ratings = pd.read_csv(args.ratings)
    rating_set = pd.read_parquet(args.rating_set)
    if args.rater_id is not None:
        ratings = ratings[ratings["rater_id"] == args.rater_id]
        if ratings.empty:
            sys.stderr.write(f"ERROR: no ratings from rater_id={args.rater_id!r}\n")
            return 2

    joined = ratings.merge(
        rating_set[["detection_id", *_COMPONENTS]],
        on="detection_id",
        how="inner",
    )
    if joined.empty:
        sys.stderr.write("ERROR: no overlap between ratings and rating set\n")
        return 1

    joined = joined.dropna(subset=[*list(_COMPONENTS), "rating"])
    if len(joined) < 30:
        sys.stderr.write(
            f"ERROR: only {len(joined)} usable ratings; need at least 30 for any \n"
            "meaningful Spearman estimate. Rate more samples first.\n"
        )
        return 1

    components = joined[list(_COMPONENTS)].to_numpy(dtype=np.float64)
    targets = joined["rating"].to_numpy(dtype=np.float64)
    n = len(targets)

    print(f"Calibrating against {n} ratings.")
    print("  Component coverage:")
    for c in _COMPONENTS:
        v = components[:, _COMPONENTS.index(c)]
        print(
            f"    {c:<14} mean={v.mean():.2f}  std={v.std():.2f}  "
            f"non-NaN={(~np.isnan(v)).sum()}/{n}"
        )

    # ---------------------------------------------------------- baselines
    prod_w = np.array([_PROD_WEIGHTS[c] for c in _COMPONENTS])
    uniform_w = np.full(len(_COMPONENTS), 1.0 / len(_COMPONENTS))

    rho_prod = _spearman_rho(prod_w, components, targets)
    rho_uniform = _spearman_rho(uniform_w, components, targets)

    # ---------------------------------------------------------- fit
    fitted_w = _fit_weights(components, targets)
    rho_fitted_train = _spearman_rho(fitted_w, components, targets)

    # ---------------------------------------------------------- k-fold CV
    folds = _kfold_indices(n, args.folds, args.seed)
    cv_rhos: list[float] = []
    for train_idx, val_idx in folds:
        w_fold = _fit_weights(components[train_idx], targets[train_idx])
        cv_rhos.append(_spearman_rho(w_fold, components[val_idx], targets[val_idx]))
    cv_mean = float(np.mean(cv_rhos))
    cv_std = float(np.std(cv_rhos, ddof=1)) if len(cv_rhos) > 1 else 0.0

    # ---------------------------------------------------------- bootstrap CI
    print(f"\nBootstrapping ({args.bootstrap} resamples)…")
    boots = _bootstrap_weights(components, targets, args.bootstrap, args.seed)
    weight_ci = {
        c: (float(np.quantile(boots[:, i], 0.025)), float(np.quantile(boots[:, i], 0.975)))
        for i, c in enumerate(_COMPONENTS)
    }

    # ---------------------------------------------------------- report
    print("\n" + "=" * 78)
    print("Calibration result")
    print("=" * 78)
    print(f"\nSamples used:               {n}")
    print(f"Folds (CV):                  {args.folds}")
    print(f"Bootstrap resamples:         {args.bootstrap}")
    print("\nProduction weights (current scorer):")
    print(_format_weights({c: prod_w[i] for i, c in enumerate(_COMPONENTS)}))
    print(f"  Spearman ρ:                {rho_prod:+.3f}")
    print("\nUniform weights (sanity check):")
    print(_format_weights({c: uniform_w[i] for i, c in enumerate(_COMPONENTS)}))
    print(f"  Spearman ρ:                {rho_uniform:+.3f}")
    print("\nFitted weights:")
    print(_format_weights({c: fitted_w[i] for i, c in enumerate(_COMPONENTS)}))
    print(f"  Spearman ρ (in-sample):    {rho_fitted_train:+.3f}")
    print(f"  Spearman ρ ({args.folds}-fold CV):  {cv_mean:+.3f} ± {cv_std:.3f}")
    print("\n95% CI on each fitted weight (bootstrap):")
    for c in _COMPONENTS:
        lo, hi = weight_ci[c]
        print(f"    {c:<14} [{lo:.3f}, {hi:.3f}]")

    print("\n" + "-" * 78)
    print("Interpretation")
    print("-" * 78)
    delta = cv_mean - rho_prod
    if delta > 0.05:
        verdict = (
            f"Fitted weights beat production by Δρ = {delta:+.3f} (CV). "
            "Worth shipping behind a SCORER_VERSION bump."
        )
    elif abs(delta) <= 0.05:
        verdict = (
            f"Fitted weights are within Δρ = {delta:+.3f} of production. "
            "Not worth a version bump on this calibration alone."
        )
    else:
        verdict = (
            f"Fitted weights underperform production by Δρ = {delta:+.3f} (CV). "
            "Sample size, rater drift, or component noise suspected. Investigate."
        )
    print(f"  {verdict}")
    if cv_mean < 0.5:
        print(
            "  ⚠️  CV ρ < 0.5 — the four sub-scores are not a complete description "
            "of textbook quality. Consider adding sub-scores or splitting symmetry "
            "by pattern family."
        )

    # ---------------------------------------------------------- persist
    record = {
        "calibration_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer_version": str(joined["scorer_version"].iloc[0]),
        "rater_id": args.rater_id or "all",
        "n_samples": int(n),
        "folds": int(args.folds),
        "bootstrap": int(args.bootstrap),
        "weights": {
            "production": {c: float(prod_w[i]) for i, c in enumerate(_COMPONENTS)},
            "uniform": {c: float(uniform_w[i]) for i, c in enumerate(_COMPONENTS)},
            "fitted": {c: float(fitted_w[i]) for i, c in enumerate(_COMPONENTS)},
        },
        "weight_ci_95": {c: list(weight_ci[c]) for c in _COMPONENTS},
        "spearman_rho": {
            "production": float(rho_prod),
            "uniform": float(rho_uniform),
            "fitted_in_sample": float(rho_fitted_train),
            "fitted_cv_mean": float(cv_mean),
            "fitted_cv_std": float(cv_std),
        },
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nWrote calibration record: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
