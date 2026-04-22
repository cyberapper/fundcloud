"""22 — Know your data: native profile / compare / quickview.

Before you fit a model or size a strategy, look at the data. Fundcloud's
``explore`` module gives you three layers, all ship in core (no ``[eda]``
extra needed):

* :func:`fundcloud.explore.quickview` — one-row-per-column scannable
  summary, handy inside a notebook cell or a CLI.
* :func:`fundcloud.explore.profile` — full HTML report: overview,
  per-column stats, plotly histograms, Pearson + Spearman correlation
  heatmap, missing-pattern panel, and rule-based alerts for things like
  zero variance, high correlation, high skew, excessive missing.
* :func:`fundcloud.explore.compare` — two-dataset drift report: KS +
  Wasserstein per column, overlay histograms, correlation delta, alerts
  for schema changes and distribution shifts.

The trader use case: 'split my return panel into train (pre-2024) and
holdout (2024→), check that the holdout hasn't drifted, then decide if the
fit I'm about to run is safe to run at all.'

Run:
    uv add 'fundcloud[data-yf]'
    uv run python examples/22_eda_native.py
"""

from __future__ import annotations

from pathlib import Path

from _data import pull_closes
from fundcloud.explore import compare, describe, profile

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    closes = pull_closes(
        {
            "US_EQ": "SPY",
            "EU_EQ": "VGK",
            "EM_EQ": "VWO",
            "TECH": "QQQ",
            "BONDS_AGG": "AGG",
            "GOLD": "GLD",
        },
        years=5,
    )
    if closes is None or closes.empty:
        return 1
    returns = closes.pct_change().dropna()
    split = returns.index[int(len(returns) * 0.7)]
    train = returns.loc[:split]
    holdout = returns.loc[split:]
    print(
        f"Full panel:   {returns.shape}  ({returns.index[0].date()} → {returns.index[-1].date()})"
    )
    print(f"Train:        {train.shape}  (up to {split.date()})")
    print(f"Holdout:      {holdout.shape}  (from {split.date()})")

    # ----------------------- describe: scannable table ----------------------
    # describe() is a super-set of pandas' describe — same columns plus our
    # finance extras (sharpe / cagr / vol / max_drawdown when the index is a
    # DatetimeIndex).
    print("\n--- describe (console — super-set of pandas) ---")
    print(describe(returns).round(4).to_string())

    # ----------------------- profile: Python-first report -------------------
    # profile() returns a ProfileReport object you can interrogate at the
    # REPL (.stats / .alerts / .correlations / .missing) and render to HTML
    # with .to_html(path).
    profile_path = OUT / "22_profile.html"
    report = profile(returns, output=profile_path, title="Returns panel — 5-year profile")
    print(f"\nprofile returned: {type(report).__name__}")
    print(f"  .alerts:         {len(report.alerts)}")
    print(f"  .stats shape:    {report.stats.shape}")
    print(
        f"  HTML written:    {profile_path.relative_to(HERE.parent)}  "
        f"({profile_path.stat().st_size / 1024:.1f} KB)"
    )

    # ----------------------- compare: train vs holdout drift ----------------
    compare_path = OUT / "22_compare.html"
    compare(
        train,
        holdout,
        output=compare_path,
        names=("train", "holdout"),
        title="Train vs holdout drift",
    )
    print(
        f"compare HTML:   {compare_path.relative_to(HERE.parent)}  "
        f"({compare_path.stat().st_size / 1024:.1f} KB)"
    )

    # ----------------------- compare with target ----------------------------
    # Use US_EQ as a synthetic "target"; the rest of the columns are features.
    compare_target_path = OUT / "22_compare_target.html"
    compare(
        train,
        holdout,
        output=compare_target_path,
        names=("train", "holdout"),
        target="US_EQ",
        title="Train vs holdout — feature correlations against US_EQ",
    )
    print(
        f"compare+target: {compare_target_path.relative_to(HERE.parent)}  "
        f"({compare_target_path.stat().st_size / 1024:.1f} KB)"
    )

    print("\nHow to read it:")
    print("  * describe() answers 'what dtypes, how many missing, what's the")
    print("    mean/std/median?' — it's a super-set of pandas' describe plus")
    print("    our finance extras (sharpe, cagr, volatility, max_drawdown).")
    print("  * profile's alerts flag zero-variance columns, high-correlation")
    print("    pairs, heavy-tailed features (|skew| > 2, |kurt| > 7), and")
    print("    excessive missingness — the issues that silently kill models.")
    print("  * compare's KS + Wasserstein drift table tells you whether the")
    print("    holdout actually looks like the training set. If KS > 0.2 on a")
    print("    feature you rely on, that feature has drifted and you need to")
    print("    either retrain, drop it, or gate the prediction.")
    print("  * Passing a target=... column adds a correlation-shift table:")
    print("    features whose correlation with the target has moved are your")
    print("    biggest candidates for reduced out-of-sample Sharpe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
