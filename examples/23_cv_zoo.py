"""23 — Cross-validation zoo for time-series research.

IID k-fold leaks future into past. For financial panels you want one of:

* :class:`fundcloud.validate.PurgedKFold` — k-fold with a purge buffer
  around each test block. Covered in example 06; included here for a tidy
  side-by-side.
* :class:`fundcloud.validate.EmbargoedKFold` — purge *plus* a forward-
  looking embargo that quarantines the first N rows after each test fold.
  Prevents label leakage when labels span multiple bars.
* :class:`fundcloud.validate.WalkForward` (re-exported from skfolio) —
  rolling or expanding train / test windows that never see the future.
* :class:`fundcloud.validate.CombinatorialPurgedCV` (re-exported from
  skfolio) — *k choose m* combinatorial backtesting. Reports distribution
  of OOS Sharpes, not one point estimate; the best answer we have to 'is my
  Sharpe real or a lucky draw from the fold lottery?'.

Each splitter is a drop-in :class:`sklearn.model_selection.BaseCrossValidator`,
so the boilerplate is identical across all four.

Run:
    uv add 'fundcloud[pf,data-yf]'
    uv run python examples/23_cv_zoo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _data import pull_closes
from fundcloud.validate import EmbargoedKFold, PurgedKFold

HERE = Path(__file__).parent


def _flatten(indices: object) -> np.ndarray:
    # skfolio's CombinatorialPurgedCV yields a list of disjoint test blocks
    # per iteration. PurgedKFold / EmbargoedKFold / WalkForward yield a flat
    # array. Normalise to a single 1-D array.
    if isinstance(indices, list):
        return np.concatenate(indices) if indices else np.empty(0, dtype=int)
    return np.asarray(indices)


def _split_stats(splitter: object, n: int) -> dict:
    idx = np.arange(n)
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    first_leak_row = None
    for tr, te in splitter.split(idx):  # type: ignore[attr-defined]
        tr_arr = _flatten(tr)
        te_arr = _flatten(te)
        train_sizes.append(int(tr_arr.size))
        test_sizes.append(int(te_arr.size))
        # Check if any training row sits immediately after a test row (no gap).
        if first_leak_row is None and tr_arr.size and te_arr.size:
            tr_set = set(tr_arr.tolist())
            for t in te_arr.tolist():
                if (t + 1) in tr_set:
                    first_leak_row = int(t)
                    break
    return {
        "n_splits": len(train_sizes),
        "train_size": int(np.mean(train_sizes)) if train_sizes else 0,
        "test_size": int(np.mean(test_sizes)) if test_sizes else 0,
        "first_train_row_abutting_test": first_leak_row,
    }


def main() -> int:
    try:
        from fundcloud.validate import CombinatorialPurgedCV, WalkForward
    except (ImportError, AttributeError):
        print("This example requires skfolio — `uv add 'fundcloud[pf]'`", file=sys.stderr)
        return 1

    closes = pull_closes({"SPY": "SPY"}, years=5)
    if closes is None or closes.empty:
        return 1
    returns = closes["SPY"].pct_change().dropna()
    n = len(returns)
    print(
        f"Sample size:  {n} trading days ({returns.index[0].date()} → {returns.index[-1].date()})\n"
    )

    splitters: dict[str, object] = {
        "PurgedKFold(k=5, purge=5)": PurgedKFold(n_splits=5, purge=5),
        "EmbargoedKFold(k=5, purge=5, embargo=3)": EmbargoedKFold(n_splits=5, purge=5, embargo=3),
        "WalkForward(train=252, test=21)": WalkForward(train_size=252, test_size=21, purged_size=1),
        "CombinatorialPurgedCV(10, m=8, purge=5)": CombinatorialPurgedCV(
            n_folds=10, n_test_folds=8, purged_size=5
        ),
    }

    print(f"{'splitter':<45} {'n_splits':>8}  {'train':>7}  {'test':>7}  {'abutting':>9}")
    print("-" * 84)
    for label, spl in splitters.items():
        info = _split_stats(spl, n)
        abut = (
            "none"
            if info["first_train_row_abutting_test"] is None
            else str(info["first_train_row_abutting_test"])
        )
        print(
            f"{label:<45} {info['n_splits']:>8}  {info['train_size']:>7}  "
            f"{info['test_size']:>7}  {abut:>9}"
        )

    # Fit & score a trivial predictor (historical mean) under each splitter;
    # report OOS R² distribution across folds.
    from sklearn.metrics import mean_squared_error

    print("\nOut-of-sample baseline (historical-mean predictor):")
    y = returns.to_numpy()
    for label, spl in splitters.items():
        errs = []
        for tr, te in spl.split(np.arange(len(y))):  # type: ignore[attr-defined]
            tr_arr = _flatten(tr)
            te_arr = _flatten(te)
            if tr_arr.size == 0 or te_arr.size == 0:
                continue
            mu = float(y[tr_arr].mean())
            errs.append(mean_squared_error(y[te_arr], np.full(te_arr.size, mu)))
        mse = np.array(errs)
        if len(mse) == 0:
            print(f"  {label:<45}  no folds produced")
            continue
        rmse_bps = np.sqrt(mse) * 10_000
        print(
            f"  {label:<45}  folds={len(mse):>3}  "
            f"RMSE(bps) mean={rmse_bps.mean():>7.2f}  std={rmse_bps.std():>6.2f}"
        )

    print("\nHow to read it:")
    print("  * 'abutting' = row index where a train row sits immediately after a")
    print("    test row with no gap. PurgedKFold should show a row (purge gates")
    print("    only the OTHER side); Embargoed should show 'none'.")
    print("  * WalkForward slides forward in time — train and test never overlap.")
    print("    It's the most conservative splitter and often shows the widest")
    print("    fold-to-fold RMSE spread.")
    print("  * CombinatorialPurgedCV runs ~choose(10, 8) = 45 folds, giving you")
    print("    a distribution over OOS error — use the std, not the mean, when")
    print("    picking the strategy you trust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
