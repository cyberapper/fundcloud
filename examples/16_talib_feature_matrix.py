"""16 — Generate the full TA-Lib feature matrix.

Fundcloud auto-wraps every TA-Lib function (158 of them across 10 groups)
as a sklearn-compatible :class:`~fundcloud.features.IndicatorSpec`. This
example shows the three things a trader or ML practitioner actually wants
to do with that catalogue:

1. **Explore** — list every available indicator by group so you know what
   you have before you reach for it.
2. **Curate** — assemble a small, high-signal :class:`FeaturePipeline`
   that's easy to read and fits in a notebook cell.
3. **Bulk-compute** — run *every* TA-Lib indicator against a real OHLCV
   panel and collect the results into one wide ``DataFrame`` suitable
   for feeding to XGBoost / a NN / a feature-importance analysis.

Run:
    uv add 'fundcloud[ta,data-yf]'
    uv run python examples/16_talib_feature_matrix.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _data import pull_closes

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def _pull_ohlcv() -> pd.DataFrame | None:
    """Use _data.pull_closes's YF underpinnings to grab raw OHLCV bars."""
    try:
        from fundcloud.data import YF
    except ImportError:
        print("This example needs yfinance — `uv add 'fundcloud[data-yf]'`",
              file=sys.stderr)
        return None
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=2)
    try:
        return YF(symbols=["SPY", "QQQ"], interval="1d").read(start=start, end=end)
    except Exception as e:  # noqa: BLE001
        print(f"yfinance request failed: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------- SECTION 1


def catalog_overview() -> None:
    _section("1. What's in the catalogue")
    try:
        from fundcloud.features.indicators import GROUPS, list_indicators
    except ImportError:
        print("TA-Lib not installed — `brew install ta-lib && uv add 'fundcloud[ta]'`")
        return

    total = len(list_indicators())
    print(f"Fundcloud auto-generated {total} TA-Lib indicators across "
          f"{len(GROUPS)} groups:\n")
    for group, names in GROUPS.items():
        sample = ", ".join(names[:5]) + (" …" if len(names) > 5 else "")
        print(f"  {group:<22} {len(names):>3}  ({sample})")


# --------------------------------------------------------------- SECTION 2


def curated_pipeline(bars: pd.DataFrame) -> pd.DataFrame | None:
    _section("2. A curated pipeline — six classics in six lines")
    try:
        from fundcloud.features import FeaturePipeline
        from fundcloud.features.indicators import ATR, BBANDS, EMA, MACD, RSI, SMA
    except ImportError as e:
        print(f"skip: {e}")
        return None

    pipe = FeaturePipeline([
        ("sma_50",    SMA(timeperiod=50)),
        ("ema_20",    EMA(timeperiod=20)),
        ("rsi_14",    RSI(timeperiod=14)),
        ("macd",      MACD()),                    # multi-output (macd/signal/hist)
        ("bbands_20", BBANDS(timeperiod=20)),     # multi-output (upper/mid/lower)
        ("atr_14",    ATR(timeperiod=14)),
    ])
    out = pipe.fit_transform(bars)
    print(f"Pipeline output: {out.shape[0]} rows × {out.shape[1]} columns")
    print(f"Stable from:      {out.dropna().index[0].date()}  "
          f"(after 50-bar SMA warm-up)")
    print(f"Column sample:    {list(out.columns[:6])} …")
    print(f"Pipeline hash:    {pipe.pipeline_hash}   "
          f"(use it as a cache key)")
    return out


# --------------------------------------------------------------- SECTION 3


def full_feature_matrix(bars: pd.DataFrame) -> pd.DataFrame | None:
    _section("3. The full matrix — every TA-Lib indicator")
    try:
        import fundcloud.features.indicators as ind
    except ImportError:
        print("TA-Lib not installed.")
        return None

    names = ind.list_indicators()
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    # TA-Lib emits invalid-input warnings on some instruments; swallow them
    # so the progress log stays readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            cls = getattr(ind, name)
            try:
                inst = cls()                  # use default params
                out = inst.fit_transform(bars)
            except Exception as e:            # noqa: BLE001
                failures[name] = type(e).__name__
                continue
            if out is None or out.empty:
                failures[name] = "empty output"
                continue
            # Prefix each column with the indicator name so the final merge
            # has globally unique column labels.
            out = out.add_prefix(f"{name}__")
            frames[name] = out

    if not frames:
        print("No indicators produced output — stopping.")
        return None

    matrix = pd.concat(frames.values(), axis=1)

    print(f"Bars in:           {bars.shape[0]} rows × {bars.shape[1]} columns")
    print(f"Indicators tried:  {len(names):>4}")
    print(f"Indicators OK:     {len(frames):>4}")
    print(f"Indicators failed: {len(failures):>4}")
    print(f"Final matrix:      {matrix.shape[0]} rows × {matrix.shape[1]} columns")
    print(f"Memory:            {matrix.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Group-level success rates
    by_group = _group_success_rates(list(frames), list(failures), ind.GROUPS)
    print("\nSuccess rate per TA-Lib group:")
    for group, (ok, total) in by_group.items():
        bar = "#" * int(ok / total * 20)
        print(f"  {group:<22} {ok:>3}/{total:>3}  {bar}")

    if failures:
        print("\nSkipped indicators (most common reasons):")
        reasons: dict[str, int] = {}
        for reason in failures.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda r: -r[1])[:5]:
            print(f"  {reason:<30} {count:>3}")

    return matrix


def _group_success_rates(ok_names: list[str], failed_names: list[str],
                         groups: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    ok = set(ok_names)
    tried = set(ok_names) | set(failed_names)
    out: dict[str, tuple[int, int]] = {}
    for group, names in groups.items():
        in_group = [n for n in names if n in tried]
        hits = sum(1 for n in in_group if n in ok)
        out[group] = (hits, len(in_group))
    return out


# --------------------------------------------------------------- SECTION 4


def preview_and_save(matrix: pd.DataFrame) -> None:
    _section("4. Preview + save for downstream use")
    # Non-null percentage per indicator — identifies long-warm-up indicators.
    non_null_pct = (matrix.notna().sum() / len(matrix))
    long_warmup = non_null_pct[non_null_pct < 0.5].sort_values().head(6)
    if len(long_warmup):
        print("Indicators with < 50% non-null (long warm-up or stability issues):")
        for col, pct in long_warmup.items():
            print(f"  {col:<40} {pct * 100:>5.1f}% non-null")
        print()

    # Correlation-pruned sample — show 3 highly-correlated pairs so the
    # reader sees how redundant the raw catalogue is.
    numeric = matrix.dropna(axis=1, how="all").ffill().bfill()
    corr = numeric.corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    flat = corr.unstack().sort_values(ascending=False)
    print("Top 3 highly-correlated indicator pairs (|ρ| closest to 1):")
    seen_pairs: set[frozenset[str]] = set()
    count = 0
    for (a, b), value in flat.items():
        pair = frozenset({a, b})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        print(f"  {a:<30} — {b:<30}  |ρ| = {value:.3f}")
        count += 1
        if count >= 3:
            break

    # Parquet is the right format for a wide feature matrix.
    path = OUT / "16_feature_matrix.parquet"
    matrix.to_parquet(path)
    print(f"\nSaved full matrix to: {path.relative_to(HERE.parent)}  "
          f"({path.stat().st_size / 1e6:.1f} MB on disk)")


def main() -> int:
    bars = _pull_ohlcv()
    if bars is None or bars.empty:
        return 1
    print(f"Live data:  {bars.index[0].date()} → {bars.index[-1].date()}  "
          f"({len(bars)} bars × {len(bars.columns.get_level_values(-1).unique())} assets)")

    catalog_overview()
    curated_pipeline(bars)
    matrix = full_feature_matrix(bars)
    if matrix is not None:
        preview_and_save(matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
