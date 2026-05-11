"""36 — Full chart-pattern flow, end-to-end on equities + crypto.

The post-FLG-1015 headline demo. Walks the four-step pipeline in order:

1. ``scan_all_patterns(bars)`` — runs every registered detector at once
   (the TA-Lib-style fan-out — registry + uniform call).
2. ``feature_quality.per_pattern(events, bars)`` — ranks patterns by
   realised quality (hit rate, expectancy, MFE/MAE).
3. ``pattern_direction.direction_map_from_outcomes(events, bars)`` —
   infers per-pattern direction empirically. Detection is geometry-
   only; whether a "double top" is bearish on *your* assets is an
   empirical answer, not a textbook prior.
4. ``PatternStrategy(..., direction_map=dmap)`` — backtests with the
   empirical direction map driving signed entries.

Universe: equities (SPY, QQQ, Mag7) + crypto (BTC, ETH, SOL).

Prereqs:
    uv run python examples/32_pattern_scan_real_data.py

Run:
    uv run python examples/36_full_flow_demo.py
    uv run python examples/36_full_flow_demo.py --crypto-only
    uv run python examples/36_full_flow_demo.py --backtest head_and_shoulders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fundcloud
import pandas as pd
from fundcloud.features.indicators.base import _REGISTRY
from fundcloud.features.patterns import Pattern, scan_all_patterns
from fundcloud.metrics import feature_quality as fq
from fundcloud.metrics import pattern_direction as pd_
from fundcloud.strategies import PatternStrategy

PARQUET = Path("examples/out/pattern_scan_bars.parquet")
HORIZON = 20
MIN_QUALITY = 0.0
MIN_SAMPLES = 30


def _load_bars(*, crypto_only: bool, equities_only: bool) -> pd.DataFrame:
    if not PARQUET.exists():
        sys.stderr.write(
            f"ERROR: {PARQUET} not found. Regenerate with:\n"
            "    uv run python examples/32_pattern_scan_real_data.py\n"
        )
        sys.exit(1)
    bars = pd.read_parquet(PARQUET)
    crypto_assets = {"BTC-USD", "ETH-USD", "SOL-USD"}
    asset_levels = bars.columns.get_level_values("asset")
    if crypto_only:
        keep = [a for a in asset_levels if a in crypto_assets]
    elif equities_only:
        keep = [a for a in asset_levels if a not in crypto_assets]
    else:
        keep = sorted(set(asset_levels))
    if not keep:
        sys.stderr.write(
            "ERROR: requested universe is empty in the cached parquet.\n"
            "Regenerate with the matching --crypto-only / --equities-only flag on example 32.\n"
        )
        sys.exit(1)
    cols = [c for c in bars.columns if c[1] in keep]
    return bars.loc[:, cols].sort_index(axis=1)


# ----------------------------------------------------------------------- main


def _print_section(n: int, title: str) -> None:
    print(f"\n{'═' * 8} {n}. {title} {'═' * 8}")


def _format_direction_map(dmap: dict[str, fundcloud.features.patterns.Direction]) -> str:
    if not dmap:
        return "  (no patterns reached min_samples — every pattern would fall back to default)"
    lines = []
    for pat in sorted(dmap.keys()):
        lines.append(f"  {pat:<28} → {dmap[pat].value}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    universe = ap.add_mutually_exclusive_group()
    universe.add_argument("--crypto-only", action="store_true")
    universe.add_argument("--equities-only", action="store_true")
    ap.add_argument(
        "--backtest",
        default=Pattern.HEAD_AND_SHOULDERS.value,
        help=f"pattern to backtest in step 4 (default: {Pattern.HEAD_AND_SHOULDERS.value})",
    )
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES,
        help=(
            "per-pattern event floor for direction-map confidence; below this, "
            "the pattern falls back to the default direction"
        ),
    )
    ap.add_argument("--min-quality", type=float, default=MIN_QUALITY)
    args = ap.parse_args()

    # ------------------------------------------------------------------- 0. data
    print("═" * 8, "0. Universe + bars", "═" * 8)
    bars = _load_bars(crypto_only=args.crypto_only, equities_only=args.equities_only)
    assets = sorted(set(bars.columns.get_level_values("asset")))
    print(f"  assets        : {assets}")
    print(f"  bars frame    : {bars.shape}  index={bars.index[0].date()} → {bars.index[-1].date()}")

    # --------------------------------------------------- 1. scan_all_patterns
    _print_section(1, "scan_all_patterns(bars) — every registered detector at once")
    events = scan_all_patterns(bars)
    if events.empty:
        print("  (no events detected; try lowering --min-quality on the upstream scan)")
        return
    # Apply the same min-quality cutoff every TA-Lib-equivalent flow uses.
    events = events[events["quality"] >= args.min_quality].reset_index(drop=True)
    print(
        f"  {len(events)} events across {events['pattern'].nunique()} patterns × "
        f"{events['asset'].nunique()} assets"
    )
    print("\n  Counts per (pattern, asset class):")
    is_crypto = events["asset"].isin({"BTC-USD", "ETH-USD", "SOL-USD"})
    counts = (
        events
        .assign(asset_class=lambda d: ["crypto" if c else "equity" for c in is_crypto])
        .groupby(["pattern", "asset_class"])
        .size()
        .unstack(fill_value=0)
    )
    if isinstance(counts.index[0], Pattern):  # type: ignore[index]
        counts.index = counts.index.map(lambda p: p.value)
    print(counts.to_string())

    # ---------------------------------------------------------- 2. per_pattern
    _print_section(2, "per_pattern(...) — ranking by realised quality")
    ranking = fq.per_pattern(events, bars, horizon=args.horizon, trade_direction="long")
    cols = [
        "n_events",
        "hit_rate",
        "baseline_hit",
        "expectancy",
        "edge_ratio",
        "mfe_atr",
        "mae_atr",
    ]
    print(f"  Treating every pattern as long for the ranking (horizon={args.horizon}):\n")
    print(ranking[cols].round(3).sort_values("expectancy", ascending=False).to_string())

    # ------------------------------------------------- 3. direction_map_from_outcomes
    _print_section(3, "direction_map_from_outcomes(...) — empirical direction")
    dmap = pd_.direction_map_from_outcomes(
        events,
        bars,
        horizon=args.horizon,
        min_samples=args.min_samples,
    )
    print(f"  Per-pattern direction (horizon={args.horizon}, min_samples={args.min_samples}):\n")
    print(_format_direction_map(dmap))
    if dmap:
        means = pd_.mean_forward_returns(events, bars, horizon=args.horizon)
        print(f"\n  Mean forward returns (h={args.horizon}) — the underlying statistic:")
        for pat in sorted(means):
            stats = means[pat]
            print(f"  {pat:<28} count={stats['count']:>4}  mean_fwd_ret={stats['mean']:+.4f}")

    # --------------------------------------------------------- 4. backtest
    _print_section(4, f"PatternStrategy({args.backtest}, direction_map=dmap)")
    if args.backtest not in _REGISTRY:
        print(
            f"  ✗ pattern {args.backtest!r} is not registered. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
        return
    indicator_cls = _REGISTRY[args.backtest]
    indicator = indicator_cls(min_quality=args.min_quality)
    strat = PatternStrategy(indicator, direction_map=dmap, size=0.1)
    result = bars.fc.run_strategy(strat)
    summary = result.summary()
    print(summary.round(4).to_string())
    print(f"\n  trades placed : {len(result.trades)}")
    if dmap and args.backtest in dmap:
        print(
            f"  empirical direction for {args.backtest}: "
            f"{dmap[args.backtest].value}  "
            "(strategy applies it; long-only execution skips short-resolved events)"
        )
    elif dmap:
        print(
            f"  {args.backtest} not in dmap → strategy falls back to "
            "Direction.BULLISH default (assume long)"
        )


if __name__ == "__main__":
    main()
