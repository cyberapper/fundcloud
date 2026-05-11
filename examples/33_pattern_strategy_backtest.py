"""33 — Pattern feature quality + strategy backtest, end-to-end.

What this script demonstrates:

1. Loading real OHLCV bars (the cached parquet from example 32, regenerate
   it via ``uv run python examples/32_pattern_scan_real_data.py`` if missing).
2. Building the headline feature-quality panel via
   ``bars.fc.evaluate_pattern(...)`` — hit rate, expectancy, edge ratio,
   MFE/MAE in ATR, IC, ICIR, baseline comparison.
3. The ``trade_direction='long'`` vs ``'short'`` knob — detection is
   direction-agnostic, so the same events frame can be evaluated either
   way to test the "trade with the textbook" vs "fade the textbook"
   hypotheses, with the baseline transformed in lockstep so the
   comparison stays honest.
4. ``quality_buckets`` — the diagnostic that validates the geometric
   scorer. If Q5 outperforms Q1 monotonically, the scorer is doing
   useful work; flat buckets are a recalibration signal.
5. ``apply_condition`` — fills ``target_price`` / ``stop_price`` per
   ``PatternCondition`` so downstream R-multiples reflect a real exit
   policy.
6. ``bars.fc.run_pattern(...)`` — full backtest via ``PatternStrategy``
   on a configured indicator + condition.
7. ``pattern_direction.direction_map_from_outcomes`` — close the loop
   by inferring the direction empirically and re-running the backtest
   under the data's preferred sign. See example 36 for the full
   four-step flow.

Run:
    uv run python examples/33_pattern_strategy_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fundcloud  # noqa: F401  — registers the .fc accessor
import pandas as pd
from fundcloud.features.patterns import (
    Direction,
    Pattern,
    PatternCondition,
    StopMethod,
    TargetMethod,
    scan_all_patterns,
)
from fundcloud.metrics import feature_quality as fq
from fundcloud.metrics import pattern_direction as pd_

PARQUET = Path("examples/out/pattern_scan_bars.parquet")


def _load_bars() -> pd.DataFrame:
    if not PARQUET.exists():
        sys.stderr.write(
            f"ERROR: {PARQUET} not found. Regenerate with:\n"
            "    uv run python examples/32_pattern_scan_real_data.py\n"
        )
        sys.exit(1)
    return pd.read_parquet(PARQUET)


def main() -> None:
    bars = _load_bars()
    print(f"Loaded {bars.shape[0]:,} bars × {bars.columns.get_level_values(-1).nunique()} assets")

    # ----------------------------------------------- 1. Headline feature panel
    print("\n=== Headline feature-quality panel: DoubleBottom ===")
    panel = bars.fc.evaluate_pattern(Pattern.DOUBLE_BOTTOM, horizons=(5, 10, 20, 60))
    print(
        panel[
            [
                "n_events",
                "hit_rate",
                "baseline_hit",
                "expectancy",
                "edge_ratio",
                "mae_p95_atr",
                "ic",
            ]
        ]
        .round(3)
        .to_string()
    )

    # ----------------------------------------------- 2. Long vs short test
    print("\n=== trade_direction='long' vs 'short' on DoubleTop ===")
    print("(Detection is direction-agnostic. Same events frame, two signs —")
    print(" picks up which one matches the data without committing in advance.)")
    long_panel = bars.fc.evaluate_pattern(
        Pattern.DOUBLE_TOP, horizons=(20, 60), trade_direction="long"
    )
    short_panel = bars.fc.evaluate_pattern(
        Pattern.DOUBLE_TOP, horizons=(20, 60), trade_direction="short"
    )
    pair = pd.concat(
        [
            long_panel[["hit_rate", "baseline_hit", "expectancy"]].add_prefix("long_"),
            short_panel[["hit_rate", "baseline_hit", "expectancy"]].add_prefix("short_"),
        ],
        axis=1,
    )
    print(pair.round(3).to_string())
    edge_long = (long_panel["hit_rate"] - long_panel["baseline_hit"]).round(3)
    edge_short = (short_panel["hit_rate"] - short_panel["baseline_hit"]).round(3)
    print(f"\nedge over baseline — long : {edge_long.to_dict()}")
    print(f"edge over baseline — short: {edge_short.to_dict()}")

    # ----------------------------------------------- 3. Quality buckets
    print("\n=== quality_buckets (h=20): InverseHeadAndShoulders ===")
    events_ihs = bars.fc.pattern_events(Pattern.INVERSE_HEAD_AND_SHOULDERS, min_quality=0.0)
    bk = fq.quality_buckets(events_ihs, bars, horizon=20, n_buckets=5)
    print(bk.round(3).to_string())
    print("\n(monotonic Q1→Q5 expectancy → scorer earns its 30% symmetry weight)")

    # ----------------------------------------------- 4. apply_condition + backtest
    print("\n=== Backtest: DoubleBottom, MEASURED_MOVE target, BELOW_PIVOT stop ===")
    condition = PatternCondition(
        target_method=TargetMethod.MEASURED_MOVE,
        stop_method=StopMethod.BELOW_PIVOT,
        time_stop_bars=40,
    )
    result = bars.fc.run_pattern(
        Pattern.DOUBLE_BOTTOM,
        condition=condition,
        size=0.1,
        min_quality=60,
    )
    print("\n" + result.summary().round(4).to_string())

    print("\n=== Same pattern, evaluated WITH the same condition ===")
    cond_panel = bars.fc.evaluate_pattern(
        Pattern.DOUBLE_BOTTOM,
        horizons=(20, 60),
        condition=condition,
        min_quality=60,
    )
    print(cond_panel[["n_events", "hit_rate", "baseline_hit", "expectancy"]].round(3).to_string())
    print(
        "\n(R-multiples shrink vs the no-condition panel because real stops are"
        "\n further from entry than 1×ATR for confirmed double bottoms.)"
    )

    # --------------------------------------------- 5. Empirical direction map
    print("\n=== Empirical direction map across all registered patterns ===")
    print("(Replaces the textbook prior baked into the old `direction` column.)")
    events = scan_all_patterns(bars)
    dmap = pd_.direction_map_from_outcomes(events, bars, horizon=20, min_samples=30)
    if not dmap:
        print("  (no pattern reached min_samples=30 on this universe)")
    else:
        for pat in sorted(dmap):
            print(f"  {pat:<28} → {dmap[pat].value}")
        print(
            "\nFeed this dict to PatternStrategy(..., direction_map=dmap) to backtest"
            "\neach pattern with its data-implied sign. Full demo: example 36."
        )

    # Default-direction tweak as a sanity-check: bullish-as-default vs
    # bearish-as-default on patterns below the sample floor.
    short_default = pd_.direction_map_from_outcomes(
        events, bars, horizon=20, min_samples=30, default=Direction.BEARISH
    )
    diffs = sorted(p for p in short_default if dmap.get(p) != short_default.get(p))
    if diffs:
        print(
            f"\n  patterns affected by --default Direction.BEARISH (below min_samples): {diffs}"
        )


if __name__ == "__main__":
    main()
