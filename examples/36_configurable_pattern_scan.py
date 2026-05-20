"""36 — Configurable pattern scan: tiered pivots + per-detector knobs.

What this script demonstrates:

1. **Tiered pivot scanning** (`pivot_tiers`). Disjoint pivot scales
   surface short, intermediate, and multi-month formations in one call.
   Compare to the single-tier scan and see how many long-window
   patterns become reachable.

2. **Per-detector knobs.** Every detector now accepts its sensitivity
   thresholds as constructor kwargs. Compare default DoubleTop with a
   "loose" variant tuned for noisy assets and a "strict" variant for
   conservative signal generation. See ``docs/guides/patterns/knobs.md``
   for the full inventory.

3. **Quality threshold as a tradable knob.** Sweep ``min_quality``
   against detection count and downstream feature-quality (hit rate,
   expectancy at horizon=20). Pick the cutoff that earns its keep.

The intent is to give a library user enough working code to ask
themselves "what's the right configuration for *my* universe?" — not to
prescribe defaults.

Run:
    uv run python examples/36_configurable_pattern_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fundcloud  # noqa: F401  — registers the .fc accessor
import pandas as pd
from fundcloud.features.patterns import DoubleTop, HeadAndShoulders, Pattern

PARQUET = Path("examples/out/pattern_scan_bars.parquet")


def _load_bars() -> pd.DataFrame:
    if not PARQUET.exists():
        sys.stderr.write(
            f"ERROR: {PARQUET} not found. Regenerate with:\n"
            "    uv run python examples/32_pattern_scan_real_data.py\n"
        )
        sys.exit(1)
    return pd.read_parquet(PARQUET)


def _summarize(events: pd.DataFrame) -> dict[str, float]:
    if events.empty:
        return {"n": 0, "long_n": 0, "very_long_n": 0, "max_bars": 0, "mean_q": 0.0}
    bars_len = (events["formation_end"] - events["formation_start"]).dt.days.astype(int)
    return {
        "n": len(events),
        "long_n": int((bars_len > 60).sum()),
        "very_long_n": int((bars_len > 120).sum()),
        "max_bars": int(bars_len.max()),
        "mean_q": round(float(events["quality"].mean()), 1),
    }


def _print_row(label: str, summary: dict[str, float]) -> None:
    print(
        f"  {label:<30} n={summary['n']:>4}  >60bars={summary['long_n']:>3}  "
        f">120bars={summary['very_long_n']:>3}  max_bars={summary['max_bars']:>4}  "
        f"mean_quality={summary['mean_q']}"
    )


def section_1_tiered_vs_single(bars: pd.DataFrame) -> None:
    print("\n=== 1. Tiered pivot scanning — single-tier (legacy) vs default ===")
    print("Same detector, same data, different `pivot_tiers` setting.\n")

    single_tier = DoubleTop(min_quality=75, pivot_tiers=()).events(bars)
    multi_tier = DoubleTop(min_quality=75).events(bars)
    long_only = DoubleTop(min_quality=75, pivot_tiers=((34, 55),)).events(bars)

    print("DoubleTop, min_quality=75:")
    _print_row("single-tier (3,5,8)", _summarize(single_tier))
    _print_row("default tiered", _summarize(multi_tier))
    _print_row("large-tier only (34, 55)", _summarize(long_only))

    print(
        "\n  Tiered scanning surfaces formations that the small-order pivots\n"
        "  hide — large swings get fragmented unless we run a separate\n"
        "  scan with only the larger orders."
    )


def section_2_per_detector_knobs(bars: pd.DataFrame) -> None:
    print("\n=== 2. Per-detector knobs — strict vs default vs loose ===")
    print("DoubleTop with peak_tolerance + min_trough_depth swept three ways.\n")

    strict = DoubleTop(min_quality=75, peak_tolerance=0.005, min_trough_depth=0.05)
    default = DoubleTop(min_quality=75)
    loose = DoubleTop(min_quality=75, peak_tolerance=0.03, min_trough_depth=0.015)

    print("DoubleTop:")
    _print_row("strict (0.5% peaks / 5% trough)", _summarize(strict.events(bars)))
    _print_row("default (1.5% / 3%)", _summarize(default.events(bars)))
    _print_row("loose (3% / 1.5%)", _summarize(loose.events(bars)))

    print("\n=== Same exercise on Head & Shoulders ===\n")
    hs_strict = HeadAndShoulders(min_quality=73, shoulder_tolerance=0.05, min_head_prominence=0.05)
    hs_default = HeadAndShoulders(min_quality=73)
    hs_long_prior = HeadAndShoulders(min_quality=73, prior_trend_window=30)

    print("HeadAndShoulders:")
    _print_row("strict shoulders/head", _summarize(hs_strict.events(bars)))
    _print_row("default", _summarize(hs_default.events(bars)))
    _print_row("longer prior_trend_window=30", _summarize(hs_long_prior.events(bars)))


def section_3_quality_sweep(bars: pd.DataFrame) -> None:
    print("\n=== 3. min_quality sweep — count vs downstream feature quality ===")
    print(
        "What does raising the quality cutoff buy you? "
        "Hit rate / expectancy at 20-bar horizon, n_events alongside.\n"
    )

    cutoffs = [0, 30, 50, 70]
    rows = []
    for q in cutoffs:
        events = bars.fc.pattern_events(Pattern.DOUBLE_BOTTOM, min_quality=float(q))
        if events.empty:
            rows.append({"min_q": q, "n": 0, "hit_rate": float("nan"), "expectancy": float("nan")})
            continue
        # evaluate_pattern wraps fq.evaluate; we replicate the panel for one cutoff.
        panel = bars.fc.evaluate_pattern(
            Pattern.DOUBLE_BOTTOM, horizons=(20,), min_quality=float(q)
        )
        rows.append({
            "min_q": q,
            "n": int(panel.loc[20, "n_events"]),
            "hit_rate": float(panel.loc[20, "hit_rate"]),
            "baseline_hit": float(panel.loc[20, "baseline_hit"]),
            "expectancy": float(panel.loc[20, "expectancy"]),
        })

    df = pd.DataFrame(rows).set_index("min_q").round(3)
    print(df.to_string())

    print(
        "\n  Look for the cutoff where (hit_rate − baseline_hit) is widest and\n"
        "  n_events is still meaningful. That's the production min_quality\n"
        "  for this pattern on this universe."
    )


def main() -> None:
    bars = _load_bars()
    print(f"Loaded {bars.shape[0]:,} bars × {bars.columns.get_level_values(-1).nunique()} assets")

    section_1_tiered_vs_single(bars)
    section_2_per_detector_knobs(bars)
    section_3_quality_sweep(bars)

    print(
        "\nDone. Knob reference: docs/guides/patterns/knobs.md\n"
        "Strategy walkthrough: examples/33_pattern_strategy_backtest.py"
    )


if __name__ == "__main__":
    main()
