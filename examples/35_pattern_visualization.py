"""35 — Visualize detected chart patterns on candlestick charts.

Produces a gallery of HTML files under ``examples/out/charts/`` showing:

1. **Top-quality detections per pattern** — for each of the 9 patterns,
   plot the highest-quality detection on a real asset. Pivots are
   *connected* into the formation shape; ``breakout_ts`` and
   ``breakout_ts + horizon`` are marked so you can see the metric
   grading window. Trend lines and target / stop levels overlaid.
2. **Per-asset, per-pattern overview** — every detection of one pattern
   on one asset, formations drawn, holding window shaded.
3. **All-patterns asset chart** — single chart per asset, every pattern
   drawn with its own colour, click any pattern in the legend to toggle
   its detections on / off. The "what's been happening on AAPL?" view.

Open the resulting HTML files in any browser. Plotly figures are
fully interactive: zoom (drag-select), pan, hover for OHLC values,
and click legend entries to toggle pattern traces.

Run:
    uv run python examples/35_pattern_visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fundcloud  # noqa: F401  — registers the .fc accessor
import pandas as pd
from fundcloud.features.patterns import (
    Pattern,
    PatternCondition,
    apply_condition,
)
from fundcloud.plots.patterns import (
    plot_asset_patterns,
    plot_pattern_event,
    plot_patterns_overview,
)

PARQUET = Path("examples/out/pattern_scan_bars.parquet")
OUT_DIR = Path("examples/out/charts")


def _load_bars() -> pd.DataFrame:
    if not PARQUET.exists():
        sys.stderr.write(
            f"ERROR: {PARQUET} not found. Regenerate with:\n"
            "    uv run python examples/32_pattern_scan_real_data.py\n"
        )
        sys.exit(1)
    return pd.read_parquet(PARQUET)


def _top_event_per_pattern(bars: pd.DataFrame) -> dict[Pattern, pd.Series | None]:
    """For each pattern, return the highest-quality detection across the
    entire universe (with target / stop levels filled by apply_condition).
    """
    out: dict[Pattern, pd.Series | None] = {}
    for p in Pattern:
        events = bars.fc.pattern_events(p, min_quality=0.0)
        if events.empty:
            out[p] = None
            continue
        events = apply_condition(events, PatternCondition(), bars)
        # Drop events whose targets / stops failed to fill (e.g., pivot
        # geometry too thin for measured-move height).
        events = events.dropna(subset=["target_price", "stop_price"])
        if events.empty:
            out[p] = None
            continue
        out[p] = events.sort_values("quality", ascending=False).iloc[0]
    return out


def _save_top_per_pattern(bars: pd.DataFrame, top: dict[Pattern, pd.Series | None]) -> None:
    print("\n=== Top-quality detection per pattern ===")
    for p, ev in top.items():
        if ev is None:
            print(f"  {p.value:<28} (no qualifying events)")
            continue
        fig = plot_pattern_event(ev, bars, padding=20)
        path = OUT_DIR / f"{p.value}.html"
        fig.write_html(path)
        ts = pd.Timestamp(ev["breakout_ts"]).strftime("%Y-%m-%d")
        print(f"  {p.value:<28} {ev['asset']:<6} {ts}  Q={ev['quality']:.0f}  → {path}")


def _save_per_asset_overview(bars: pd.DataFrame) -> None:
    """One overview chart per (asset, pattern) cell that earned tradeable
    status in example 34. Hand-picked subset to keep the output small.
    """
    picks = [
        ("META", Pattern.TRIPLE_TOP),
        ("NVDA", Pattern.TRIPLE_TOP),
        ("SPY", Pattern.TRIPLE_BOTTOM),
        ("AAPL", Pattern.SYMMETRICAL_TRIANGLE),
        ("AMZN", Pattern.DOUBLE_TOP),
    ]
    print("\n=== Per-asset overview charts (each shows every detection) ===")
    for asset, pattern in picks:
        events = bars.fc.pattern_events(pattern, min_quality=73.0)
        events = events[events["asset"] == asset]
        if events.empty:
            print(f"  {asset:<6} {pattern.value:<28} (no events)")
            continue
        fig = plot_patterns_overview(events, bars, asset=asset)
        path = OUT_DIR / f"overview_{asset}_{pattern.value}.html"
        fig.write_html(path)
        print(f"  {asset:<6} {pattern.value:<28} {len(events):>3} events → {path}")


def _save_all_patterns_per_asset(bars: pd.DataFrame) -> None:
    """One chart per asset showing every pattern's detections; legend
    toggles per pattern. The "what's been happening on AAPL?" view.
    """
    print("\n=== All-patterns single-asset charts ===")
    assets = sorted(bars.columns.get_level_values(-1).unique())
    for asset in assets:
        # No horizon shading on the multi-pattern overview — too noisy
        # with hundreds of events. Per-event detail (with horizon) is
        # available on plot_pattern_event for individual investigation.
        fig = plot_asset_patterns(bars, asset, min_quality=73.0)
        path = OUT_DIR / f"all_patterns_{asset}.html"
        fig.write_html(path)
        # Count formation traces (one per event); subtract 1 for candlestick.
        n_events = sum(1 for t in fig.data if t.type == "scatter" and "lines" in (t.mode or ""))
        print(f"  {asset:<6} {n_events:>4} events drawn → {path}")


def main() -> None:
    bars = _load_bars()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Loaded {bars.shape[0]:,} bars × "
        f"{bars.columns.get_level_values(-1).nunique()} assets ("
        f"{bars.index.min().strftime('%Y-%m-%d')} → {bars.index.max().strftime('%Y-%m-%d')})"
    )

    top = _top_event_per_pattern(bars)
    _save_top_per_pattern(bars, top)
    _save_per_asset_overview(bars)
    _save_all_patterns_per_asset(bars)

    n_files = len(list(OUT_DIR.glob("*.html")))
    print(f"\nWrote {n_files} HTML files to {OUT_DIR}/")
    print("Open any of them in a browser — they're fully interactive (zoom, hover, etc.).")
    print("Tip: open all_patterns_<asset>.html and toggle patterns via the legend.")


if __name__ == "__main__":
    main()
