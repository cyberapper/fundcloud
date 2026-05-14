"""34 — Cross-asset / cross-pattern leaderboard.

Sweep all 9 patterns × 2 trade directions (long + short) on all 9
assets in the bundled parquet (QQQ + SPY + AAPL + AMZN + GOOGL + META +
MSFT + NVDA + TSLA — daily bars since the early 1990s). Produces:

1. **Per-asset leaderboard** — for each asset, the top patterns ranked
   by edge-over-baseline at h=20 with expectancy and edge_ratio.
2. **Global tradeable combinations** — every (asset, pattern, direction)
   tuple that meets a "tradeable" bar (n_events ≥ 30, hit_rate beats
   baseline by ≥ 5pp, expectancy > 0), sorted by edge.
3. **Pattern × asset matrix** — wide compact view of edge-over-baseline,
   so you can scan for "where does pattern X work" at a glance.

Run:
    uv run python examples/34_pattern_leaderboard.py

Regenerate the parquet first if missing:
    uv run python examples/32_pattern_scan_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fundcloud  # noqa: F401  — registers the .fc accessor
import pandas as pd
from fundcloud.features.patterns import EVENTS_COLUMNS, Pattern
from fundcloud.metrics import feature_quality as fq

PARQUET = Path("examples/out/pattern_scan_bars.parquet")
HORIZON = 20
MIN_EVENTS = 30
EDGE_THRESHOLD = 0.05  # 5 percentage points
MIN_QUALITY = 50.0


def _load_bars() -> pd.DataFrame:
    if not PARQUET.exists():
        sys.stderr.write(
            f"ERROR: {PARQUET} not found. Regenerate with:\n"
            "    uv run python examples/32_pattern_scan_real_data.py\n"
        )
        sys.exit(1)
    return pd.read_parquet(PARQUET)


def _resolve_indicator(p: Pattern) -> type:
    """Map a Pattern enum to its registered indicator class."""
    from fundcloud.features.indicators.base import _REGISTRY

    return _REGISTRY[p.value]


def _evaluate_one(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    pattern: Pattern,
    asset: str,
    direction: str,
) -> dict | None:
    """Evaluate a single (asset, pattern, direction) cell at the headline
    horizon. Returns None when there are no usable events for the asset.
    """
    asset_events = events[events["asset"] == asset]
    if len(asset_events) < MIN_EVENTS:
        return None
    panel = fq.evaluate(
        asset_events,
        bars,
        horizons=(HORIZON,),
        trade_direction=direction,
    )
    row = panel.iloc[0]
    if row["n_events"] < MIN_EVENTS:
        return None
    edge = float(row["hit_rate"] - row["baseline_hit"])
    return {
        "asset": asset,
        "pattern": pattern.value,
        "direction": direction,
        "n": int(row["n_events"]),
        "hit": float(row["hit_rate"]),
        "base": float(row["baseline_hit"]),
        "edge": edge,
        "expectancy": float(row["expectancy"]),
        "edge_ratio": float(row["edge_ratio"]),
        "ic": float(row["ic"]),
    }


def _is_tradeable(row: dict) -> bool:
    """Tradeable if sample is meaningful, edge beats baseline by ≥ 5pp,
    and expectancy is positive (some realised payoff per trade).
    """
    return row["n"] >= MIN_EVENTS and row["edge"] >= EDGE_THRESHOLD and row["expectancy"] > 0


def _build_leaderboard(bars: pd.DataFrame) -> pd.DataFrame:
    """Run the full sweep, return a long-format DataFrame of every
    non-empty (asset, pattern, direction) cell.
    """
    assets = sorted(bars.columns.get_level_values(-1).unique())
    rows: list[dict] = []
    for pattern in Pattern:
        Cls = _resolve_indicator(pattern)
        events = Cls(min_quality=MIN_QUALITY).events(bars)
        if events.empty:
            continue
        # Make sure every column the metric expects is present, even on
        # patterns whose detector emits nothing for some assets.
        events = events.reindex(columns=list(EVENTS_COLUMNS))
        for asset in assets:
            for direction in ("long", "short"):
                row = _evaluate_one(events, bars, pattern, asset, direction)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "asset",
            "pattern",
            "direction",
            "n",
            "hit",
            "base",
            "edge",
            "expectancy",
            "edge_ratio",
            "ic",
        ],
    )


def _print_per_asset(lb: pd.DataFrame, top_n: int = 5) -> None:
    """For each asset, the top-`top_n` (pattern, direction) by edge."""
    print(f"\n{'=' * 78}")
    print(f"Per-asset leaderboard — top {top_n} patterns by edge over baseline at h={HORIZON}")
    print(f"{'=' * 78}")
    cols = ["pattern", "direction", "n", "hit", "base", "edge", "expectancy", "edge_ratio"]
    for asset, group in lb.groupby("asset"):
        top = group.sort_values("edge", ascending=False).head(top_n)[cols]
        print(f"\n--- {asset} ---")
        print(
            top.to_string(
                index=False,
                formatters={
                    "hit": "{:.3f}".format,
                    "base": "{:.3f}".format,
                    "edge": "{:+.3f}".format,
                    "expectancy": "{:+.3f}".format,
                    "edge_ratio": "{:.2f}".format,
                },
            )
        )


def _print_tradeable(lb: pd.DataFrame) -> None:
    """Global rank of cells meeting the tradeable bar."""
    tradeable = lb[lb.apply(_is_tradeable, axis=1)].copy()
    tradeable = tradeable.sort_values(["edge", "expectancy"], ascending=False)
    print(f"\n{'=' * 78}")
    print(
        f"Tradeable combinations — n ≥ {MIN_EVENTS}, edge ≥ {EDGE_THRESHOLD * 100:.0f}pp, "
        f"expectancy > 0  (h={HORIZON})"
    )
    print(f"{'=' * 78}")
    if tradeable.empty:
        print("(no tradeable combinations on this universe at this bar)")
        return
    cols = ["asset", "pattern", "direction", "n", "edge", "expectancy", "edge_ratio", "ic"]
    print(
        tradeable[cols].to_string(
            index=False,
            formatters={
                "edge": "{:+.3f}".format,
                "expectancy": "{:+.3f}".format,
                "edge_ratio": "{:.2f}".format,
                "ic": "{:+.3f}".format,
            },
        )
    )
    print(f"\n{len(tradeable)} tradeable combinations found.")


def _print_matrix(lb: pd.DataFrame) -> None:
    """Wide pattern × asset matrix of edge over baseline. Picks the
    better of long / short for each cell.
    """
    best = (
        lb
        .sort_values("edge", ascending=False)
        .groupby(["asset", "pattern"], as_index=False)
        .first()
    )
    matrix = best.pivot(index="pattern", columns="asset", values="edge")
    print(f"\n{'=' * 78}")
    print(f"Edge-over-baseline matrix (best of long/short) at h={HORIZON}")
    print("Cells are hit_rate − baseline_hit (positive = pattern beats random)")
    print(f"{'=' * 78}\n")
    print(matrix.round(3).to_string(na_rep=" — "))


def main() -> None:
    bars = _load_bars()
    print(
        f"Loaded {bars.shape[0]:,} bars × "
        f"{bars.columns.get_level_values(-1).nunique()} assets ("
        f"{bars.index.min().strftime('%Y-%m-%d')} → {bars.index.max().strftime('%Y-%m-%d')})"
    )
    print(f"Min quality: {MIN_QUALITY}, horizon: {HORIZON}, sample floor: {MIN_EVENTS} events")

    lb = _build_leaderboard(bars)
    _print_per_asset(lb)
    _print_matrix(lb)
    _print_tradeable(lb)


if __name__ == "__main__":
    main()
