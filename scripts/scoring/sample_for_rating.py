"""Sample detections for hand-rating, stratified across the quality range.

Run every pattern detector against the cached bars, take a stratified
sample across (pattern × quality band) so the rating set spans the
full distribution, render one interactive HTML chart per sampled
detection, and persist the rating set as a parquet file.

The rating set is the input to ``rate.py``. It contains everything the
rating CLI needs to display each detection plus the per-component
scorer output that ``calibrate.py`` will fit weights against.

Run:
    uv run python scripts/scoring/sample_for_rating.py \\
        --bars examples/out/pattern_scan_bars.parquet \\
        --out scripts/scoring/rating_set.parquet \\
        --charts-dir scripts/scoring/charts \\
        --n 200
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fundcloud.features.patterns import (
    AscendingTriangle,
    DescendingTriangle,
    DoubleBottom,
    DoubleTop,
    HeadAndShoulders,
    InverseHeadAndShoulders,
    Pattern,
    SymmetricalTriangle,
    TripleBottom,
    TripleTop,
)
from fundcloud.plots.patterns import plot_pattern_event

# Pattern enum value → indicator class.
_REGISTRY: dict[str, type] = {
    Pattern.HEAD_AND_SHOULDERS.value: HeadAndShoulders,
    Pattern.INVERSE_HEAD_AND_SHOULDERS.value: InverseHeadAndShoulders,
    Pattern.DOUBLE_TOP.value: DoubleTop,
    Pattern.DOUBLE_BOTTOM.value: DoubleBottom,
    Pattern.TRIPLE_TOP.value: TripleTop,
    Pattern.TRIPLE_BOTTOM.value: TripleBottom,
    Pattern.ASCENDING_TRIANGLE.value: AscendingTriangle,
    Pattern.DESCENDING_TRIANGLE.value: DescendingTriangle,
    Pattern.SYMMETRICAL_TRIANGLE.value: SymmetricalTriangle,
}

# Quality bands for stratification — same boundaries as the canonical
# fixture set so the calibration record stays consistent.
_BANDS: tuple[tuple[str, float, float], ...] = (
    ("excellent", 95.0, 100.0),
    ("good", 70.0, 94.99),
    ("marginal", 40.0, 69.99),
    ("poor", 1.0, 39.99),
)


def _detection_id(row: pd.Series) -> str:
    """Stable hash so the same detection gets the same id across runs."""
    payload = f"{row['pattern']}|{row['asset']}|{row['breakout_ts'].isoformat()}"
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _band_for(quality: float) -> str:
    for label, lo, hi in _BANDS:
        if lo <= quality <= hi:
            return label
    return "out_of_range"


def _scan_all_patterns(bars: pd.DataFrame, min_quality: float) -> pd.DataFrame:
    """Run every detector against the bars and concatenate their event tables."""
    frames: list[pd.DataFrame] = []
    for pattern_value, cls in _REGISTRY.items():
        events = cls(min_quality=min_quality).events(bars)
        if events.empty:
            continue
        events = events.copy()
        events["pattern_value"] = pattern_value
        frames.append(events)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _stratified_sample(events: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratify by (pattern × quality band), sample uniformly within each cell."""
    events = events.copy()
    events["band"] = events["quality"].apply(_band_for)
    events = events[events["band"] != "out_of_range"]
    if events.empty:
        return events

    cells = events.groupby(["pattern_value", "band"], group_keys=False)
    n_cells = cells.ngroups
    per_cell = max(1, n // n_cells)

    rng = np.random.default_rng(seed)

    def _take(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) <= per_cell:
            return g
        idx = rng.choice(len(g), size=per_cell, replace=False)
        return g.iloc[idx]

    sampled = cells.apply(_take).reset_index(drop=True)
    if len(sampled) > n:
        idx = rng.choice(len(sampled), size=n, replace=False)
        sampled = sampled.iloc[idx].reset_index(drop=True)
    return sampled


def _extract_components(meta: dict[str, Any]) -> dict[str, float]:
    """Pull sub-score components out of meta['features'] as floats in 0..=1."""
    feats = meta.get("features", {}) if isinstance(meta, dict) else {}
    return {
        "symmetry": float(feats.get("symmetry", float("nan"))),
        "volume": float(feats.get("volume", float("nan"))),
        "trendline_r2": float(feats.get("trendline_r2", float("nan"))),
        "completeness": float(feats.get("completeness", float("nan"))),
    }


def _render_charts(
    sampled: pd.DataFrame,
    bars: pd.DataFrame,
    charts_dir: Path,
) -> dict[str, Path]:
    """Render one interactive HTML chart per sampled detection. Idempotent."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for _, row in sampled.iterrows():
        det_id = row["detection_id"]
        out = charts_dir / f"{det_id}.html"
        paths[det_id] = out
        if out.exists():
            continue
        try:
            fig = plot_pattern_event(row, bars, padding=20)
            fig.write_html(str(out), include_plotlyjs="cdn")
        except Exception as e:
            sys.stderr.write(f"  warn: failed to render {det_id}: {e}\n")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--charts-dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.bars.exists():
        sys.stderr.write(
            f"ERROR: {args.bars} not found. Generate via:\n"
            "    uv run python examples/32_pattern_scan_real_data.py\n"
        )
        return 2

    print(f"Loading bars from {args.bars} …")
    bars = pd.read_parquet(args.bars)
    print(f"  {len(bars)} bars × {bars.columns.get_level_values(-1).nunique()} assets")

    print("Scanning all patterns …")
    events = _scan_all_patterns(bars, args.min_quality)
    if events.empty:
        sys.stderr.write("  no events found; nothing to sample\n")
        return 1
    print(f"  {len(events)} total detections across {events['pattern_value'].nunique()} patterns")

    events["detection_id"] = events.apply(_detection_id, axis=1)
    components = events["meta"].apply(_extract_components).apply(pd.Series)
    events = pd.concat([events, components], axis=1)
    events["scorer_version"] = events["meta"].apply(
        lambda m: m.get("scorer_version") if isinstance(m, dict) else None
    )

    print(f"Sampling {args.n} (stratified by pattern × quality band) …")
    sampled = _stratified_sample(events, args.n, args.seed)
    print(f"  {len(sampled)} sampled, by band:")
    for band, cnt in sampled["band"].value_counts().items():
        print(f"    {band}: {cnt}")

    print(f"Rendering charts → {args.charts_dir} …")
    chart_paths = _render_charts(sampled, bars, args.charts_dir)
    sampled["chart_path"] = sampled["detection_id"].map(lambda d: str(chart_paths.get(d, "")))

    keep_cols = [
        "detection_id",
        "pattern_value",
        "asset",
        "direction",
        "formation_start",
        "formation_end",
        "breakout_ts",
        "quality",
        "band",
        "symmetry",
        "volume",
        "trendline_r2",
        "completeness",
        "scorer_version",
        "chart_path",
    ]
    out = sampled[keep_cols].copy()
    out["direction"] = out["direction"].astype(str)
    out["pattern_value"] = out["pattern_value"].astype(str)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nWrote rating set: {args.out}")
    print(f"Charts: {args.charts_dir}/")
    print("\nNext: uv run python scripts/scoring/rate.py --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
