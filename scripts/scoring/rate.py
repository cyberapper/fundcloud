"""Interactive CLI for hand-rating sampled detections on textbook quality.

Reads the rating set produced by ``sample_for_rating.py``, walks each
detection in shuffled order, opens its chart in the default browser,
and prompts for a 0–100 grade. Resumable across runs — already-rated
detections are skipped.

The CLI **does not display the current scorer's quality** for the
detection. The point is to grade against your judgment of textbook
cleanliness, not against what v1.0.0 happens to score. See
``docs/scoring/quality.md#anti-patterns``.

Run:
    uv run python scripts/scoring/rate.py \\
        --rating-set scripts/scoring/rating_set.parquet \\
        --charts-dir scripts/scoring/charts \\
        --ratings scripts/scoring/ratings.csv \\
        --rater-id peter
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# CSV column order — written once on first run, appended thereafter.
_RATINGS_COLUMNS: tuple[str, ...] = (
    "detection_id",
    "rater_id",
    "rating",
    "note",
    "rated_at",
    "scorer_version",
    "pattern_value",
    "asset",
    "breakout_ts",
)


def _load_existing_ratings(path: Path) -> set[tuple[str, str]]:
    """Return (detection_id, rater_id) pairs already rated."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(zip(df["detection_id"].astype(str), df["rater_id"].astype(str), strict=True))


def _ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_RATINGS_COLUMNS)


def _append_rating(path: Path, row: dict) -> None:
    """Append + fsync. Crashes mid-rating don't lose work."""
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row[c] for c in _RATINGS_COLUMNS])
        f.flush()
        os.fsync(f.fileno())


def _prompt_int(prompt: str, lo: int, hi: int) -> int | None:
    """Read an int in [lo, hi] from stdin. Empty input returns None.

    Returns ``None`` for skip, raises ``KeyboardInterrupt`` for quit.
    """
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            raise KeyboardInterrupt() from None
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt()
        if raw in {"s", "skip", ""}:
            return None
        try:
            v = int(raw)
        except ValueError:
            print(f"  not an integer; expected {lo}–{hi}, 's' to skip, 'q' to quit")
            continue
        if not (lo <= v <= hi):
            print(f"  out of range; expected {lo}–{hi}")
            continue
        return v


def _prompt_str(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _open_chart(path: Path) -> None:
    if not path.exists():
        print(f"  ⚠️  chart missing: {path}")
        return
    # `webbrowser.open` returns True on most platforms even if the OS
    # is racing the call; that's fine for our use case.
    webbrowser.open(f"file://{path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rating-set", required=True, type=Path)
    parser.add_argument("--charts-dir", required=True, type=Path)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--rater-id", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't try to open the chart in a browser; print the path instead.",
    )
    args = parser.parse_args()

    if not args.rating_set.exists():
        sys.stderr.write(
            f"ERROR: rating set {args.rating_set} not found. Run sample_for_rating.py first.\n"
        )
        return 2

    rating_set = pd.read_parquet(args.rating_set)
    print(f"Loaded {len(rating_set)} candidate detections from {args.rating_set}")

    _ensure_header(args.ratings)
    already = _load_existing_ratings(args.ratings)
    rating_set = rating_set[
        ~rating_set["detection_id"].apply(lambda d: (d, args.rater_id) in already)
    ].reset_index(drop=True)
    print(f"  {len(already)} already rated by '{args.rater_id}', {len(rating_set)} to go")

    if rating_set.empty:
        print("Nothing left to rate. Exiting.")
        return 0

    # Shuffle deterministically so consecutive runs don't bias toward
    # particular bands but the order is reproducible per (rater, seed).
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in args.rater_id))
    order = rng.permutation(len(rating_set))
    rating_set = rating_set.iloc[order].reset_index(drop=True)

    print(
        "\nInstructions:\n"
        "  Rate each formation 0–100 on **how textbook it looks**, *purely*\n"
        "  on geometry. Do NOT factor in whether the trade worked.\n"
        "  • 95–100: textbook-perfect — could be a teaching example.\n"
        "  • 70–94:  good — recognisably the pattern, minor flaws.\n"
        "  • 40–69:  marginal — pattern detectable, several flaws.\n"
        "  • 1–39:   poor — barely the pattern.\n"
        "  • 0:      adversarial — calling this the pattern is a stretch.\n"
        "  's' to skip, 'q' to quit (saved progress is durable).\n"
    )

    rated = 0
    try:
        for i, row in rating_set.iterrows():
            print(f"\n[{i + 1}/{len(rating_set)}] {row['pattern_value']} on {row['asset']}")
            print(
                f"  formation: {pd.Timestamp(row['formation_start']).date()} → "
                f"{pd.Timestamp(row['formation_end']).date()} "
                f"({(pd.Timestamp(row['formation_end']) - pd.Timestamp(row['formation_start'])).days} days)"
            )
            chart = Path(row["chart_path"])
            if args.no_browser:
                print(f"  chart: {chart}")
            else:
                _open_chart(chart)

            rating = _prompt_int("  rating (0-100, s=skip, q=quit): ", 0, 100)
            if rating is None:
                continue
            note = _prompt_str("  note (optional, enter to skip): ")
            _append_rating(
                args.ratings,
                {
                    "detection_id": row["detection_id"],
                    "rater_id": args.rater_id,
                    "rating": rating,
                    "note": note,
                    "rated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "scorer_version": row["scorer_version"],
                    "pattern_value": row["pattern_value"],
                    "asset": row["asset"],
                    "breakout_ts": pd.Timestamp(row["breakout_ts"]).isoformat(),
                },
            )
            rated += 1
    except KeyboardInterrupt:
        print("\nQuit. Progress saved.")

    print(f"\nRated {rated} detection(s) this session → {args.ratings}")
    if rated >= 50:
        print("\nReady to calibrate? See:")
        print("  uv run python scripts/scoring/calibrate.py --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
