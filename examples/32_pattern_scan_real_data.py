"""32 — Chart-pattern scan on real market data (QQQ + SPY + Mag7).

Pulls daily OHLCV for QQQ, SPY, and the Mag7 (AAPL, MSFT, GOOGL, AMZN,
NVDA, META, TSLA) from inception to today via ``yfinance``. Caches the
download to ``examples/out/pattern_scan_bars.parquet`` so re-runs are
instant. Then runs both Head & Shoulders detectors on the full history
of each ticker and prints:

* a per-ticker summary (how many bars, how many detections, mean quality)
* the top 10 highest-quality detections across the whole universe
* the most-recent detection per ticker (useful for "is this set up now?")

Run:
    uv run python examples/32_pattern_scan_real_data.py

Optional flags:
    --refresh                  re-download even if the cache exists
    --min-quality 70           override the default 60.0 quality cutoff
    --tickers AAPL MSFT NVDA   restrict the universe
    --start 2010-01-01         override per-ticker inception fetch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fundcloud.features.patterns import (
    HeadAndShoulders,
    InverseHeadAndShoulders,
    Pattern,
)

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "pattern_scan_bars.parquet"


# Default universe: QQQ + SPY + Mag7. Easily edited in --tickers.
DEFAULT_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
)


# --------------------------------------------------------------------- data


def _download(tickers: tuple[str, ...], start: str | None) -> pd.DataFrame:
    """Download daily OHLCV for ``tickers`` and return a Bars MultiIndex frame.

    The frame has ``(field, asset)`` columns for fields
    ``open / high / low / close / volume`` and a UTC DatetimeIndex.
    Per-ticker series are joined on the union of trading dates, with
    missing values forward-filled to keep each asset's path contiguous.
    """
    import yfinance as yf

    # yfinance's default range is the last ~1 month — we must explicitly
    # ask for `period="max"` (or pass `start=...`) to get full history.
    print(
        f"  downloading {len(tickers)} tickers from yfinance "
        f"({'inception (period=max)' if start is None else start} → now) ..."
    )
    if start is None:
        raw = yf.download(
            list(tickers),
            period="max",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    else:
        raw = yf.download(
            list(tickers),
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

    # yfinance returns a (field, ticker) MultiIndex when group_by="ticker"
    # is set with multiple tickers. Normalize column names to lowercase and
    # reorder so the outer level is the field, inner level is the asset.
    per_asset: dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        if tkr not in raw.columns.get_level_values(0):
            print(f"    [warn] no data returned for {tkr}; skipping")
            continue
        sub = raw[tkr].rename(columns=str.lower)
        sub = sub[["open", "high", "low", "close", "volume"]].dropna(how="all")
        per_asset[tkr] = sub

    if not per_asset:
        msg = "yfinance returned no data for any ticker"
        raise RuntimeError(msg)

    # Build the MultiIndex Bars frame.
    fields = ("open", "high", "low", "close", "volume")
    columns = pd.MultiIndex.from_product(
        [list(fields), sorted(per_asset.keys())],
        names=["field", "asset"],
    )
    out = pd.DataFrame(
        index=sorted({i for df in per_asset.values() for i in df.index}),
        columns=columns,
        dtype=np.float64,
    )
    for asset, df in per_asset.items():
        for field in fields:
            out[(field, asset)] = df[field].reindex(out.index)

    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    out.sort_index(inplace=True)
    return out


def _load_or_download(
    tickers: tuple[str, ...],
    *,
    start: str | None,
    refresh: bool,
) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        cached = pd.read_parquet(CACHE)
        cached_assets = sorted(set(cached.columns.get_level_values("asset")))
        wanted = sorted(tickers)
        if cached_assets == wanted:
            print(
                f"  using cache: {CACHE.relative_to(HERE.parent)} "
                f"({len(cached)} rows × {len(cached_assets)} assets)"
            )
            return cached
        print(f"  cache covers {cached_assets}; need {wanted} — re-downloading")
    bars = _download(tickers, start)
    bars.to_parquet(CACHE)
    print(f"  cached → {CACHE.relative_to(HERE.parent)} ({len(bars)} rows × {len(tickers)} assets)")
    return bars


# ------------------------------------------------------------------ scanning


def _scan_one(indicator_cls: type, bars: pd.DataFrame, *, min_quality: float) -> pd.DataFrame:
    """Run ``indicator_cls(min_quality=...)`` and return the events table."""
    indicator = indicator_cls(min_quality=min_quality)
    events = indicator.events(bars)
    if not events.empty:
        events = events.copy()
        events["pattern_class"] = indicator_cls.__name__
    return events


def _per_ticker_summary(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-asset summary table."""
    rows: list[dict[str, Any]] = []
    for asset in sorted(set(bars.columns.get_level_values("asset"))):
        # Trading-day range for this ticker.
        close = bars[("close", asset)].dropna()
        if close.empty:
            continue
        first = close.index[0].date()
        last = close.index[-1].date()
        n_bars = len(close)
        per_asset = events[events["asset"] == asset] if not events.empty else events
        n_events = len(per_asset)
        rows.append({
            "asset": asset,
            "first": first,
            "last": last,
            "n_bars": n_bars,
            "n_detections": n_events,
            "mean_quality": (round(per_asset["quality"].mean(), 1) if n_events else float("nan")),
            "max_quality": (round(per_asset["quality"].max(), 1) if n_events else float("nan")),
            "n_bullish": (int((per_asset["pattern"] == Pattern.INVERSE_HEAD_AND_SHOULDERS).sum())),
            "n_bearish": (int((per_asset["pattern"] == Pattern.HEAD_AND_SHOULDERS).sum())),
        })
    return pd.DataFrame(rows).set_index("asset")


def _top_n(events: pd.DataFrame, n: int) -> pd.DataFrame:
    if events.empty:
        return events
    return (
        events
        .sort_values("quality", ascending=False)
        .head(n)[
            [
                "asset",
                "pattern",
                "formation_start",
                "formation_end",
                "entry_price",
                "quality",
            ]
        ]
        .reset_index(drop=True)
    )


def _most_recent_per_ticker(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    latest = events.sort_values("breakout_ts").groupby("asset", as_index=False).tail(1)
    return latest.sort_values("breakout_ts", ascending=False)[
        [
            "asset",
            "pattern",
            "breakout_ts",
            "entry_price",
            "quality",
        ]
    ].reset_index(drop=True)


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refresh", action="store_true", help="re-download bypass the cache")
    ap.add_argument("--min-quality", type=float, default=73.0, help="quality cutoff (0–100)")
    ap.add_argument(
        "--tickers", nargs="+", default=list(DEFAULT_TICKERS), help="override the universe"
    )
    ap.add_argument(
        "--start",
        default=None,
        help="ISO date; default is each ticker's inception (yfinance default)",
    )
    ap.add_argument("--top", type=int, default=10, help="how many top-quality events to print")
    args = ap.parse_args()
    tickers = tuple(args.tickers)

    print(f"\n{'─' * 8} 1. Universe + data {'─' * 8}")
    print(f"  tickers       : {list(tickers)}")
    bars = _load_or_download(tickers, start=args.start, refresh=args.refresh)
    print(f"  bars frame    : {bars.shape}  index={bars.index[0].date()} → {bars.index[-1].date()}")

    print(f"\n{'─' * 8} 2. Scan settings {'─' * 8}")
    print("  detectors     : HeadAndShoulders, InverseHeadAndShoulders")
    print(f"  min_quality   : {args.min_quality}")
    print("  pivot_orders  : (3, 5, 8)  (default)")

    print(f"\n{'─' * 8} 3. Running scan {'─' * 8}")
    bear = _scan_one(HeadAndShoulders, bars, min_quality=args.min_quality)
    print(f"  bearish H&S          → {len(bear)} detections")
    bull = _scan_one(InverseHeadAndShoulders, bars, min_quality=args.min_quality)
    print(f"  bullish Inverse H&S  → {len(bull)} detections")
    events = (
        pd.concat([bear, bull], ignore_index=True) if (not bear.empty or not bull.empty) else bear
    )

    print(f"\n{'─' * 8} 4. Per-ticker summary {'─' * 8}")
    summary = _per_ticker_summary(events, bars)
    print(summary.to_string())

    print(f"\n{'─' * 8} 5. Top {args.top} detections (by quality) {'─' * 8}")
    top = _top_n(events, args.top)
    if top.empty:
        print("  (no detections at the current threshold)")
    else:
        # Pretty-format dates and prices for the table.
        view = top.copy()
        view["pattern"] = view["pattern"].map(lambda p: p.value)
        view["formation_start"] = view["formation_start"].dt.date
        view["formation_end"] = view["formation_end"].dt.date
        view["entry_price"] = view["entry_price"].round(2)
        view["quality"] = view["quality"].round(1)
        print(view.to_string(index=False))

    print(f"\n{'─' * 8} 6. Most-recent detection per ticker {'─' * 8}")
    recent = _most_recent_per_ticker(events)
    if recent.empty:
        print("  (no detections)")
    else:
        view = recent.copy()
        view["pattern"] = view["pattern"].map(lambda p: p.value)
        view["breakout_ts"] = view["breakout_ts"].dt.date
        view["entry_price"] = view["entry_price"].round(2)
        view["quality"] = view["quality"].round(1)
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
