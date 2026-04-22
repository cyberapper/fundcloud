"""17 — Local data pipeline: CSV → Backend → Catalog → Features.

Quant-shop scenario: you dump nightly broker exports (per-symbol CSVs) into
a research folder and want a clean, addressable, cacheable dataset. This
example walks the full Fundcloud data layer end-to-end without touching the
network:

1. **Load** — :class:`fundcloud.data.CSV` (read-only source backend).
2. **Persist** — :class:`fundcloud.data.Parquet` and :class:`fundcloud.data.DuckDB`
   (swap-in writable backends; same protocol).
3. **Sync** — :meth:`Backend.sync_to` for direct one-off transfers
   (network → store) with explicit ``mode='upsert'`` semantics.
4. **Orchestrate** — :class:`Catalog` + :class:`DatasetSpec` bind a name to a
   source + sink and handle incremental refresh from a watermark, with
   ``refresh_kwargs`` for lookback corrections.
5. **Transform** — :mod:`fundcloud.data.bars` helpers (``to_prices``,
   ``to_returns``, ``to_log_returns``, ``resample``, ``align``,
   ``as_long`` / ``as_wide``).
6. **Cache features** — :class:`FeatureStore` keys by ``(dataset,
   pipeline_hash)`` so the second fit is an O(read) operation.

Run:
    uv run python examples/17_local_data_pipeline.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from fundcloud.data import (
    CSV,
    Catalog,
    DuckDB,
    Memory,
    Parquet,
    align,
    as_long,
    as_wide,
    resample,
    to_log_returns,
    to_returns,
)
from fundcloud.features import FeaturePipeline, FeatureStore

HERE = Path(__file__).parent
OUT = HERE / "out" / "17_pipeline"


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def _synthetic_csvs(dir_path: Path, symbols: list[str], n_bars: int = 504) -> None:
    """Write one OHLCV CSV per symbol — stand-in for a nightly broker dump."""
    dir_path.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range("2023-01-02", periods=n_bars)
    rng = np.random.default_rng(42)
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0003, 0.012, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(rets)) * (1 + i * 0.1)
        df = pd.DataFrame({
            "date": idx,
            "open": close * (1 + rng.normal(0, 0.001, n_bars)),
            "high": close * (1 + np.abs(rng.normal(0.002, 0.001, n_bars))),
            "low": close * (1 - np.abs(rng.normal(0.002, 0.001, n_bars))),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n_bars).astype(float),
        })
        df.to_csv(dir_path / f"{sym}.csv", index=False)


# --------------------------------------------------------------- SECTION 1


def load_from_csv(csv_dir: Path) -> pd.DataFrame:
    _rule("1. CSV — one file per symbol, auto-joined into Bars")
    bars = CSV(csv_dir).read()
    print(f"Symbols:        {sorted(set(bars.columns.get_level_values(-1)))}")
    print(f"Fields:         {sorted(set(bars.columns.get_level_values(0)))}")
    print(f"Shape:          {bars.shape}  ({len(bars)} bars × {bars.shape[1]} columns)")
    print(f"Date range:     {bars.index[0].date()} → {bars.index[-1].date()}")
    return bars


# --------------------------------------------------------------- SECTION 2


def persist_to_stores(bars: pd.DataFrame) -> tuple[Parquet, DuckDB]:
    _rule("2. Parquet + DuckDB backends — same protocol, different storage")
    parquet_dir = OUT / "parquet"
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)
    parquet = Parquet(parquet_dir)
    parquet.write("equity/us/daily", bars)

    duckdb_path = OUT / "warehouse.duckdb"
    if duckdb_path.exists():
        duckdb_path.unlink()
    # DuckDB's SQL surface only handles flat column frames cleanly; persist a
    # single-symbol slice for the round-trip demo. (Parquet handles MultiIndex
    # natively via pyarrow.)
    spy_flat = bars.xs("SPY", axis=1, level=-1)
    with DuckDB(duckdb_path) as duck:
        duck.write("equity/us/spy_daily", spy_flat)

    print(
        f"Parquet:   {parquet_dir.relative_to(HERE.parent)} "
        f"({sum(p.stat().st_size for p in parquet_dir.rglob('*.parquet')) / 1024:.1f} KB)"
    )
    print(
        f"DuckDB:    {duckdb_path.relative_to(HERE.parent)} "
        f"({duckdb_path.stat().st_size / 1024:.1f} KB)"
    )

    # Round-trip read demonstrates the keyed read API works the same on both.
    via_parquet = parquet.read("equity/us/daily")
    via_duckdb = DuckDB(duckdb_path).read("equity/us/spy_daily")
    print(f"Re-read via Parquet (MultiIndex):  {via_parquet.shape}")
    print(f"Re-read via DuckDB (flat SPY):     {via_duckdb.shape}, cols={list(via_duckdb.columns)}")
    return parquet, DuckDB(duckdb_path)


# --------------------------------------------------------------- SECTION 3


def sync_to_demo(bars: pd.DataFrame) -> None:
    _rule("3. sync_to — direct source → sink composition (mode='upsert')")
    spy = bars.xs("SPY", axis=1, level=-1)  # flat for DuckDB compatibility
    src = Memory({"spy_daily": spy})
    sync_db = OUT / "sync_demo.duckdb"
    if sync_db.exists():
        sync_db.unlink()
    duck = DuckDB(sync_db)

    # First sync: full copy.
    src.sync_to(duck, key="spy_daily", mode="upsert")
    print(f"After first sync_to:  {duck.read('spy_daily').shape}")

    # Repeat sync: upsert dedups on timestamp index, no row growth.
    src.sync_to(duck, key="spy_daily")
    print(f"After repeat sync_to: {duck.read('spy_daily').shape}  (idempotent)")

    # mode='append' is the escape hatch — it duplicates if the source overlaps.
    src.sync_to(duck, key="spy_daily", mode="append")
    print(f"After mode='append':  {duck.read('spy_daily').shape}  (rows doubled)")

    # Read with a time-window + column filter — proves the read API surface.
    sliced = duck.read(
        "spy_daily",
        start=spy.index[10],
        end=spy.index[20],
        columns=["close", "volume"],
    )
    print(f"Sliced read (10 bars, 2 cols): {sliced.shape}")
    duck.delete("spy_daily")
    duck.close()


# --------------------------------------------------------------- SECTION 4


def catalog_workflow(parquet: Parquet, csv_dir: Path) -> pd.DataFrame:
    _rule("4. Catalog — named datasets + watermark refresh + lookback corrections")
    cat = Catalog(store=Memory())
    cat.register(
        "equity-us-daily",
        CSV(csv_dir),
        store_key="equity/us/daily",
        refresh_kwargs={"lookback": "5D"},  # re-pull last 5 days to absorb corrections
        tags=("daily", "us"),
    )
    cat.register(
        "synthetic-inline",
        Memory({
            "default": pd.DataFrame(
                {"close": np.linspace(100, 110, 5)},
                index=pd.date_range("2024-01-02", periods=5),
            ),
        }),
        store_key="misc/inline",
    )
    bars = cat.load("equity-us-daily")
    print(f"Loaded 'equity-us-daily':  {bars.shape}")
    print()
    print("catalog.describe():")
    desc = cat.describe()
    print(desc[["name", "store_key", "last_index", "tags"]].to_string(index=False))

    # Second read prefers the store (no source call).
    _ = cat.load("equity-us-daily")

    # refresh() pulls from (watermark - lookback) using mode='upsert' so the
    # 5-day overlap dedups correctly; for a static CSV no new rows arrive.
    fresh = cat.refresh("equity-us-daily")
    print(f"\ncatalog.refresh('equity-us-daily') re-synced {len(fresh)} bars (upsert).")
    return bars


# --------------------------------------------------------------- SECTION 5


def bars_helpers(bars: pd.DataFrame) -> pd.DataFrame:
    _rule("5. bars helpers — prices, returns, resample, align, long/wide")
    closes = bars.xs("close", axis=1, level=0)
    print(f"close-panel:  {closes.shape}")

    simple = to_returns(closes)
    log = to_log_returns(closes)
    print(f"to_returns (simple) head:\n{simple.head(3).round(4)}")
    print(f"to_log_returns head:\n{log.head(3).round(4)}")

    monthly = resample(bars, rule="ME")
    print(f"\nresample -> monthly OHLCV:  {monthly.shape}")

    half_a = closes.iloc[:-20]
    half_b = closes.iloc[20:]
    aligned_a, aligned_b = align(half_a, half_b, how="inner")
    print(f"align(inner): {aligned_a.shape}, {aligned_b.shape}")

    long = as_long(closes, value_name="close")
    wide = as_wide(long, value="close")
    print(f"as_long (3-col tall format):  {long.shape}, columns={list(long.columns)}")
    print(f"as_wide round-trip:           {wide.shape}  (same asset count)")
    return simple


# --------------------------------------------------------------- SECTION 6


def feature_cache_round_trip(bars: pd.DataFrame, parquet: Parquet) -> None:
    _rule("6. FeatureStore — compute once, load forever (keyed by pipeline_hash)")
    try:
        from fundcloud.features.indicators import RSI, SMA
    except ImportError:
        print("TA-Lib not installed — `uv add 'fundcloud[ta]'`")
        return

    pipe = FeaturePipeline([("rsi_14", RSI(timeperiod=14)), ("sma_50", SMA(timeperiod=50))])
    store = FeatureStore(parquet, prefix="features")

    key = ("equity-us-daily", pipe)
    print(f"pipeline_hash:  {pipe.pipeline_hash}")
    print(f"cached?         {store.has(*key)}")

    feats = store.get_or_compute("equity-us-daily", pipe, bars)
    print(f"first call (cold compute):   shape={feats.shape}")

    feats2 = store.get_or_compute("equity-us-daily", pipe, bars)
    assert feats.equals(feats2), "cache miss on second call"
    print(f"second call (cache hit):     shape={feats2.shape}")
    print(f"keys under prefix:           {store.list('equity-us-daily')}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_dir = OUT / "csv"
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    _synthetic_csvs(csv_dir, ["SPY", "QQQ", "IWM"])

    bars = load_from_csv(csv_dir)
    parquet, _ = persist_to_stores(bars)
    sync_to_demo(bars)
    _ = catalog_workflow(parquet, csv_dir)
    _ = bars_helpers(bars)
    feature_cache_round_trip(bars, parquet)

    print("\nArtefacts:")
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(HERE.parent)}  ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
