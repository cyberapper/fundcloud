"""37 — Research data layer: extract + clean daily OHLCV from ClickHouse.

The :mod:`fundcloud.research` data layer pulls leak-free daily bars from the raw
``md_ohlcv_data`` feed and guarantees they are clean before any research uses them.

It handles the two quirks of that ``ReplacingMergeTree`` table that a naive read
gets wrong:

1. **Un-merged replacing-merge versions** — collapsed by ``argMax(…, updated_at)``.
2. **Two-to-three stamps per trading day** (00:00 / 04:00 / 20:00 UTC, same bar) —
   collapsed to one row per ``(symbol, date)`` by keeping the latest stamp.

What it shows:

1. **load_universe** — symbols with enough history (survivorship-free by default).
2. **load_bars** — clean ``(field, symbol)`` panel, exactly one bar per day.
3. **clean_panel / check_quality** — the data-quality gate.
4. **persist** — write the cleaned panel to Parquet and read it back.

Run against the real feed::

    export CLICKHOUSE_HOST=j8xa7yjxjo.asia-southeast1.gcp.clickhouse.cloud
    export CLICKHOUSE_USER=crescendo
    export CLICKHOUSE_PASSWORD=$(op-keychain resolve op://Agent/clickhouse-fundcloud-prod/value)
    uv run python examples/37_research_clean_data.py

Note: the library speaks ``clickhouse-connect`` (HTTPS on **8443**) — this example
forces 8443 and ignores the native ``9440`` some profiles advertise. With no
``CLICKHOUSE_HOST`` set it spins up an ephemeral ClickHouse via testcontainers,
seeds a fixture that reproduces every quirk above, and proves the cleaning offline.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fundcloud.research import check_quality, clean_panel, load_bars, load_universe

_OUT = Path(__file__).parent / "out"


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


# ------------------------------------------------------------------ fixture


@contextmanager
def _spawned_clickhouse() -> Any:
    """Spin up a fresh ClickHouse and seed an ``md_ohlcv_data``-shaped fixture.

    The fixture reproduces the production quirks so the cleaning is verifiable
    offline: 2 stamps/day per bar, a stale+corrected replacing-merge pair, a
    1970 epoch-garbage row, and one malformed ``high < low`` bar.
    """
    try:
        from testcontainers.clickhouse import ClickHouseContainer
    except ImportError as e:
        msg = (
            "This example needs 'testcontainers' to spin up a throwaway ClickHouse. "
            "Install with: uv add 'fundcloud[data-clickhouse]' && uv add testcontainers"
        )
        raise ImportError(msg) from e
    import clickhouse_connect

    with ClickHouseContainer("clickhouse/clickhouse-server:24.8") as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(8123))
        client = clickhouse_connect.get_client(
            host=host, port=port, username=container.username,
            password=container.password, database=container.dbname, secure=False,
        )
        try:
            client.command("DROP TABLE IF EXISTS md_ohlcv_data")
            client.command("""
                CREATE TABLE md_ohlcv_data (
                    prefix String, region String DEFAULT 'US',
                    symbol String, adapter String, asset_type String, timeframe String,
                    timestamp DateTime64(3),
                    open Float64, high Float64, low Float64, close Float64, volume Float64,
                    updated_at DateTime DEFAULT now()
                ) ENGINE = ReplacingMergeTree(updated_at)
                ORDER BY (symbol, timeframe, timestamp, prefix, adapter, asset_type)
            """)
            client.insert("md_ohlcv_data", _fixture_rows(), column_names=[
                "prefix", "symbol", "adapter", "asset_type", "timeframe",
                "timestamp", "open", "high", "low", "close", "volume", "updated_at",
            ])
        finally:
            client.close()
        yield {"host": host, "port": port, "user": container.username,
               "password": container.password, "database": container.dbname,
               "ssl": False, "table": "md_ohlcv_data"}


def _fixture_rows() -> list[tuple[Any, ...]]:
    """Rows reproducing the md_ohlcv_data quirks (see :func:`_spawned_clickhouse`)."""
    rows: list[tuple[Any, ...]] = []
    base_day = datetime(2024, 1, 2)
    upd = datetime(2024, 6, 1, 0, 0, 0)
    for sym, px0 in (("AAA", 100.0), ("BBB", 50.0)):
        for d in range(12):
            day = base_day + timedelta(days=d)
            if day.weekday() >= 5:  # skip weekends
                continue
            close = px0 + d
            o, h, low = close - 0.5, close + 1.0, close - 1.0
            # Same daily bar emitted at 04:00 and 20:00 UTC (20:00 = fuller volume).
            for hour, vol in ((4, 1_000_000.0), (20, 1_000_500.0)):
                ts = day.replace(hour=hour)
                rows.append((
                    "", sym, "polygon", "stock", "1D", ts,
                    o, h, low, close, vol, upd,
                ))
    # Stale + corrected replacing-merge pair on AAA day 0 @20:00 (argMax(updated_at) wins).
    fix_ts = base_day.replace(hour=20)
    rows.append(("", "AAA", "polygon", "stock", "1D", fix_ts,
                 99.5, 101.0, 99.0, 100.0, 1_000_500.0, datetime(2024, 1, 1)))   # stale
    rows.append(("", "AAA", "polygon", "stock", "1D", fix_ts,
                 99.5, 101.0, 99.0, 100.25, 1_000_500.0, datetime(2024, 12, 1)))  # corrected
    # Epoch-garbage row (excluded by the 2005 floor).
    rows.append(("", "AAA", "polygon", "stock", "1D", datetime(1970, 1, 2), 1.0, 1.0, 1.0, 1.0, 1.0, upd))
    # Malformed bar high < low (removed by clean_panel) — BBB last day @20:00.
    bad_day = base_day + timedelta(days=11)
    rows.append(("", "BBB", "polygon", "stock", "1D", bad_day.replace(hour=20),
                 61.0, 60.0, 62.0, 61.5, 1.0, datetime(2024, 12, 1)))
    return rows


# ------------------------------------------------------------------ run


def run_demo(conn: dict[str, Any], *, full_feed: bool) -> None:
    table = conn.pop("table", "default.md_ohlcv_data")

    _rule("1. load_universe — symbols with enough history (survivorship-free)")
    universe = load_universe(min_days=250 if full_feed else 5, table=table, **conn)
    print(f"universe size: {len(universe)} symbols")
    print(f"first 10: {universe[:10]}")
    symbols = (["AAPL", "MSFT", "SPY", "NVDA", "AMZN"] if full_feed else universe)

    _rule("2. load_bars — clean (field, symbol) panel, one bar per (symbol, date)")
    start = "2005-01-01"
    bars = load_bars(symbols=symbols, start=start, table=table, chunk_years=5, **conn)
    print(f"shape: {bars.shape}  (rows = trading days, cols = field x symbol)")
    if not bars.empty:
        print(f"date span: {bars.index.min().date()} -> {bars.index.max().date()}")
        print(f"symbols: {sorted(set(bars.columns.get_level_values(1)))}")
        dups = int(bars.index.duplicated().sum())
        print(f"duplicate dates: {dups}  (0 confirms one bar per day)")

    _rule("3. check_quality + clean_panel — the data-quality gate")
    raw_report = check_quality(bars, min_days=250 if full_feed else 5)
    print(f"pre-clean: high<low={raw_report.high_lt_low}, "
          f"ohlc_inconsistent={raw_report.ohlc_inconsistent}, "
          f"null_ohlc={raw_report.null_ohlc}, ok={raw_report.ok}")
    clean, report = clean_panel(bars)
    print(f"post-clean: n_symbols={report.n_symbols}, n_bars={report.n_bars}, "
          f"zero_volume={report.zero_volume}, missing_days={report.missing_trading_days}")
    print(f"clean.ok = {report.ok}")

    if not full_feed and "close" in clean.columns.get_level_values(0):
        # Fixture assertion: corrected close (argMax(updated_at)) must win.
        aaa = clean.xs("close", axis=1, level=0)["AAA"].iloc[0]
        print(f"AAA day-0 close = {aaa:.2f}  (expect 100.25 — corrected version won)")

    _rule("4. persist — write the cleaned panel to Parquet and read it back")
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / "research_clean_bars.parquet"
    clean.to_parquet(path)
    roundtrip = pd.read_parquet(path)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.1f} KiB)")
    print(f"read back: shape={roundtrip.shape}, matches={roundtrip.shape == clean.shape}")


def main() -> None:
    host = os.environ.get("CLICKHOUSE_HOST")
    if host:
        # Force HTTPS 8443 — clickhouse-connect cannot use the native 9440.
        conn: dict[str, Any] = {
            "host": host,
            "port": int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8443")),
            "table": "default.md_ohlcv_data",
        }
        if user := os.environ.get("CLICKHOUSE_USER"):
            conn["user"] = user
        if password := os.environ.get("CLICKHOUSE_PASSWORD"):
            conn["password"] = password
        if database := os.environ.get("CLICKHOUSE_DATABASE"):
            conn["database"] = database
        run_demo(conn, full_feed=True)
    else:
        print("CLICKHOUSE_HOST unset — using an ephemeral testcontainers ClickHouse.",
              file=sys.stderr)
        with _spawned_clickhouse() as conn:
            run_demo(conn, full_feed=False)


if __name__ == "__main__":
    main()
