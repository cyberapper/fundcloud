"""31 — ClickHouse data source: discover, read, feature-engineer, optimise.

The :class:`fundcloud.data.ClickHouse` backend is read-only in v0.1 and is
designed for tables where many symbols (and optionally many timeframes)
coexist in a single wide layout — the typical materialised-view shape for
tick / aggregated bar storage. This example walks the full lifecycle
against a fixture table that this script seeds itself, so it runs from a
clean checkout without any pre-existing database.

What it shows:

1. **Connect** — credentials from constructor args or ``CLICKHOUSE_*`` env
   vars, with HTTPS enabled by default.
2. **Configure schema** — composite asset identifier (``["prefix",
   "code"]``), custom timestamp / timeframe columns, custom OHLCV
   column names via ``ohlcv_map``, and arbitrary feature columns flowing
   through automatically.
3. **Discover** — :meth:`ClickHouse.assets` shows every asset with its
   period coverage; :meth:`ClickHouse.keys` returns the joined-string
   identifiers.
4. **Read** — single asset, all assets as a ``(field, symbol)``
   MultiIndex, and a windowed slice for a portfolio optimiser.
5. **Engineer features** — feed the bars into a :class:`FeaturePipeline`
   to compute TA-Lib indicators alongside the columns ClickHouse already
   carries (e.g. an ML sentiment score).
6. **Optimise** — small HRP / equal-weight comparison on the closes.

Run:
    # Spins up an ephemeral clickhouse-server via Docker testcontainers,
    # seeds a fixture table, and runs end-to-end:
    uv run python examples/31_clickhouse_data_source.py

    # Or point at an existing Clickhouse via env vars:
    export CLICKHOUSE_HOST=...; export CLICKHOUSE_USER=...; export CLICKHOUSE_PASSWORD=...
    export FC_EXAMPLE_TABLE=tv_whale_snapshot_latest_mv
    uv run python examples/31_clickhouse_data_source.py
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from fundcloud.data import ClickHouse


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


# ------------------------------------------------------------------ fixture


@contextmanager
def _spawned_clickhouse() -> Any:
    """Spin up a fresh clickhouse/clickhouse-server via testcontainers and seed it.

    Yields a dict of connection kwargs (host/port/...) the example can pass
    straight into ``ClickHouse(...)``.
    """
    try:
        from testcontainers.clickhouse import ClickHouseContainer
    except ImportError as e:
        msg = (
            "This example requires the 'testcontainers' package to spin up a "
            "throwaway Clickhouse for the demo. Install with: "
            "uv add 'fundcloud[data-clickhouse]' && uv add testcontainers"
        )
        raise ImportError(msg) from e
    import clickhouse_connect

    with ClickHouseContainer("clickhouse/clickhouse-server:24.8") as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(8123))
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=container.username,
            password=container.password,
            database=container.dbname,
            secure=False,
        )
        try:
            client.command("DROP TABLE IF EXISTS bars")
            client.command("""
                CREATE TABLE bars (
                    ts DateTime64(3),
                    prefix String,
                    code String,
                    tf String,
                    o Float64, h Float64, l Float64, c Float64, v Float64,
                    rsi_14 Float64, sentiment Float64
                ) ENGINE = MergeTree() ORDER BY (prefix, code, tf, ts)
            """)
            rng = np.random.default_rng(42)
            ts_index = pd.date_range("2023-01-02", periods=252, freq="1D", tz="UTC")
            rows: list[tuple[Any, ...]] = []
            for prefix, code in [
                ("HKEX", "0001"),
                ("HKEX", "0002"),
                ("TSE", "7203"),
                ("TSE", "6758"),
            ]:
                base = 100.0 + rng.normal(0, 5)
                drift = rng.normal(0, 1, len(ts_index))
                close = base + np.cumsum(drift)
                open_ = close + rng.normal(0, 0.2, len(ts_index))
                high = np.maximum(open_, close) + rng.uniform(0, 0.5, len(ts_index))
                low = np.minimum(open_, close) - rng.uniform(0, 0.5, len(ts_index))
                vol = rng.integers(100_000, 200_000, len(ts_index)).astype(float)
                rsi = rng.uniform(20, 80, len(ts_index))
                sent = rng.normal(0, 0.3, len(ts_index))
                for i, ts in enumerate(ts_index):
                    rows.append((
                        ts.to_pydatetime(),
                        prefix, code, "1d",
                        float(open_[i]), float(high[i]), float(low[i]),
                        float(close[i]), float(vol[i]),
                        float(rsi[i]), float(sent[i]),
                    ))
            client.insert(
                "bars", rows,
                column_names=[
                    "ts", "prefix", "code", "tf", "o", "h", "l", "c", "v",
                    "rsi_14", "sentiment",
                ],
            )
        finally:
            client.close()
        yield {
            "host": host,
            "port": port,
            "user": container.username,
            "password": container.password,
            "database": container.dbname,
            "ssl": False,
            "table": "bars",
        }


# ------------------------------------------------------------------ run


def run_demo(kwargs: dict[str, Any]) -> None:
    _rule("1. Connect — composite asset key + custom OHLCV column names")
    ch = ClickHouse(
        asset_cols=["prefix", "code"],     # HK / JP-style composite identifier
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1d",                    # filter to daily bars
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        # feature_cols defaults to "*"  -> rsi_14 & sentiment flow through
        **kwargs,
    )
    print(f"connected: name={ch.name}, read_only={ch.read_only}")

    _rule("2. Discover — every asset with its period coverage")
    catalog = ch.assets()
    print(catalog.to_string(index=False))
    print()
    print(f"keys() (composite-id strings): {ch.keys()}")
    print(f"last_index() (max ts across all assets): {ch.last_index()}")

    _rule("3. Read one asset (flat columns)")
    one = ch.read(key="HKEX:0001", start="2023-06-01", end="2023-09-30")
    print(f"shape: {one.shape}")
    print(f"columns: {list(one.columns)}")
    print(one.head(3).round(3).to_string())

    _rule("4. Read every asset (MultiIndex columns)")
    bars = ch.read(start="2023-06-01", end="2023-09-30")
    print(f"shape: {bars.shape}")
    print(f"symbols: {sorted(set(bars.columns.get_level_values(1)))}")
    print(f"fields: {list(dict.fromkeys(bars.columns.get_level_values(0)))}")
    print()
    print("closes:")
    closes = bars.xs("close", axis=1, level=0)
    print(closes.head(3).round(3).to_string())

    _rule("5. Feature engineering — add a TA-Lib indicator on top of CH features")
    try:
        from fundcloud.features import FeaturePipeline
        from fundcloud.features.indicators import RSI, SMA
    except ImportError:
        print("Skip feature step — install with `uv add 'fundcloud[ta]'`.", file=sys.stderr)
    else:
        pipe = FeaturePipeline([("sma_50", SMA(timeperiod=50)), ("rsi_14", RSI(timeperiod=14))])
        feats = pipe.fit_transform(bars).dropna()
        print(f"feature matrix: {feats.shape}")
        print(feats.tail(3).round(3).to_string())
        print()
        print("ClickHouse-stored features (rsi_14, sentiment) come back alongside OHLCV:")
        ch_feats = bars.loc[:, bars.columns.get_level_values(0).isin(["rsi_14", "sentiment"])]
        print(ch_feats.tail(3).round(3).to_string())

    _rule("6. Optimise — equal-weight portfolio on closes")
    weights = pd.Series(1.0 / closes.shape[1], index=closes.columns, name="w")
    rets = closes.pct_change().dropna()
    pf_returns = (rets * weights).sum(axis=1)
    print(f"equal-weight portfolio: mean={pf_returns.mean():+.4%}/d, "
          f"std={pf_returns.std():.4%}, sharpe(annualised)={pf_returns.mean() / pf_returns.std() * (252**0.5):.2f}")


def main() -> None:
    host = os.environ.get("CLICKHOUSE_HOST")
    if host:
        kwargs: dict[str, Any] = {
            "table": os.environ.get("FC_EXAMPLE_TABLE", "bars"),
            "host": host,
        }
        port = os.environ.get("CLICKHOUSE_PORT")
        if port:
            kwargs["port"] = int(port)
        if user := os.environ.get("CLICKHOUSE_USER"):
            kwargs["user"] = user
        if password := os.environ.get("CLICKHOUSE_PASSWORD"):
            kwargs["password"] = password
        if database := os.environ.get("CLICKHOUSE_DATABASE"):
            kwargs["database"] = database
        run_demo(kwargs)
    else:
        with _spawned_clickhouse() as kwargs:
            run_demo(kwargs)


if __name__ == "__main__":
    main()
