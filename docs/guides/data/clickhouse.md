# Reading from ClickHouse

`fundcloud.data.ClickHouse` is a read-only backend that reads OHLCV +
arbitrary feature columns from a Clickhouse table directly into the
canonical `(field, symbol)` MultiIndex DataFrame the rest of the library
expects. It's designed for the case where many symbols (and optionally
many timeframes) coexist in a single wide table — the typical materialised
view layout.

## Install

```bash
uv add 'fundcloud[data-clickhouse]'   # or 'fundcloud[data]' for the full bundle
```

The driver is `clickhouse-connect` (HTTPS, port 8443/8123 — works with
ClickHouse Cloud out of the box).

## 60-second tour

```python
from fundcloud.data import ClickHouse

ch = ClickHouse(
    host="...clickhouse.cloud", port=8443,
    user="viewer", password="...",
    database="default",
    table="bars",
    asset_cols=["symbol"],          # one column identifies an asset
    timestamp_col="timestamp",      # default
)

print(ch.assets())                  # asset, start, end, n_rows
bars = ch.read(start="2024-01-01")  # DatetimeIndex × (field, symbol) MultiIndex
```

`ch.read()` always normalises to lowercase OHLCV in canonical order
(`open, high, low, close, volume`), with feature columns appearing after.

## Connection

Pass connection params explicitly — the backend never reads env vars
on its own. `host` is required; the others are optional.

```python
import os

ch = ClickHouse(
    host=os.environ["CLICKHOUSE_HOST"],
    port=int(os.environ.get("CLICKHOUSE_PORT", "8443")),
    user=os.environ.get("CLICKHOUSE_USER"),
    password=os.environ.get("CLICKHOUSE_PASSWORD"),
    database=os.environ.get("CLICKHOUSE_DATABASE"),
    table="bars",
)
```

`ssl=True` is the default and selects HTTPS via `clickhouse-connect`'s
`secure=True` flag — the right choice for ClickHouse Cloud.

## Composite asset identifiers

Hong Kong and Japanese markets use numeric stock codes that need an
exchange prefix to disambiguate. ClickHouse's `asset_cols` lists the
columns that together identify one asset; values are joined with
`asset_separator` (default `":"`) into the symbol level of the
`(field, symbol)` MultiIndex.

```python
ch = ClickHouse(
    table="bars",
    asset_cols=["prefix", "code"],     # composite key
    asset_separator=":",               # default
    ...
)

bars = ch.read()
bars.columns
# MultiIndex: [('open','HKEX:0001'), ('open','TSE:7203'), ..., ('rsi_14','HKEX:0001'), ...]

ch.keys()
# ['HKEX:0001', 'HKEX:0002', 'TSE:6758', 'TSE:7203']

ch.read(key="HKEX:0001")               # filter to one asset
```

If you don't pass `asset_cols`, the WHERE-filtered slice is treated as
one anonymous asset — `read()` returns flat columns, and `keys()` /
`assets()` are empty.

## Discovering what's in the table

`assets()` returns a one-row-per-asset DataFrame with the period
covered by each asset and its row count:

```python
ch.assets()
#        asset      start        end  n_rows
#    HKEX:0001 2023-01-02 2024-12-30   500
#    HKEX:0002 2023-06-15 2024-12-30   400
#     TSE:7203 2023-01-02 2024-12-30   500
#     TSE:6758 2023-01-02 2024-12-30   500
```

Use it to sanity-check coverage before loading into a portfolio
optimiser, or to drive a watchlist.

## Custom OHLCV column names

Different shops name OHLCV columns differently. `ohlcv_map` lets you
override only the names that differ from the canonical
`open/high/low/close/volume`:

```python
ch = ClickHouse(
    table="bars",
    ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
)
```

Missing OHLCV columns are simply absent from the result — no error.
That makes it safe to point the backend at a feature-only or
close-only table.

## Feature columns

By default (`feature_cols="*"`), every column not consumed by the
asset / timestamp / timeframe / OHLCV mapping flows through as an extra
feature column, alongside OHLCV in the same MultiIndex:

```python
bars.columns.get_level_values(0).unique()
# ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'sentiment']
```

Restrict to a specific list when you don't need the full set:

```python
ClickHouse(table="bars", feature_cols=["rsi_14", "sentiment"])
```

Or drop all features and read pure OHLCV:

```python
ClickHouse(table="bars", feature_cols=None)
```

## Timeframe filter

If your table interleaves multiple intervals (1m, 1h, 1d), pair
`timeframe_col` (the column name) with `timeframe` (the value to keep)
to filter at the SQL level — Clickhouse drops the irrelevant rows
server-side, so you don't pay for them:

```python
ClickHouse(
    table="bars",
    timeframe_col="tf",
    timeframe="1h",
)
```

Both default to `None`. The `timeframe_col` is dropped from the output
once the filter has been applied.

## Arbitrary SQL filters — the `where` escape hatch

The structured options cover the common cases. For everything else,
pass a raw SQL fragment that gets ANDed onto every query:

```python
ClickHouse(
    table="bars",
    where="source = 'whale' AND venue IN ('binance','bybit')",
)
```

Values inside `where` are *not* parameterised — escape them yourself.
Use this for shop-specific filters (`source`, `region`, `asset_class`,
…) you'd otherwise have to wrap in a Clickhouse `VIEW`.

## Read-only in v0.1

`ClickHouse(...)` is read-only. Calling `write()` or `delete()` raises
`ReadOnlyError`. Writes will arrive in a later version; for now, ingest
into Clickhouse via your existing pipeline and use fundcloud just for
reads.

You can still pipe data *out of* Clickhouse into a writable cache:

```python
from fundcloud.data import ClickHouse, DuckDB

ClickHouse(table="bars", asset_cols=["symbol"]).sync_to(
    DuckDB("warehouse.duckdb"),
    key="bars_cache",
    mode="upsert",
)
```

## Working with a `Catalog`

The `Catalog` orchestrator pairs a source backend with a sink. Use
ClickHouse as the source for any dataset whose canonical home is in
your warehouse:

```python
from fundcloud.data import Catalog, ClickHouse, DuckDB

cat = Catalog(store=DuckDB("warehouse.duckdb"))
cat.register(
    "us_eq",
    source=ClickHouse(
        table="bars",
        asset_cols=["symbol"],
        timeframe_col="tf",
        timeframe="1d",
    ),
    store_key="us_eq",
)

cat.refresh("us_eq")               # pulls forward from the cache watermark
cat.load("us_eq", start="2024-01-01")
```

## Reference

API docs: [`ClickHouse`][fundcloud.data.ClickHouse] in the
[Data API reference](../../reference/data.md).

End-to-end example, including spinning up a throwaway Clickhouse via
Docker testcontainers: `examples/31_clickhouse_data_source.py`.
