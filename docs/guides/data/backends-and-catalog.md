# Pulling and caching market data

Grab bars from Yahoo Finance and save them to DuckDB. That's the
headline use case — this page walks it end-to-end in plain English,
then shows the same pattern for other providers and stores.

## 60-second tour — YF into DuckDB

```python
from fundcloud.data import YF, DuckDB

YF(["SPY", "AAPL"]).sync_to(DuckDB("warehouse.duckdb"), key="us_eq")
```

Three things just happened:

1. Fundcloud asked Yahoo Finance for daily bars for SPY and AAPL.
2. It wrote them to `warehouse.duckdb` on disk, in a table named `us_eq`.
3. If `us_eq` already existed, overlapping days were deduplicated by
   timestamp — so you can run the same line every morning without
   producing duplicate rows.

Read it back whenever you need the data:

```python
from fundcloud.data import DuckDB

duck = DuckDB("warehouse.duckdb")
bars = duck.read("us_eq", start="2024-01-01", columns=["close"])
```

`start` / `end` accept strings or `pd.Timestamp`. `columns=["close"]`
trims the frame to just the close field across all symbols.

## Running it every day

For a scheduled job that keeps the cache fresh, ask the store where it
left off yesterday and pull forward from there:

```python
import pandas as pd
from fundcloud.data import YF, DuckDB

duck = DuckDB("warehouse.duckdb")
last = duck.last_index("us_eq")                         # most recent bar we have
start = last + pd.Timedelta(days=1) if last else None   # None = full history on first run

YF(["SPY", "AAPL"]).sync_to(duck, key="us_eq", start=start)
```

`sync_to` defaults to `mode="upsert"`, which dedupes by timestamp. If
you pass a `start` that overlaps with what's already cached, nothing
breaks — overlapping rows are replaced, not duplicated.

## Swap DuckDB for Parquet

The second argument to `sync_to` is the storage backend. Parquet works
the same way — files on disk instead of one DB file:

```python
from fundcloud.data import YF, Parquet

YF(["SPY", "AAPL"]).sync_to(Parquet("data/"), key="us_eq")
```

`Parquet("data/")` keeps one `.parquet` per key under `data/`.
`DuckDB("warehouse.duckdb")` keeps one table per key inside a single
DuckDB file. Pick whichever matches your workflow — the rest of your
code is unchanged.

## Other data providers

Every provider uses the same pattern. Swap `YF` for whichever one you
have keys for:

| Provider | Class | Install | Auth |
|---|---|---|---|
| Yahoo Finance | `YF` | `uv add 'fundcloud[data-yf]'` | none |
| FinancialModelingPrep | `FMP` | `uv add 'fundcloud[data-fmp]'` | `FMP_API_KEY` env var (or `api_key=`) |
| Alpha Vantage | `AV` | `uv add 'fundcloud[data-av]'` | `ALPHAVANTAGE_API_KEY` env var (or `api_key=`) |
| Binance | `Binance` | `uv add 'fundcloud[data-bn]'` | none |
| ClickHouse table | `ClickHouse` | `uv add 'fundcloud[data-clickhouse]'` | `host=` (required, explicit) |

`FMP` / `AV` / `FundCloud` accept `api_key=` explicitly *or* fall back
to the named env var. The ClickHouse backend is stricter — pass
`host=` (and any other connection params) explicitly so the credential
source is always traceable from the call site.

Reading from a Clickhouse table — including tables with composite asset
identifiers (HK / JP markets), multiple timeframes in one wide layout,
and arbitrary ML feature columns alongside OHLCV — has its own page:
[Reading from ClickHouse](clickhouse.md). It plugs into the same
`Backend` protocol, so everything below — `sync_to`, the `Catalog`,
write modes — works the same way.

### Adjusted vs raw equity prices

The three equity providers (`YF`, `FMP`, `AV`) default to **adjusted**
close prices — i.e. dividends and splits folded into a continuous
total-return series. This is the right default for backtests; raw
prices would give you fake "dividend day" drops.

Pass `adjust=False` if you genuinely need the as-traded price:

```python
total_return = YF("SGOV").read()                       # adjusted (default)
as_traded    = YF("SGOV", adjust=False).read()         # raw, with dividend dips

total_return = FMP("AAPL").read()                      # uses FMP's `adjClose`
as_traded    = FMP("AAPL", adjust=False).read()        # FMP's raw `close`

total_return = AV("IBM").read()                        # `..._ADJUSTED` endpoint
as_traded    = AV("IBM", adjust=False).read()          # `TIME_SERIES_DAILY` (free-tier)
```

`Binance` doesn't take `adjust` — crypto doesn't have dividends or
corporate actions to adjust for.

### Default window — 1 year if you don't specify one

Network providers can return decades of history. To keep accidental
calls cheap, **a bare `read()` defaults to one year**:

```python
YF("SPY").read()                       # last ~365 days
YF("SPY").read(start="2010-01-01")     # explicit — full history since 2010
YF("SPY").read(end="2024-12-31")       # 2023-12-31 → 2024-12-31
```

The default applies to every network backend (`YF`, `FMP`, `AV`,
`Binance`). Cache backends (`Parquet`, `DuckDB`, `Memory`, `CSV`) are
*not* affected — a bare `.read("us_eq")` on them returns whatever is
cached, which is the more useful default for local data.

```python
from fundcloud.data import FMP, AV, Binance, DuckDB

duck = DuckDB("warehouse.duckdb")

FMP(["AAPL", "MSFT"]).sync_to(duck, key="us_tech")       # needs FMP_API_KEY
AV("IBM").sync_to(duck, key="ibm_av")                     # needs ALPHAVANTAGE_API_KEY
Binance(["BTCUSDT"], interval="1h").sync_to(duck, key="btc_hourly")
```

## Other places to cache

Same `sync_to` signature, different destination:

| Store | Class | Good for |
|---|---|---|
| DuckDB | `DuckDB("file.duckdb")` | single-file warehouse, SQL over cached data |
| Parquet | `Parquet("folder/")` | one file per key, cloud / DVC sync |
| In-memory | `Memory()` | tests, notebooks, short-lived scripts |
| CSV | `CSV("file.csv")` | read-only by default; use Parquet/DuckDB for writes |

## Reading what you've cached

All cache stores expose the same small read API:

```python
duck = DuckDB("warehouse.duckdb")

duck.keys()                           # ['us_eq', 'us_tech', 'btc_hourly']
duck.exists("us_eq")                  # True
duck.last_index("us_eq")              # Timestamp('2026-04-18')

# Time-slice + column-prune read
window = duck.read("us_eq", start="2024-01-01", end="2024-03-31", columns=["close"])

# Copy one key to another store
duck.sync_to(Parquet("snapshots/"), key="us_eq")

# Remove a key
duck.delete("us_eq")
```

## Column naming — same shape from every provider

Each provider returns OHLCV in the same canonical shape, so downstream
code doesn't have to special-case yfinance vs FMP vs Alpha Vantage:

| What you get | Example |
|---|---|
| Lowercase field names | `open`, `high`, `low`, `close`, `volume` |
| Snake-cased extras | `adj_close` (from yfinance's `Adj Close`, FMP's `adjClose`) |
| Canonical order | `open → high → low → close → volume → adj_close` |
| Two-level columns: `(field, symbol)` | `bars[("close", "SPY")]` |

Apply the same normalisation to your own frames (e.g. a CSV with weird
headers) with the helpers Fundcloud uses internally:

```python
from fundcloud.data import normalize_ohlcv_columns, canonicalize_ohlcv_order

cleaned = canonicalize_ohlcv_order(normalize_ohlcv_columns(my_csv_frame))
```

## Many datasets at once — Catalog

When you're pulling three, five, ten datasets regularly, wiring each
by hand gets noisy. `Catalog` is the bookkeeping layer: register each
dataset once with its source and settings, then refresh them all with
one call.

```python
from fundcloud.data import Catalog, DuckDB, YF, Binance, FMP

cat = Catalog(store=DuckDB("warehouse.duckdb"))
cat.register("us_eq",  YF(["SPY", "AAPL"]),              tags=("equity",))
cat.register("crypto", Binance(["BTCUSDT"], interval="1h"))
cat.register("macro",  FMP(["CPI"]))

cat.refresh_all()                       # pull new rows for every dataset
cat.load("us_eq", start="2024-01-01")   # read from DuckDB, no network call

cat.describe()                          # DataFrame: one row per dataset
```

### Per-dataset refresh settings

Each dataset can carry defaults so you don't have to pass them on
every call:

```python
cat.register(
    "us_eq",
    YF(["SPY"]),
    refresh_kwargs={"start": "2010-01-01", "lookback": "5D"},
)
```

- **`start`** — earliest date to pull on the first run. Ignored on
  subsequent refreshes (the cache watermark takes over).
- **`end`** — latest date. Usually omitted to mean "today".
- **`lookback`** — how far back to re-pull each refresh. Useful when
  upstream corrects recent bars (stock splits, exchange revisions) —
  set it to a few days so corrections flow through. Default is 0.

### Save the catalog to a file

`Catalog.to_spec()` returns a plain dict — easy to round-trip through
YAML for a config-driven pipeline:

```python
spec = cat.to_spec()
cat2 = Catalog.from_spec(DuckDB("warehouse.duckdb"), spec)
```

## Write modes — when to use which

Every write (whether you call `.write()` directly or via `sync_to` /
`Catalog.refresh`) accepts a `mode` argument:

| mode | What it does | When to use |
|---|---|---|
| `overwrite` | Replace everything under that key | Initial load, full re-sync |
| `upsert` | Append + dedupe by timestamp | Daily refresh (default for `sync_to`) |
| `append` | Concatenate raw, no dedupe | You've validated no overlap is possible |
| `error` | Fail if key already exists | One-shot bootstrap, refuse to clobber |

## Read-only safety

All network providers (`YF`, `FMP`, `AV`, `Binance`) are read-only —
the `.write()` method raises `ReadOnlyError` on them. You can't
accidentally try to push data back to Yahoo.

For your local stores, set `read_only=True` to get the same guarantee
on a production read path:

```python
from fundcloud.data import Parquet

prod = Parquet("data/", read_only=True)
prod.read("us_eq")                     # ✅ fine
prod.write("us_eq", fresh_df)          # ❌ raises ReadOnlyError
```

## Reference — the `Backend` protocol

Everything above — `YF`, `DuckDB`, `Parquet`, `Memory`, `CSV`, `FMP`,
`AV`, `Binance` — implements the same small protocol. You only need to
read this section if you're writing your own backend or curious what
`sync_to` is really doing.

```python
class Backend(Protocol):
    name: ClassVar[str]                # short id ("yf", "duckdb", ...)
    read_only: bool

    # Available on every backend
    def read(self, key=None, *, start=None, end=None, columns=None) -> DataFrame: ...
    def keys(self) -> list[str]: ...
    def exists(self, key: str) -> bool: ...
    def last_index(self, key=None) -> Timestamp | None: ...

    # On read-write backends; raises ReadOnlyError when read_only=True
    def write(self, key, df, *, mode='overwrite') -> None: ...
    def delete(self, key) -> None: ...

    # One-liner source → sink transfer (read + write rolled together)
    def sync_to(self, sink, *, key=None, start=None, end=None, mode='upsert') -> DataFrame: ...
```

Full auto-generated class reference: [Data API reference](../../reference/data.md).
