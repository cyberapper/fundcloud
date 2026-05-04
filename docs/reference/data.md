# Data

The `fundcloud.data` package is the single entry point for loading and persisting market data. Every backend implements the same [`Backend`](../guides/data/backends-and-catalog.md#reference-the-backend-protocol) protocol — reads always work, writes are gated by a `read_only` constructor flag. The [`Catalog`](../guides/data/backends-and-catalog.md#many-datasets-at-once-catalog) composes a source backend onto a sink backend with watermark-driven incremental refresh. For the task-first walkthrough, start with the [Pulling and caching market data](../guides/data/backends-and-catalog.md) guide.

::: fundcloud.data
    options:
      members:
        - Backend
        - BaseBackend
        - ReadOnlyError
        - WriteMode
        - YF
        - FMP
        - AV
        - Binance
        - FundCloud
        - ClickHouse
        - CSV
        - Parquet
        - DuckDB
        - Memory
        - Catalog
        - DatasetSpec
        - OHLCV_COLUMNS
        - normalize_field
        - normalize_ohlcv_columns
        - canonicalize_ohlcv_order

::: fundcloud.data.bars

::: fundcloud.data.catalog
