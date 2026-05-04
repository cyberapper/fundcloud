"""Market data — unified ``Backend`` abstraction + ``Catalog`` orchestrator.

Every data backend (network providers like :class:`YF`, :class:`FMP`,
:class:`AV`, :class:`Binance`; local format backends like :class:`CSV`,
:class:`Parquet`, :class:`DuckDB`, :class:`Memory`) implements the single
:class:`Backend` protocol. Reads always work; writes are gated by the
``read_only`` constructor flag and raise :class:`ReadOnlyError` when locked.

:class:`Catalog` binds named datasets to (source, sink) pairs and handles
incremental refresh from sink watermarks via :meth:`Backend.sync_to`.

Network backends are lazy-imported via :func:`__getattr__` so installs
without ``yfinance`` / ``ccxt`` / ``httpx`` keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fundcloud.data import bars
from fundcloud.data._base import Backend, BaseBackend, ReadOnlyError, WriteMode
from fundcloud.data._columns import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_field,
    normalize_ohlcv_columns,
)
from fundcloud.data.bars import (
    align,
    as_long,
    as_wide,
    resample,
    to_log_returns,
    to_prices,
    to_returns,
)
from fundcloud.data.catalog import Catalog, DatasetSpec
from fundcloud.data.csv import CSV
from fundcloud.data.duckdb import DuckDB
from fundcloud.data.memory import Memory
from fundcloud.data.parquet import Parquet

__all__ = [
    "AV",
    "CSV",
    "FMP",
    "OHLCV_COLUMNS",
    "YF",
    "Backend",
    "BaseBackend",
    "Binance",
    "Catalog",
    "ClickHouse",
    "DatasetSpec",
    "DuckDB",
    "FundCloud",
    "Memory",
    "Parquet",
    "ReadOnlyError",
    "WriteMode",
    "align",
    "as_long",
    "as_wide",
    "bars",
    "canonicalize_ohlcv_order",
    "normalize_field",
    "normalize_ohlcv_columns",
    "resample",
    "to_log_returns",
    "to_prices",
    "to_returns",
]


_LAZY: dict[str, tuple[str, str]] = {
    "YF": ("fundcloud.data.yf", "YF"),
    "FMP": ("fundcloud.data.fmp", "FMP"),
    "AV": ("fundcloud.data.av", "AV"),
    "Binance": ("fundcloud.data.binance", "Binance"),
    "FundCloud": ("fundcloud.data.fundcloud", "FundCloud"),
    "ClickHouse": ("fundcloud.data.clickhouse", "ClickHouse"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        from importlib import import_module

        module = import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'fundcloud.data' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise lazy-loaded backends to IDEs / REPL tab-completion."""
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # pragma: no cover — static-analysis only
    from fundcloud.data.av import AV
    from fundcloud.data.binance import Binance
    from fundcloud.data.clickhouse import ClickHouse
    from fundcloud.data.fmp import FMP
    from fundcloud.data.fundcloud import FundCloud
    from fundcloud.data.yf import YF
