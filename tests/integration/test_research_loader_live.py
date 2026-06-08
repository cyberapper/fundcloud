"""Live ClickHouse test for fundcloud.research.loader (network-marked).

Skipped by default. To run against the real ``md_ohlcv_data`` feed::

    export CLICKHOUSE_HOST=...                 # ClickHouse Cloud host
    export CLICKHOUSE_USER=...  CLICKHOUSE_PASSWORD=...
    uv run pytest tests/integration/test_research_loader_live.py -m network
"""

from __future__ import annotations

import os

import pytest
from fundcloud.research import clean_panel, load_bars

pytestmark = pytest.mark.network


def _conn() -> dict[str, object]:
    host = os.environ.get("CLICKHOUSE_HOST")
    if not host:
        pytest.skip("CLICKHOUSE_HOST unset — skipping live ClickHouse test")
    return {
        "host": host,
        "port": int(os.environ.get("CLICKHOUSE_HTTP_PORT", "8443")),
        "user": os.environ.get("CLICKHOUSE_USER"),
        "password": os.environ.get("CLICKHOUSE_PASSWORD"),
        "database": os.environ.get("CLICKHOUSE_DATABASE"),
    }


def test_live_load_bars_is_clean_one_bar_per_day() -> None:
    conn = _conn()
    bars = load_bars(symbols=["AAPL", "MSFT"], start="2024-01-01", end="2024-04-01", **conn)
    assert not bars.empty
    # the collapse guarantees one bar per (symbol, date)
    assert bars.index.is_monotonic_increasing
    assert int(bars.index.duplicated().sum()) == 0
    assert bars.index.tz is not None
    assert {"AAPL", "MSFT"}.issubset(set(bars.columns.get_level_values(1)))

    _clean, report = clean_panel(bars)
    assert report.ok
    assert report.n_symbols >= 1
