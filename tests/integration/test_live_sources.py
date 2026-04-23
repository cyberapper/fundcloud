"""Live integration tests — one per data backend.

Every test in this module is marked ``@pytest.mark.network`` and is
**skipped by default**. To run them:

    uv run pytest tests/integration -m network -q

Requires:

* ``FMP_API_KEY`` (for FMP tests — free tier works).
* ``ALPHAVANTAGE_API_KEY`` or ``ALPHA_VANTAGE_API_KEY`` (for AV tests).
* Internet connectivity to yfinance and Binance for those two.

Each test hits the provider once, validates the shape of the returned
``Bars`` frame, and runs a tiny Simulator + tear sheet to prove the
frame can flow through the full Fundcloud pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fundcloud.data import DuckDB
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet
from fundcloud.sim import Simulator
from fundcloud.strategies import Hold

pytestmark = pytest.mark.network


OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def _require_env(*names: str) -> str:
    """Return the first env var value that's set, else skip."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    pytest.skip(f"none of {names!r} is set in the environment")


def _assert_sane_bars(bars: pd.DataFrame, *, symbol: str, min_rows: int = 10) -> None:
    """Shared post-conditions: MultiIndex columns, datetime index, OHLCV present."""
    assert isinstance(bars, pd.DataFrame), f"expected DataFrame, got {type(bars)}"
    assert len(bars) >= min_rows, f"too few bars: {len(bars)}"
    assert isinstance(bars.index, pd.DatetimeIndex), "expected DatetimeIndex"
    assert isinstance(bars.columns, pd.MultiIndex), "expected (field, symbol) MultiIndex"
    fields = set(bars.columns.get_level_values(0))
    assert {"open", "high", "low", "close"}.issubset({f.lower() for f in fields}), (
        f"missing required OHLC fields; got {fields}"
    )
    assets = set(bars.columns.get_level_values(-1))
    assert symbol in assets, f"symbol {symbol!r} not found in {assets}"


def _run_hold(bars: pd.DataFrame, symbol: str, *, cash: float = 10_000.0) -> Portfolio:
    """Run Hold(all-in) through the Simulator to prove end-to-end flow."""
    result = Simulator(bars, cash=cash).run_strategy(Hold(weights={symbol: 1.0}))
    return result.portfolio


# ---------------------------------------------------------------------- yfinance


def test_yf_spy_daily() -> None:
    pytest.importorskip("yfinance")
    from fundcloud.data import YF

    src = YF(symbols=["SPY"], interval="1d")
    bars = src.read(start="2024-01-02", end="2024-03-31")

    _assert_sane_bars(bars, symbol="SPY", min_rows=40)
    assert bars.index.is_monotonic_increasing

    portfolio = _run_hold(bars, "SPY")
    assert portfolio.returns.notna().all()
    Tearsheet(portfolio, title="YF SPY — live test").render_html(OUT / "yf_spy.html")


def test_yf_multiple_symbols() -> None:
    pytest.importorskip("yfinance")
    from fundcloud.data import YF

    src = YF(symbols=["SPY", "AGG"], interval="1d")
    bars = src.read(start="2024-01-02", end="2024-03-31")

    _assert_sane_bars(bars, symbol="SPY", min_rows=40)
    _assert_sane_bars(bars, symbol="AGG", min_rows=40)


# ---------------------------------------------------------------------- FMP


def test_fmp_aapl_daily() -> None:
    _require_env("FMP_API_KEY")
    from fundcloud.data import FMP

    src = FMP(symbols="AAPL", interval="1d")
    bars = src.read(start="2024-01-02", end="2024-03-31")

    _assert_sane_bars(bars, symbol="AAPL", min_rows=40)
    portfolio = _run_hold(bars, "AAPL")
    assert portfolio.returns.notna().all()
    Tearsheet(portfolio, title="FMP AAPL — live test").render_html(OUT / "fmp_aapl.html")


# --------------------------------------------------------------------- Alpha Vantage


def test_av_ibm_daily() -> None:
    _require_env("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY")
    from fundcloud.data import AV

    # IBM is Alpha Vantage's canonical free-tier test ticker.
    src = AV(symbols="IBM", interval="1d")
    bars = src.read()

    _assert_sane_bars(bars, symbol="IBM", min_rows=250)
    portfolio = _run_hold(bars.iloc[-60:], "IBM")
    assert portfolio.returns.notna().all()
    Tearsheet(portfolio, title="AV IBM — live test (last 60d)").render_html(OUT / "av_ibm.html")


# ---------------------------------------------------------------------- Binance


def test_binance_btcusdt_daily() -> None:
    pytest.importorskip("ccxt")
    from fundcloud.data import Binance

    src = Binance(symbols="BTC/USDT", interval="1d", limit=500)
    bars = src.read(start="2024-01-01", end="2024-03-31")

    _assert_sane_bars(bars, symbol="BTC/USDT", min_rows=60)
    assert bars.index.is_monotonic_increasing

    portfolio = _run_hold(bars, "BTC/USDT")
    assert portfolio.returns.notna().all()
    Tearsheet(portfolio, title="Binance BTC/USDT — live test").render_html(OUT / "binance_btc.html")


# --------------------------------------------------------------- sync_to round-trip


def test_yf_sync_to_duckdb_is_idempotent(tmp_path: Path) -> None:
    """Live network → DuckDB cache; double-sync via mode='upsert' must dedup."""
    pytest.importorskip("yfinance")
    from fundcloud.data import YF

    src = YF(symbols="SPY", interval="1d")
    sink = DuckDB(tmp_path / "warehouse.duckdb")

    spy_flat = src.read(start="2024-01-02", end="2024-01-31").xs("SPY", axis=1, level=-1)
    sink.write("spy_daily", spy_flat, mode="overwrite")
    initial_len = len(sink.read("spy_daily"))

    # Repeat sync with overlapping range — upsert must dedup, no row growth.
    sink.write("spy_daily", spy_flat, mode="upsert")
    assert len(sink.read("spy_daily")) == initial_len
    sink.close()
