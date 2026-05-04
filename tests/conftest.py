"""Shared fixtures for the unit test suite."""

from __future__ import annotations

# Force matplotlib onto a non-interactive backend before anyone imports pyplot.
# Windows CI runners default to TkAgg but ship without Tcl/Tk, so the first
# `plt.figure()` blows up with `_tkinter.TclError`. The Agg backend is headless
# and bundled with matplotlib, so it's safe everywhere.
import matplotlib

matplotlib.use("Agg")

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def returns_series(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(datetime(2021, 1, 4), periods=252)
    return pd.Series(rng.normal(0.0005, 0.01, size=252), index=idx, name="strat")


@pytest.fixture
def returns_panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(datetime(2021, 1, 4), periods=252)
    cols = ["AAA", "BBB", "CCC"]
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(252, len(cols))),
        index=idx,
        columns=cols,
    )


@pytest.fixture
def synthetic_ib_full_year_csv() -> str:
    """Synthetic IB Flex Query export covering ~one trading year.

    Replaces the previously-skipped real-anonymised export. The shape is
    chosen to satisfy the same correctness invariants the original sample
    locked in:

    * **>=262 daily NAV rows**, USD base, ``Cash`` and ``Total`` *differ*
      on the late rows (so a parser that read ``Cash`` instead of
      ``Total`` would yield a wildly different last AUM).
    * **Exactly two HKD deposits** with realistic FXRateToBase values:
      HKD 50,000 @ 0.12889 -> USD ~6,445 and HKD 184,000 @ 0.12739 ->
      USD ~23,440.
    * **One dividend row** that the parser must filter out of the
      capital-flows view but keep in the cash-tx ledger.

    The account ID and amounts are synthetic placeholders — no broker
    data is committed (per project convention: never put real account
    data in committed artifacts).
    """
    account = "U_TEST_FULLYEAR"
    start = pd.Timestamp("2024-01-02")
    biz_days = pd.bdate_range(start, periods=265)

    # NAV: starts at 1,000, ramps so late rows have Total ~33,000 with
    # Cash deliberately negative (proves the parser reads Total, not Cash).
    nav_rows = ['"ClientAccountID","CurrencyPrimary","ReportDate","Cash","Total"']
    n = len(biz_days)
    for i, ts in enumerate(biz_days):
        progress = i / (n - 1)
        total = 1_000.0 + progress * 32_000.0  # 1,000 -> 33,000
        # Cash starts equal to Total, swings negative late (margin loan)
        cash = total - progress * 50_000.0  # late rows: ~ -17k
        nav_rows.append(f'"{account}","USD","{ts.strftime("%Y%m%d")}","{cash:.2f}","{total:.2f}"')

    # Cash transactions: two HKD deposits + one dividend (filtered out).
    cash_rows = [
        '"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"',
        f'"{account}","HKD","0.12889","20240115","50000","Deposits/Withdrawals"',
        f'"{account}","HKD","0.12739","20240611","184000","Deposits/Withdrawals"',
        f'"{account}","USD","1.0","20240701","12.50","Dividends"',
    ]
    return "\n".join(nav_rows) + "\n\n" + "\n".join(cash_rows) + "\n"


@pytest.fixture
def ohlcv_panel(seed: int) -> pd.DataFrame:
    """Tiny two-asset OHLCV frame, MultiIndex on the columns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(datetime(2022, 1, 3), periods=50)
    data = {}
    for sym in ("AAA", "BBB"):
        close = 100 + np.cumsum(rng.normal(0, 1, size=len(idx)))
        data[("open", sym)] = close + rng.normal(0, 0.1, size=len(idx))
        data[("high", sym)] = close + np.abs(rng.normal(0, 0.5, size=len(idx)))
        data[("low", sym)] = close - np.abs(rng.normal(0, 0.5, size=len(idx)))
        data[("close", sym)] = close
        data[("volume", sym)] = rng.integers(1_000_000, 5_000_000, size=len(idx)).astype(float)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)
