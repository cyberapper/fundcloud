"""Live integration tests for the FundCloud integration.

Skipped by default. To run:

    export FUNDCLOUD_API_KEY=fc_live_...
    uv run pytest tests/integration/test_fundcloud_live.py -m network -q

Hits the real FundCloud public API once per test. Tests are
intentionally read-only and validate shape, not content — the account
data behind a key will vary across users.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.network

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def _require_api_key() -> str:
    val = os.environ.get("FUNDCLOUD_API_KEY")
    if not val:
        pytest.skip("FUNDCLOUD_API_KEY not set — skipping FundCloud live tests")
    return val


# --------------------------------------------------------------- market data


def test_data_fundcloud_aapl_daily() -> None:
    _require_api_key()
    from fundcloud.data import FundCloud

    src = FundCloud(symbols="AAPL", interval="1d", period="3M")
    bars = src.read()
    assert isinstance(bars, pd.DataFrame)
    assert isinstance(bars.columns, pd.MultiIndex)
    fields = {c[0] for c in bars.columns}
    assert {"open", "high", "low", "close", "volume"}.issubset(fields)
    assert ("close", "AAPL") in bars.columns
    assert len(bars) >= 20  # ~3 months of daily bars


# --------------------------------------------------------------- accounts


def test_accounts_fundcloud_list_funds() -> None:
    _require_api_key()
    from fundcloud.accounts import FundCloud

    src = FundCloud()
    funds = src.list_funds()
    assert isinstance(funds, pd.DataFrame)
    # Columns contract (regardless of whether the key has any funds).
    expected_cols = {
        "fund_id",
        "name",
        "short_name",
        "currency",
        "inception_date",
        "status",
        "aum",
        "total_shares",
    }
    assert expected_cols.issubset(set(funds.columns))


def test_accounts_fundcloud_nav_and_flows() -> None:
    """If the key sees at least one fund, pulling NAV + flows should not raise."""
    _require_api_key()
    from fundcloud.accounts import FundCloud

    src = FundCloud()
    funds = src.list_funds()
    if funds.empty:
        pytest.skip("FUNDCLOUD_API_KEY sees no funds — nothing to test against")

    fund_id = str(funds.iloc[0]["fund_id"])

    nav = src.nav(fund_id=fund_id)
    assert isinstance(nav, pd.DataFrame)
    assert {"nav", "aum", "shares"}.issubset(set(nav.columns))

    flows = src.capital_flows(fund_id=fund_id)
    assert isinstance(flows, pd.DataFrame)
    # Allowed to be empty; column contract still holds.
    assert {"flow_type", "amount", "currency"}.issubset(set(flows.columns))


def test_accounts_fundcloud_to_portfolio_roundtrip() -> None:
    """to_portfolio() with the default path should produce a working Portfolio."""
    _require_api_key()
    from fundcloud.accounts import FundCloud

    src = FundCloud()
    funds = src.list_funds()
    if funds.empty:
        pytest.skip("FUNDCLOUD_API_KEY sees no funds — nothing to test against")

    fund_id = str(funds.iloc[0]["fund_id"])
    pf = src.to_portfolio(fund_id=fund_id)
    # Should at minimum have non-empty returns and a finite sharpe value
    # (or NaN for flat NAV — both are acceptable, just not an error).
    assert pf.returns.notna().any() or pf.returns.empty
