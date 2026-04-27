"""Tests for fundcloud.accounts — AccountProvider + FundCloud provider + flow handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------- helpers / fixtures


def _funds_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "f1",
                "name": "Global Multi-Asset Fund",
                "short_name": "GMAF",
                "currency": "USD",
                "inception_date": "2023-01-01",
                "status": "ACTIVE",
                "aum": 2_500_000_000.0,
                "total_shares": 25_000_000.0,
                "fund_type": "hedge_fund",
                "info": {},
                "capital_flow_approval_required": False,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2024-04-24T00:00:00Z",
            },
        ],
        "meta": {"total": 1, "page": 1, "page_size": 20, "total_pages": 1, "has_next": False},
    }


def _nav_payload_page1() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "n1",
                "fund_id": "f1",
                "date": "2024-04-22",
                "nav": 100.0,
                "aum": 1_000_000.0,
                "shares": 10_000.0,
                "daily_return": 0.0,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [
                    {
                        "account_id": "ACC_1",
                        "account_name": "Primary",
                        "nav": 600_000.0,
                        "fill_type": "actual",
                    },
                    {
                        "account_id": "ACC_2",
                        "account_name": "Secondary",
                        "nav": 400_000.0,
                        "fill_type": "actual",
                    },
                ],
                "created_at": "2024-04-22T00:00:00Z",
                "updated_at": "2024-04-22T00:00:00Z",
            },
            {
                "id": "n2",
                "fund_id": "f1",
                "date": "2024-04-23",
                "nav": 101.0,
                "aum": 1_010_000.0,
                "shares": 10_000.0,
                "daily_return": 0.01,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-04-23T00:00:00Z",
                "updated_at": "2024-04-23T00:00:00Z",
            },
        ],
        "meta": {"total": 3, "page": 1, "page_size": 2, "total_pages": 2, "has_next": True},
    }


def _nav_payload_page2() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "n3",
                "fund_id": "f1",
                "date": "2024-04-24",
                "nav": 103.0,
                "aum": 1_030_000.0,
                "shares": 10_000.0,
                "daily_return": 0.0198,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-04-24T00:00:00Z",
                "updated_at": "2024-04-24T00:00:00Z",
            },
        ],
        "meta": {"total": 3, "page": 2, "page_size": 2, "total_pages": 2, "has_next": False},
    }


def _flows_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "c1",
                "fund_id": "f1",
                "account_id": "ACC_1",
                "flow_type": "INJECTION",
                "amount": 5000.0,
                "flow_date": "2024-04-22",
                "notes": "",
                "created_at": "2024-04-22T00:00:00Z",
                "updated_at": "2024-04-22T00:00:00Z",
            },
            {
                "id": "c2",
                "fund_id": "f1",
                "account_id": "ACC_1",
                "flow_type": "DISTRIBUTION",
                "amount": 1000.0,
                "flow_date": "2024-04-23",
                "notes": "quarterly distribution",
                "created_at": "2024-04-23T00:00:00Z",
                "updated_at": "2024-04-23T00:00:00Z",
            },
        ],
        "meta": {"total": 2, "page": 1, "page_size": 20, "total_pages": 1, "has_next": False},
    }


def _positions_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "p1",
                "fund_id": "f1",
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "asset_type": "EQUITY",
                "quantity": 100.0,
                "avg_cost": 150.0,
                "current_price": 180.0,
                "market_value": 18000.0,
                "currency": "USD",
                "weight": 0.3,
                "unrealized_pnl": 3000.0,
                "unrealized_pnl_percent": 20.0,
                "account_name": "Primary",
                "external_account_id": "ACC_1",
            },
            {
                "id": "p2",
                "fund_id": "f1",
                "symbol": "MSFT",
                "name": "Microsoft",
                "asset_type": "EQUITY",
                "quantity": 50.0,
                "avg_cost": 300.0,
                "current_price": 320.0,
                "market_value": 16000.0,
                "currency": "USD",
                "weight": 0.26,
                "unrealized_pnl": 1000.0,
                "unrealized_pnl_percent": 6.67,
                "account_name": "Secondary",
                "external_account_id": "ACC_2",
            },
        ],
        "meta": {"total": 2, "page": 1, "page_size": 20, "total_pages": 1, "has_next": False},
    }


def _trades_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "t1",
                "fund_id": "f1",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10.0,
                "price": 180.0,
                "amount": 1800.0,
                "currency": "USD",
                "trade_date": "2024-04-20",
                "settlement_date": "2024-04-22",
                "status": "CLOSED",
                "broker": "IBKR",
                "fee": 1.5,
                "account_name": "Primary",
                "external_account_id": "ACC_1",
            },
        ],
        "meta": {"total": 1, "page": 1, "page_size": 20, "total_pages": 1, "has_next": False},
    }


class _FakeClient:
    """Drop-in replacement for FundCloudClient that returns scripted payloads."""

    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        # key: (path, marker) — marker is 'page=<n>' or 'single'
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, path: str, params: Any = None) -> Any:
        params = dict(params or {})
        self.calls.append((path, params))
        # Try page-specific key first
        page = params.get("page", 1)
        key = (path, f"page={page}")
        if key in self._responses:
            return self._responses[key]
        return self._responses[(path, "single")]

    def get_paginated(self, path: str, params: Any = None) -> Any:
        """Drain across pages as the real client does."""
        current = dict(params or {})
        current.setdefault("page", 1)
        current.setdefault("page_size", 100)
        while True:
            payload = self.get(path, current)
            data = payload.get("data", [])
            yield from data
            meta = payload.get("meta") or {}
            if not meta.get("has_next", False):
                break
            current["page"] = int(current["page"]) + 1


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a dummy API key unless it explicitly deletes it."""
    monkeypatch.setenv("FUNDCLOUD_API_KEY", "fc_test_unit")


# ---------------------------------------------------------------- returns_from_nav


def test_returns_from_nav_none_matches_pct_change() -> None:
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series(
        [100.0, 101.0, 102.5, 101.5],
        index=pd.date_range("2024-01-01", periods=4),
    )
    r = returns_from_nav(nav, method="none")
    expected = nav.pct_change().iloc[1:]
    assert np.allclose(r.values, expected.values)


def test_returns_from_nav_total_return_adds_distribution() -> None:
    """NAV drops from 100→99 but $1 per-share was distributed → total return = 0."""
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series([100.0, 99.0, 100.0], index=pd.date_range("2024-01-01", periods=3))
    distrib = pd.Series([0.0, 1.0, 0.0], index=nav.index)
    r = returns_from_nav(nav, distributions=distrib, method="total_return")
    assert np.isclose(r.iloc[0], 0.0)
    # Day 3: pure pct_change (100-99)/99
    assert np.isclose(r.iloc[1], 1 / 99)


def test_returns_from_nav_modified_dietz_formula() -> None:
    """NAV 100 → 110, $5 injected mid-period → Mod Dietz: 5 / 102.5."""
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))
    flows = pd.Series([0.0, 5.0], index=nav.index)
    r = returns_from_nav(nav, capital_flows=flows, method="modified_dietz")
    assert np.isclose(r.iloc[0], 5 / 102.5)


def test_returns_from_nav_daily_twr_formula() -> None:
    """Same scenario, daily TWR: (110-100-5)/100."""
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))
    flows = pd.Series([0.0, 5.0], index=nav.index)
    r = returns_from_nav(nav, capital_flows=flows, method="daily_twr")
    assert np.isclose(r.iloc[0], 5 / 100)


def test_returns_from_nav_modified_dietz_without_flows_raises() -> None:
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="requires"):
        returns_from_nav(nav, method="modified_dietz")


def test_returns_from_nav_unknown_method_raises() -> None:
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="unknown method"):
        returns_from_nav(nav, method="bogus")  # type: ignore[arg-type]


def test_returns_from_nav_series_name_is_returns() -> None:
    from fundcloud.metrics import returns_from_nav

    nav = pd.Series([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2))
    assert returns_from_nav(nav).name == "returns"


# ---------------------------------------------------------------- Portfolio.from_nav


def test_portfolio_from_nav_constructs_with_sharpe() -> None:
    from fundcloud.portfolio import Portfolio

    nav = pd.Series(
        [100.0, 101.0, 103.0, 102.0, 105.0],
        index=pd.date_range("2024-01-01", periods=5),
    )
    pf = Portfolio.from_nav(nav, name="test")
    assert pf.name == "test"
    assert not pf.returns.empty
    # Sharpe should be a real number
    assert np.isfinite(pf.sharpe(periods_per_year=252))


def test_portfolio_from_nav_accepts_dataframe_with_nav_column() -> None:
    from fundcloud.portfolio import Portfolio

    idx = pd.date_range("2024-01-01", periods=3)
    nav_df = pd.DataFrame(
        {"nav": [100.0, 101.0, 102.0], "aum": [1000.0, 1010.0, 1020.0]}, index=idx
    )
    pf = Portfolio.from_nav(nav_df)
    assert not pf.returns.empty
    # Returns should come from the 'nav' column, not 'aum' (they happen to be equal here)
    assert pf.returns.iloc[0] > 0


def test_portfolio_from_nav_stashes_trades_and_positions() -> None:
    from fundcloud.portfolio import Portfolio

    nav = pd.Series([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2))
    trades = pd.DataFrame({"symbol": ["AAPL"], "qty": [10]})
    positions = pd.DataFrame({"symbol": ["AAPL"], "quantity": [10]})

    pf = Portfolio.from_nav(nav, trades=trades, positions=positions)
    assert pf._source_trades is trades  # type: ignore[attr-defined]
    assert pf._source_positions is positions  # type: ignore[attr-defined]


# ---------------------------------------------------------------- flow-signing helpers


def test_flows_to_per_share_distributions_only_distribution_counts() -> None:
    from fundcloud.accounts._base import _flows_to_per_share_distributions

    dates = pd.date_range("2024-01-01", periods=3)
    shares = pd.Series([1000.0, 1000.0, 1000.0], index=dates)
    flows = pd.DataFrame(
        {
            "flow_type": ["INJECTION", "DISTRIBUTION", "WITHDRAWAL"],
            "amount": [5000.0, 500.0, 2000.0],
        },
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    d = _flows_to_per_share_distributions(flows, shares)
    # Only the distribution ($500 on day 2, 1000 shares → $0.5/share)
    assert np.isclose(d.loc["2024-01-02"], 0.5)
    assert np.isclose(d.loc["2024-01-01"], 0.0)
    assert np.isclose(d.loc["2024-01-03"], 0.0)


def test_flows_to_signed_aum_series_signs_correctly() -> None:
    from fundcloud.accounts._base import _flows_to_signed_aum_series

    dates = pd.date_range("2024-01-01", periods=4)
    flows = pd.DataFrame(
        {
            "flow_type": ["INJECTION", "WITHDRAWAL", "DISTRIBUTION"],
            "amount": [10000.0, 3000.0, 500.0],
        },
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    signed = _flows_to_signed_aum_series(flows, dates)
    assert np.isclose(signed.loc["2024-01-01"], 10000.0)
    assert np.isclose(signed.loc["2024-01-02"], -3000.0)
    assert np.isclose(signed.loc["2024-01-03"], -500.0)
    assert np.isclose(signed.loc["2024-01-04"], 0.0)


def test_weekend_flow_attributed_to_next_nav_date() -> None:
    """Saturday flow should land on the next Monday NAV date."""
    from fundcloud.accounts._base import _flows_to_per_share_distributions

    # NAV only on Friday and Monday
    nav_dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])
    shares = pd.Series([1000.0, 1000.0], index=nav_dates)
    # Distribution on Saturday
    weekend = pd.DataFrame(
        {"flow_type": ["DISTRIBUTION"], "amount": [300.0]},
        index=pd.DatetimeIndex(["2024-01-06"]),
    )
    d = _flows_to_per_share_distributions(weekend, shares)
    assert np.isclose(d.loc["2024-01-08"], 0.3)
    assert np.isclose(d.loc["2024-01-05"], 0.0)


# ---------------------------------------------------------------- FundCloud provider


def test_list_funds_paginates_and_returns_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds", "single"): _funds_payload()})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    funds = src.list_funds()
    assert list(funds.columns) == [
        "fund_id",
        "name",
        "short_name",
        "currency",
        "inception_date",
        "status",
        "aum",
        "total_shares",
        "fund_type",
        "info",
    ]
    assert funds.loc[0, "fund_id"] == "f1"
    assert funds.loc[0, "short_name"] == "GMAF"


def test_list_accounts_reads_account_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({
        ("/funds", "single"): _funds_payload(),
        ("/funds/f1/nav", "single"): _nav_payload_page1(),
    })
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    accts = src.list_accounts()
    assert len(accts) == 2
    assert sorted(accts["account_id"].tolist()) == ["ACC_1", "ACC_2"]
    assert set(accts["fund_id"].unique()) == {"f1"}
    assert set(accts["fund_name"].unique()) == {"Global Multi-Asset Fund"}


def test_nav_drains_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({
        ("/funds/f1/nav", "page=1"): _nav_payload_page1(),
        ("/funds/f1/nav", "page=2"): _nav_payload_page2(),
    })
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    nav = src.nav()
    assert len(nav) == 3
    assert list(nav.columns) == ["nav", "aum", "shares", "daily_return", "fill_type"]
    assert nav.index.name == "date"
    # Sorted
    assert nav.index.is_monotonic_increasing


def test_get_paginated_bails_out_on_runaway_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that lies about pagination (has_next=True forever) must not
    cause a memory blow-up. ``_MAX_PAGES`` caps the loop and surfaces a
    :class:`TransientError`.
    """
    from fundcloud._clients import fundcloud as client_mod
    from fundcloud.errors import TransientError

    def fake_get(self: Any, path: str, params: Any = None) -> Any:
        # Always claim there's a next page — the production code must
        # cap the loop instead of trusting this.
        return {"data": [{"id": "x"}], "meta": {"has_next": True}}

    monkeypatch.setattr(client_mod.FundCloudClient, "get", fake_get)
    # Drop the cap so the test runs in milliseconds, not minutes.
    monkeypatch.setattr(client_mod, "_MAX_PAGES", 5)

    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(TransientError, match="did not terminate"):
        list(client.get_paginated("/x"))


def test_get_paginated_falls_back_to_total_pages_when_has_next_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live API sometimes returns meta without ``has_next``; pagination
    must still drain via ``page < total_pages``.
    """
    from fundcloud._clients import fundcloud as client_mod

    pages = {
        1: {
            "data": [{"id": "a"}, {"id": "b"}],
            "meta": {"page": 1, "page_size": 2, "total": 5, "total_pages": 3},  # NO has_next
        },
        2: {
            "data": [{"id": "c"}, {"id": "d"}],
            "meta": {"page": 2, "page_size": 2, "total": 5, "total_pages": 3},
        },
        3: {
            "data": [{"id": "e"}],
            "meta": {"page": 3, "page_size": 2, "total": 5, "total_pages": 3},
        },
    }

    def fake_get(self: Any, path: str, params: Any = None) -> Any:
        return pages[int((params or {}).get("page", 1))]

    monkeypatch.setattr(client_mod.FundCloudClient, "get", fake_get)

    client = client_mod.FundCloudClient(api_key="fc_test")
    items = list(client.get_paginated("/x"))
    assert [i["id"] for i in items] == ["a", "b", "c", "d", "e"]


def test_nav_aggregation_is_always_daily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggregation must be ``daily`` whether or not account_id is set."""
    from fundcloud.accounts import fundcloud as fc_mod

    captured: list[dict[str, Any]] = []

    def fake_get_paginated(self: Any, path: str, params: Any = None) -> Any:
        captured.append(dict(params or {}))
        return iter([])

    fake = _FakeClient({})
    monkeypatch.setattr(type(fake), "get_paginated", fake_get_paginated, raising=False)
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    src.nav()
    src.nav(account_id="ACC_1")

    assert captured[0]["aggregation"] == "daily"
    assert captured[1]["aggregation"] == "daily"
    assert captured[1]["account_id"] == "ACC_1"


def test_nav_default_start_is_one_year_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit start, nav() sends today − 1 year as ``start_date``."""
    from fundcloud.accounts import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get_paginated(self: Any, path: str, params: Any = None) -> Any:
        captured.update(params or {})
        return iter([])

    fake = _FakeClient({})
    monkeypatch.setattr(type(fake), "get_paginated", fake_get_paginated, raising=False)
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    src.nav()
    sent = pd.Timestamp(captured["start_date"])
    expected = pd.Timestamp.now().normalize() - pd.DateOffset(years=1)
    # Allow a 1-day fudge for clock drift across midnight.
    assert abs((sent - expected).days) <= 1


def test_nav_adjust_for_flows_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get_paginated(self: Any, path: str, params: Any = None) -> Any:
        captured.update(params or {})
        return iter([])

    fake = _FakeClient({})
    monkeypatch.setattr(type(fake), "get_paginated", fake_get_paginated, raising=False)
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    src.nav()
    assert captured["adjust_for_flows"] == "true"


def test_nav_adjust_for_flows_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get_paginated(self: Any, path: str, params: Any = None) -> Any:
        captured.update(params or {})
        return iter([])

    fake = _FakeClient({})
    monkeypatch.setattr(type(fake), "get_paginated", fake_get_paginated, raising=False)
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    src.nav(adjust_for_flows=False)
    assert captured["adjust_for_flows"] == "false"


def test_capital_flows_default_start_is_one_year_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get_paginated(self: Any, path: str, params: Any = None) -> Any:
        captured.update(params or {})
        return iter([])

    fake = _FakeClient({})
    monkeypatch.setattr(type(fake), "get_paginated", fake_get_paginated, raising=False)
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    src.capital_flows()
    sent = pd.Timestamp(captured["start_date"])
    expected = pd.Timestamp.now().normalize() - pd.DateOffset(years=1)
    assert abs((sent - expected).days) <= 1


def test_to_portfolio_passes_adjust_for_flows_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """to_portfolio must opt out of server-side adjustment to avoid double-counting."""
    from fundcloud.accounts import fundcloud as fc_mod

    seen: list[dict[str, Any]] = []

    def fake_get_paginated(self: Any, path: str, params: Any = None) -> Any:
        seen.append({"path": path, "params": dict(params or {})})
        # Provide minimal NAV + empty flows so to_portfolio can build a Portfolio.
        if "/nav" in path:
            return iter([
                {
                    "id": "n1",
                    "fund_id": "f1",
                    "date": "2024-01-01",
                    "nav": 100.0,
                    "aum": 100_000.0,
                    "shares": 1000.0,
                    "daily_return": 0.0,
                    "is_aggregated": True,
                    "fill_type": "actual",
                    "account_breakdown": [],
                },
                {
                    "id": "n2",
                    "fund_id": "f1",
                    "date": "2024-01-02",
                    "nav": 101.0,
                    "aum": 101_000.0,
                    "shares": 1000.0,
                    "daily_return": 0.01,
                    "is_aggregated": True,
                    "fill_type": "actual",
                    "account_breakdown": [],
                },
            ])
        return iter([])

    fake = _FakeClient({})
    monkeypatch.setattr(type(fake), "get_paginated", fake_get_paginated, raising=False)
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    src.to_portfolio()

    nav_calls = [s for s in seen if "/nav" in s["path"]]
    assert nav_calls, "expected to_portfolio to call nav()"
    assert nav_calls[0]["params"]["adjust_for_flows"] == "false"


def test_capital_flows_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds/f1/capital-flows", "single"): _flows_payload()})
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    flows = src.capital_flows()
    assert list(flows.columns) == [
        "flow_type",
        "amount",
        "currency",
        "account_id",
        "notes",
    ]
    assert set(flows["flow_type"]) == {"INJECTION", "DISTRIBUTION"}
    # amounts are positive (direction comes from flow_type)
    assert (flows["amount"] > 0).all()


def test_positions_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds/f1/positions", "single"): _positions_payload()})
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    pos = src.positions()
    assert len(pos) == 2
    assert "symbol" in pos.columns
    assert "market_value" in pos.columns
    assert "account_id" in pos.columns


def test_trades_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds/f1/trades", "single"): _trades_payload()})
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    trades = src.trades()
    assert len(trades) == 1
    assert trades.index.name == "trade_date"
    assert "side" in trades.columns


def test_resolve_fund_auto_single(monkeypatch: pytest.MonkeyPatch) -> None:
    """One visible fund → resolve_fund auto-picks it."""
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds", "single"): _funds_payload()})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    resolved = src._resolve_fund(None)
    assert resolved == "f1"


def test_resolve_fund_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import AmbiguousError

    two_funds = {
        "data": [
            {**_funds_payload()["data"][0], "id": "f1", "name": "Alpha"},
            {**_funds_payload()["data"][0], "id": "f2", "name": "Beta"},
        ],
        "meta": {"total": 2, "page": 1, "page_size": 20, "total_pages": 1, "has_next": False},
    }
    fake = _FakeClient({("/funds", "single"): two_funds})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    with pytest.raises(AmbiguousError, match="Multiple funds"):
        src._resolve_fund(None)


def test_resolve_fund_via_account_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """When only account_id is passed, _resolve_fund looks up the parent fund."""
    from fundcloud.accounts import fundcloud as fc_mod

    two_funds = {
        "data": [
            {**_funds_payload()["data"][0], "id": "f1", "name": "Alpha"},
            {**_funds_payload()["data"][0], "id": "f2", "name": "Beta"},
        ],
        "meta": {"total": 2, "page": 1, "page_size": 20, "total_pages": 1, "has_next": False},
    }
    # Each fund's NAV breakdown contains different accounts.
    f1_nav = {
        "data": [
            {
                **_nav_payload_page1()["data"][0],
                "fund_id": "f1",
                "account_breakdown": [
                    {
                        "account_id": "ACC_A1",
                        "account_name": "Alpha-1",
                        "nav": 100.0,
                        "fill_type": "actual",
                    },
                ],
            }
        ],
        "meta": {"total": 1, "page": 1, "page_size": 1, "total_pages": 1, "has_next": False},
    }
    f2_nav = {
        "data": [
            {
                **_nav_payload_page1()["data"][0],
                "fund_id": "f2",
                "account_breakdown": [
                    {
                        "account_id": "ACC_B1",
                        "account_name": "Beta-1",
                        "nav": 200.0,
                        "fill_type": "actual",
                    },
                ],
            }
        ],
        "meta": {"total": 1, "page": 1, "page_size": 1, "total_pages": 1, "has_next": False},
    }
    fake = _FakeClient({
        ("/funds", "single"): two_funds,
        ("/funds/f1/nav", "single"): f1_nav,
        ("/funds/f2/nav", "single"): f2_nav,
    })
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    # Despite multiple visible funds, account_id alone resolves cleanly.
    assert src._resolve_fund(None, account_id="ACC_A1") == "f1"
    assert src._resolve_fund(None, account_id="ACC_B1") == "f2"


def test_resolve_fund_via_unknown_account_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import NotFoundError

    fake = _FakeClient({
        ("/funds", "single"): _funds_payload(),
        ("/funds/f1/nav", "single"): _nav_payload_page1(),
    })
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    with pytest.raises(NotFoundError, match="not visible"):
        src._resolve_fund(None, account_id="DOES_NOT_EXIST")


def test_account_to_fund_map_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls with account_id only fetch list_accounts once."""
    from fundcloud.accounts import fundcloud as fc_mod

    list_accounts_calls = {"count": 0}

    fake = _FakeClient({
        ("/funds", "single"): _funds_payload(),
        ("/funds/f1/nav", "single"): _nav_payload_page1(),
    })

    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    orig = src.list_accounts

    def counting_list_accounts(*args: Any, **kw: Any) -> Any:
        list_accounts_calls["count"] += 1
        return orig(*args, **kw)

    src.list_accounts = counting_list_accounts  # type: ignore[method-assign]

    src._resolve_fund(None, account_id="ACC_1")
    src._resolve_fund(None, account_id="ACC_2")
    src._resolve_fund(None, account_id="ACC_1")  # repeat
    assert list_accounts_calls["count"] == 1


def test_nav_via_account_id_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: src.nav(account_id=X) auto-resolves the fund."""
    from fundcloud.accounts import fundcloud as fc_mod

    # Single-page NAV response with has_next=False so the fake client's
    # paginator terminates cleanly on the first iteration. Reusing
    # _nav_payload_page1 (which has has_next=True) as a "single" fallback
    # would cause _FakeClient.get_paginated to loop forever — see
    # _MAX_PAGES guard in production code.
    nav_complete = {
        **_nav_payload_page1(),
        "meta": {
            "total": 2,
            "page": 1,
            "page_size": 100,
            "total_pages": 1,
            "has_next": False,
        },
    }
    fake = _FakeClient({
        ("/funds", "single"): _funds_payload(),
        ("/funds/f1/nav", "single"): nav_complete,
    })
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    # Account ACC_1 lives in fund f1 (per _nav_payload_page1's account_breakdown).
    nav = src.nav(account_id="ACC_1")
    assert not nav.empty


def test_resolve_fund_none_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import NotFoundError

    no_funds = {
        "data": [],
        "meta": {"total": 0, "page": 1, "page_size": 20, "total_pages": 0, "has_next": False},
    }
    fake = _FakeClient({("/funds", "single"): no_funds})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]

    with pytest.raises(NotFoundError, match="No funds visible"):
        src._resolve_fund(None)


# ---------------------------------------------------------------- to_portfolio end-to-end


def test_to_portfolio_nav_per_share_total_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: NAV + distribution → Portfolio with correct returns."""
    from fundcloud.accounts import fundcloud as fc_mod

    nav_payload = {
        "data": [
            {
                "id": "n1",
                "fund_id": "f1",
                "date": "2024-01-01",
                "nav": 100.0,
                "aum": 1_000_000.0,
                "shares": 10_000.0,
                "daily_return": 0.0,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": "n2",
                "fund_id": "f1",
                "date": "2024-01-02",
                "nav": 99.0,
                "aum": 990_000.0,
                "shares": 10_000.0,
                "daily_return": -0.01,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
            {
                "id": "n3",
                "fund_id": "f1",
                "date": "2024-01-03",
                "nav": 100.0,
                "aum": 1_000_000.0,
                "shares": 10_000.0,
                "daily_return": 0.0101,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-03T00:00:00Z",
                "updated_at": "2024-01-03T00:00:00Z",
            },
        ],
        "meta": {"total": 3, "page": 1, "page_size": 100, "total_pages": 1, "has_next": False},
    }
    # Distribution of $10,000 on Jan 2 → $1 per share, which explains the -1% NAV drop
    flows_payload = {
        "data": [
            {
                "id": "c1",
                "fund_id": "f1",
                "account_id": "ACC_1",
                "flow_type": "DISTRIBUTION",
                "amount": 10_000.0,
                "flow_date": "2024-01-02",
                "notes": "",
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
        ],
        "meta": {"total": 1, "page": 1, "page_size": 100, "total_pages": 1, "has_next": False},
    }

    fake = _FakeClient({
        ("/funds/f1/nav", "single"): nav_payload,
        ("/funds/f1/capital-flows", "single"): flows_payload,
    })
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    pf = src.to_portfolio()
    # With distribution added back on Jan 2: return = (99 + 1 - 100) / 100 = 0
    assert np.isclose(pf.returns.iloc[0], 0.0, atol=1e-9)
    # Jan 3: pure pct_change (100 - 99) / 99
    assert np.isclose(pf.returns.iloc[1], 1 / 99, atol=1e-9)


def test_to_portfolio_aum_modified_dietz(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUM basis with injection flow → Modified Dietz."""
    from fundcloud.accounts import fundcloud as fc_mod

    nav_payload = {
        "data": [
            {
                "id": "n1",
                "fund_id": "f1",
                "date": "2024-01-01",
                "nav": 100.0,
                "aum": 100_000.0,
                "shares": 1000.0,
                "daily_return": 0.0,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": "n2",
                "fund_id": "f1",
                "date": "2024-01-02",
                "nav": 110.0,
                "aum": 110_000.0,
                "shares": 1000.0,
                "daily_return": 0.1,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
        ],
        "meta": {"total": 2, "page": 1, "page_size": 100, "total_pages": 1, "has_next": False},
    }
    flows_payload = {
        "data": [
            {
                "id": "c1",
                "fund_id": "f1",
                "account_id": "ACC_1",
                "flow_type": "INJECTION",
                "amount": 5000.0,
                "flow_date": "2024-01-02",
                "notes": "",
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
        ],
        "meta": {"total": 1, "page": 1, "page_size": 100, "total_pages": 1, "has_next": False},
    }

    fake = _FakeClient({
        ("/funds/f1/nav", "single"): nav_payload,
        ("/funds/f1/capital-flows", "single"): flows_payload,
    })
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    pf = src.to_portfolio(basis="aum", method="modified_dietz")
    # AUM 100k → 110k with 5k injection. Mod Dietz: (110k - 100k - 5k) / (100k + 2.5k) = 5000 / 102500
    expected = 5000 / 102_500
    assert np.isclose(pf.returns.iloc[0], expected, atol=1e-9)


def test_to_portfolio_basis_method_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """basis='nav_per_share' with method='modified_dietz' is an error."""
    from fundcloud.accounts import fundcloud as fc_mod

    nav_payload = {
        "data": [
            {
                "id": "n1",
                "fund_id": "f1",
                "date": "2024-01-01",
                "nav": 100.0,
                "aum": 100_000.0,
                "shares": 1000.0,
                "daily_return": 0.0,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": "n2",
                "fund_id": "f1",
                "date": "2024-01-02",
                "nav": 101.0,
                "aum": 101_000.0,
                "shares": 1000.0,
                "daily_return": 0.01,
                "is_aggregated": True,
                "fill_type": "actual",
                "account_breakdown": [],
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            },
        ],
        "meta": {"total": 2, "page": 1, "page_size": 100, "total_pages": 1, "has_next": False},
    }
    fake = _FakeClient({
        ("/funds/f1/nav", "single"): nav_payload,
        ("/funds/f1/capital-flows", "single"): {
            "data": [],
            "meta": {"total": 0, "page": 1, "page_size": 100, "total_pages": 0, "has_next": False},
        },
    })
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"nav_per_share.*only valid"):
        src.to_portfolio(basis="nav_per_share", method="modified_dietz")


# ---------------------------------------------------------------- error mapping


def test_missing_api_key_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import AuthError

    monkeypatch.delenv("FUNDCLOUD_API_KEY", raising=False)
    with pytest.raises(AuthError):
        fc_mod.FundCloud()


def test_http_status_401_maps_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud._clients import fundcloud as client_mod
    from fundcloud.errors import AuthError

    class _FakeResp:
        status_code = 401

    class _HTTPError(Exception):
        def __init__(self) -> None:
            self.response = _FakeResp()

    def fake_get_json(self: Any, url: str, *, params: Any = None) -> Any:
        import httpx

        raise httpx.HTTPStatusError("401", request=None, response=_FakeResp())  # type: ignore[arg-type]

    monkeypatch.setattr(client_mod.HttpClient, "get_json", fake_get_json)

    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(AuthError):
        client.get("/funds")


def test_http_status_404_maps_to_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud._clients import fundcloud as client_mod
    from fundcloud.errors import NotFoundError

    class _FakeResp:
        status_code = 404

    def fake_get_json(self: Any, url: str, *, params: Any = None) -> Any:
        import httpx

        raise httpx.HTTPStatusError("404", request=None, response=_FakeResp())  # type: ignore[arg-type]

    monkeypatch.setattr(client_mod.HttpClient, "get_json", fake_get_json)

    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(NotFoundError):
        client.get("/funds/unknown")


# ---------------------------------------------------------------- protocol satisfaction


def test_fundcloud_satisfies_account_provider_protocol() -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.accounts._base import AccountProvider

    src = fc_mod.FundCloud()
    assert isinstance(src, AccountProvider)


def test_fundcloud_accounts_lazy_registry() -> None:
    from fundcloud.accounts import FundCloud as LazyImported
    from fundcloud.accounts.fundcloud import FundCloud as DirectImported

    assert LazyImported is DirectImported
