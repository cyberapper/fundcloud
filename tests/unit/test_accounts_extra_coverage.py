"""Coverage-fill tests for the IB / FundCloud / shared client modules.

Targets the un-covered branches that the canonical happy-path tests in
:mod:`test_accounts_ib`, :mod:`test_accounts_flex`,
:mod:`test_accounts_fundcloud`, :mod:`test_data_fundcloud`, and
:mod:`test_clients_fundcloud` don't reach — error mapping, empty-frame
fall-throughs, file-object / bytes input variants, ambiguity errors,
display-name fallbacks.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

# --------------------------------------------------------------------- _clients


def test_client_close_and_context_manager() -> None:
    """close() / __enter__ / __exit__ exercise the lifecycle hooks."""
    from fundcloud._clients import fundcloud as client_mod

    closed: list[bool] = []

    class _FakeHttp:
        def close(self) -> None:
            closed.append(True)

    client = client_mod.FundCloudClient(api_key="fc_test")
    client._http = _FakeHttp()  # type: ignore[assignment]
    with client as c:
        assert c is client
    assert closed == [True]


def test_client_maps_401_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud._clients import fundcloud as client_mod
    from fundcloud.errors import AuthError

    httpx = pytest.importorskip("httpx")

    class _FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def fake_get_json(self: Any, path: str, params: Any = None) -> Any:
        raise httpx.HTTPStatusError(
            "401 Unauthorized",
            request=httpx.Request("GET", f"https://fundcloud.example.com{path}"),
            response=httpx.Response(401),
        )

    monkeypatch.setattr(client_mod.HttpClient, "get_json", fake_get_json)
    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(AuthError, match="authentication failed"):
        client.get("/funds")


def test_client_maps_403_to_quota_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud._clients import fundcloud as client_mod
    from fundcloud.errors import QuotaError

    httpx = pytest.importorskip("httpx")

    def fake_get_json(self: Any, path: str, params: Any = None) -> Any:
        raise httpx.HTTPStatusError(
            "403 Forbidden",
            request=httpx.Request("GET", f"https://fundcloud.example.com{path}"),
            response=httpx.Response(403),
        )

    monkeypatch.setattr(client_mod.HttpClient, "get_json", fake_get_json)
    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(QuotaError, match="access denied or quota"):
        client.get("/funds")


def test_client_propagates_5xx_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Status codes outside the 401/403/404 mapping bubble up raw."""
    from fundcloud._clients import fundcloud as client_mod

    httpx = pytest.importorskip("httpx")

    def fake_get_json(self: Any, path: str, params: Any = None) -> Any:
        raise httpx.HTTPStatusError(
            "500 ISE",
            request=httpx.Request("GET", f"https://fundcloud.example.com{path}"),
            response=httpx.Response(500),
        )

    monkeypatch.setattr(client_mod.HttpClient, "get_json", fake_get_json)
    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/funds")


def test_client_maps_transient_to_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud._clients import fundcloud as client_mod
    from fundcloud.data._http import TransientHttpError
    from fundcloud.errors import TransientError

    def fake_get_json(self: Any, path: str, params: Any = None) -> Any:
        raise TransientHttpError("retries exhausted")

    monkeypatch.setattr(client_mod.HttpClient, "get_json", fake_get_json)
    client = client_mod.FundCloudClient(api_key="fc_test")
    with pytest.raises(TransientError, match="transient"):
        client.get("/funds")


def test_has_more_pages_falls_back_to_false_for_unknown_meta() -> None:
    """When neither has_next nor page+total_pages are present, return False."""
    from fundcloud._clients.fundcloud import _has_more_pages

    assert _has_more_pages({"random_key": "value"}, current_page=1) is False
    assert _has_more_pages({}, current_page=1) is False


# --------------------------------------------------------------------- _flex


def _nav_header() -> str:
    return '"ClientAccountID","CurrencyPrimary","ReportDate","Cash","Total"'


def _cash_header() -> str:
    return '"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"'


def test_flex_accepts_bytes_with_bom() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    raw = (_nav_header() + '\n"U_A","USD","20240102","100","1000"\n').encode("utf-8-sig")
    out = parse_flex_csv(raw)
    assert len(out.nav) == 1


def test_flex_accepts_file_like_object() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    csv = _nav_header() + '\n"U_A","USD","20240102","100","1000"\n'
    out = parse_flex_csv(io.StringIO(csv))
    assert len(out.nav) == 1


def test_flex_accepts_file_like_returning_bytes() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    csv = (_nav_header() + '\n"U_A","USD","20240102","100","1000"\n').encode("utf-8-sig")
    out = parse_flex_csv(io.BytesIO(csv))
    assert len(out.nav) == 1


def test_flex_string_path_that_does_not_exist_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    with pytest.raises(FileNotFoundError, match="looks like a path"):
        parse_flex_csv("definitely_not_a_real_file.csv")


def test_flex_unsupported_source_type_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    with pytest.raises(TypeError, match="unsupported source type"):
        parse_flex_csv(42)  # type: ignore[arg-type]


def test_flex_missing_currency_primary_raises() -> None:
    """`_normalize_nav` requires CurrencyPrimary; classification only requires
    ReportDate + Total + Cash, so a section can pass classification but fail
    normalization when CurrencyPrimary is missing."""
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    # Drop CurrencyPrimary; NAV classification still triggers on ReportDate+Total+Cash.
    csv = '"ClientAccountID","ReportDate","Cash","Total"\n"U_A","20240102","100","1000"\n'
    with pytest.raises(MalformedDataError, match="CurrencyPrimary"):
        parse_flex_csv(csv)


def test_flex_invalid_report_date_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    csv = _nav_header() + '\n"U_A","USD","NOT_A_DATE","100","1000"\n'
    with pytest.raises(MalformedDataError, match="ReportDate"):
        parse_flex_csv(csv)


def test_flex_invalid_amount_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + _cash_header()
        + '\n"U_A","USD","1.0","20240103","not-a-number","Deposits"\n'
    )
    with pytest.raises(MalformedDataError, match="Amount"):
        parse_flex_csv(csv)


def test_flex_invalid_datetime_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + _cash_header()
        + '\n"U_A","USD","1.0","BOGUS_DT","100.0","Deposits"\n'
    )
    with pytest.raises(MalformedDataError, match="Date/Time"):
        parse_flex_csv(csv)


def test_flex_unknown_section_warns_by_default() -> None:
    """An unrecognised section alongside a real NAV section emits a warning
    but doesn't kill the parse."""
    from fundcloud.accounts._flex import parse_flex_csv

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + '"ClientAccountID","SomeOtherCol1","SomeOtherCol2"\n'
        + '"U_A","val","val"\n'
    )
    with pytest.warns(UserWarning, match="Unrecognised"):
        out = parse_flex_csv(csv)
    assert "unknown_1" in out.sections


def test_flex_unknown_section_strict_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + '"ClientAccountID","SomeOtherCol1","SomeOtherCol2"\n'
        + '"U_A","val","val"\n'
    )
    with pytest.raises(MalformedDataError, match="Unrecognised"):
        parse_flex_csv(csv, strict_unknown_sections=True)


def test_flex_require_nav_raises_when_missing() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    # Cash-tx-only export.
    csv = _cash_header() + '\n"U_A","USD","1.0","20240103","100","Deposits"\n'
    with pytest.raises(MalformedDataError, match="NAV"):
        parse_flex_csv(csv, require_nav=True)


def test_flex_require_cash_tx_raises_when_missing() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    csv = _nav_header() + '\n"U_A","USD","20240102","100","1000"\n'
    with pytest.raises(MalformedDataError, match="Cash"):
        parse_flex_csv(csv, require_cash_tx=True)


def test_flex_filters_non_deposit_withdrawal_types() -> None:
    """Dividends / broker-interest rows are silently dropped."""
    from fundcloud.accounts._flex import parse_flex_csv

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + _cash_header()
        + '\n"U_A","USD","1.0","20240103","2.50","Dividends"\n'
        + '"U_A","USD","1.0","20240104","100.0","Deposits"\n'
    )
    out = parse_flex_csv(csv)
    # Only the Deposits row survived.
    assert len(out.cash_transactions) == 1
    assert out.cash_transactions["amount_native"].iloc[0] == 100.0


# --------------------------------------------------------------------- IB extras


def test_ib_multi_currency_nav_raises() -> None:
    """A single ClientAccountID with multiple `CurrencyPrimary` values is rejected."""
    from fundcloud.accounts import IB
    from fundcloud.errors import MalformedDataError

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + '"U_A","EUR","20240103","100","900"\n'
    )
    src = IB(text=csv)
    with pytest.raises(MalformedDataError, match="multiple"):
        src.list_funds()


def test_ib_multiple_accounts_raise_ambiguous_when_no_id_given() -> None:
    from fundcloud.accounts import IB
    from fundcloud.errors import AmbiguousError

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + '"U_B","USD","20240102","200","2000"\n'
    )
    src = IB(text=csv)
    with pytest.raises(AmbiguousError, match="Multiple accounts"):
        src.nav()


def test_ib_no_accounts_raises_not_found() -> None:
    """An empty NAV section with no default account → NotFoundError on lookup.

    Construct a normal IB then clear the NAV frame to exercise the
    `_resolve_account` empty-list path.
    """
    from fundcloud.accounts import IB
    from fundcloud.errors import NotFoundError

    csv = _nav_header() + '\n"U_A","USD","20240102","100","1000"\n'
    src = IB(text=csv)
    # Clear NAV after construction — simulates an export where the NAV
    # section is present but contains zero rows post-filter.
    src._nav_df = src._nav_df.iloc[0:0]
    src._default_account_id = None
    with pytest.raises(NotFoundError, match="No accounts"):
        src.nav()


def test_ib_account_id_constructor_default_used() -> None:
    """Constructor `account_id=` is the fallback when no per-call id is set."""
    from fundcloud.accounts import IB

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240102","100","1000"\n'
        + '"U_B","USD","20240102","200","2000"\n'
    )
    src = IB(text=csv, account_id="U_B")
    nav = src.nav()
    assert nav["aum"].iloc[0] == 2000.0


def test_ib_capital_flows_with_no_tx_returns_empty_frame() -> None:
    from fundcloud.accounts import IB

    csv = _nav_header() + '\n"U_A","USD","20240102","100","1000"\n'
    src = IB(text=csv)
    flows = src.capital_flows()
    assert flows.empty
    assert "flow_type" in flows.columns


def test_ib_capital_flows_filters_by_window() -> None:
    from fundcloud.accounts import IB

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240101","100","1000"\n'
        + _cash_header()
        + '\n"U_A","USD","1.0","20240105","100","Deposits"\n'
        + '"U_A","USD","1.0","20240601","200","Deposits"\n'
    )
    src = IB(text=csv)
    flows = src.capital_flows(start="2024-01-01", end="2024-02-01")
    assert len(flows) == 1


def test_ib_nav_window_filtering() -> None:
    from fundcloud.accounts import IB

    csv = (
        _nav_header()
        + '\n"U_A","USD","20240101","100","1000"\n'
        + '"U_A","USD","20240601","100","1100"\n'
    )
    src = IB(text=csv)
    nav = src.nav(start="2024-04-01", end="2024-12-31", adjust_for_flows=False)
    assert len(nav) == 1


def test_ib_display_name_uses_resolved_account() -> None:
    """`_display_name` mentions the resolved account id."""
    from fundcloud.accounts import IB

    csv = _nav_header() + '\n"U_A","USD","20240102","100","1000"\n'
    src = IB(text=csv)
    name = src._display_name(fund_id=None, account_id="U_A")
    assert "U_A" in name


def test_ib_display_name_without_account_id() -> None:
    from fundcloud.accounts import IB

    csv = _nav_header() + '\n"U_A","USD","20240102","100","1000"\n'
    src = IB(text=csv)
    name = src._display_name(fund_id=None, account_id=None)
    # Falls back to constant "IB" when no account is available.
    assert name == "IB"


# --------------------------------------------------------------------- FundCloud provider


def _funds_payload_min() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "f1",
                "name": "Demo Fund",
                "short_name": "DF",
                "currency": "USD",
                "inception_date": "2023-01-01",
                "status": "ACTIVE",
                "aum": 1_000_000.0,
                "total_shares": 10_000.0,
                "fund_type": "fund",
                "info": {},
                "capital_flow_approval_required": False,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2024-04-24T00:00:00Z",
            },
        ],
        "meta": {"has_next": False},
    }


class _FakeClient:
    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        self._responses = responses

    def get(self, path: str, params: Any = None) -> Any:
        return self._responses.get((path, "single"), {"data": [], "meta": {}})

    def get_paginated(self, path: str, params: Any = None) -> Any:
        payload = self.get(path, params)
        yield from payload.get("data", [])


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUNDCLOUD_API_KEY", "fc_test_unit")


def test_fundcloud_list_accounts_filter_by_unknown_fund_raises() -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import NotFoundError

    fake = _FakeClient({("/funds", "single"): _funds_payload_min()})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]
    with pytest.raises(NotFoundError, match="not visible"):
        src.list_accounts(fund_id="UNKNOWN")


def test_fundcloud_list_accounts_returns_empty_when_no_breakdown() -> None:
    """A NAV payload with no `account_breakdown` yields an empty accounts frame."""
    from fundcloud.accounts import fundcloud as fc_mod

    nav_no_break = {
        "data": [
            {
                "fund_id": "f1",
                "date": "2024-04-22",
                "nav": 100.0,
                "aum": 1_000_000.0,
                "shares": 10_000.0,
                "fill_type": "actual",
                "account_breakdown": [],
            }
        ],
        "meta": {"has_next": False},
    }
    fake = _FakeClient({
        ("/funds", "single"): _funds_payload_min(),
        ("/funds/f1/nav", "single"): nav_no_break,
    })
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]
    accts = src.list_accounts()
    assert accts.empty
    assert "account_id" in accts.columns


def test_fundcloud_resolve_fund_no_funds_visible_raises() -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import NotFoundError

    empty = {"data": [], "meta": {"has_next": False}}
    fake = _FakeClient({("/funds", "single"): empty})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]
    with pytest.raises(NotFoundError, match="No funds visible"):
        src.nav()


def test_fundcloud_resolve_fund_ambiguous_when_multiple() -> None:
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import AmbiguousError

    multi = {
        "data": [
            {
                "id": "f1",
                "name": "A",
                "short_name": "A",
                "currency": "USD",
                "inception_date": "2023-01-01",
                "status": "ACTIVE",
                "aum": 1.0,
                "total_shares": 1.0,
            },
            {
                "id": "f2",
                "name": "B",
                "short_name": "B",
                "currency": "USD",
                "inception_date": "2023-01-01",
                "status": "ACTIVE",
                "aum": 1.0,
                "total_shares": 1.0,
            },
        ],
        "meta": {"has_next": False},
    }
    fake = _FakeClient({("/funds", "single"): multi})
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]
    with pytest.raises(AmbiguousError, match="Multiple funds"):
        src.nav()


def test_fundcloud_resolve_fund_account_id_lookup_raises_for_unknown() -> None:
    """`account_id=` that doesn't map to any visible fund → NotFoundError."""
    from fundcloud.accounts import fundcloud as fc_mod
    from fundcloud.errors import NotFoundError

    multi = {
        "data": [
            {
                "id": "f1",
                "name": "A",
                "short_name": "A",
                "currency": "USD",
                "inception_date": "2023-01-01",
                "status": "ACTIVE",
                "aum": 1.0,
                "total_shares": 1.0,
            },
            {
                "id": "f2",
                "name": "B",
                "short_name": "B",
                "currency": "USD",
                "inception_date": "2023-01-01",
                "status": "ACTIVE",
                "aum": 1.0,
                "total_shares": 1.0,
            },
        ],
        "meta": {"has_next": False},
    }
    nav_payload = {
        "data": [
            {
                "fund_id": "f1",
                "date": "2024-01-01",
                "nav": 100.0,
                "aum": 1.0,
                "shares": 1.0,
                "fill_type": "actual",
                "account_breakdown": [{"account_id": "ACC_1", "nav": 1.0}],
            }
        ],
        "meta": {"has_next": False},
    }
    fake = _FakeClient({
        ("/funds", "single"): multi,
        ("/funds/f1/nav", "single"): nav_payload,
        ("/funds/f2/nav", "single"): nav_payload,
    })
    src = fc_mod.FundCloud()
    src._client = fake  # type: ignore[assignment]
    with pytest.raises(NotFoundError, match="not visible"):
        src.nav(account_id="ACC_NOT_THERE")


def test_fundcloud_display_name_uses_short_name() -> None:
    """Once `list_funds` populates the cache, `_display_name` returns short_name."""
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds", "single"): _funds_payload_min()})
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]
    src.list_funds()  # populate cache

    name = src._display_name(fund_id="f1", account_id=None)
    assert "DF" in name


def test_fundcloud_display_name_appends_account_id() -> None:
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds", "single"): _funds_payload_min()})
    src = fc_mod.FundCloud(fund_id="f1")
    src._client = fake  # type: ignore[assignment]
    src.list_funds()

    name = src._display_name(fund_id="f1", account_id="ACC_X")
    assert "ACC_X" in name


def test_fundcloud_display_name_falls_back_when_id_unknown() -> None:
    """Cache miss falls back to the base implementation."""
    from fundcloud.accounts import fundcloud as fc_mod

    fake = _FakeClient({("/funds", "single"): _funds_payload_min()})
    src = fc_mod.FundCloud(fund_id="f_not_in_cache")
    src._client = fake  # type: ignore[assignment]
    src.list_funds()  # cache loaded with f1, not the constructor default

    name = src._display_name(fund_id=None, account_id=None)
    # Base impl returns a string — exact value not contractual; just check no crash.
    assert isinstance(name, str)


# --------------------------------------------------------------------- data/fundcloud


def test_data_fundcloud_candle_without_timestamp_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candles missing `timestamp` are silently dropped, not crashed on."""
    from fundcloud.data import fundcloud as fc_data

    payload = {
        "symbol": "AAPL",
        "candles": [
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100},
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
            },
        ],
    }
    monkeypatch.setattr(fc_data.FundCloudClient, "get", lambda self, path, params=None: payload)
    out = fc_data.FundCloud("AAPL").read()
    assert len(out) == 1


def test_data_fundcloud_invalid_timestamp_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_data
    from fundcloud.errors import MalformedDataError

    payload = {
        "symbol": "AAPL",
        "candles": [
            {
                "timestamp": "BOGUS",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
            },
        ],
    }
    monkeypatch.setattr(fc_data.FundCloudClient, "get", lambda self, path, params=None: payload)
    with pytest.raises(MalformedDataError, match="timestamp"):
        fc_data.FundCloud("AAPL").read()


def test_data_fundcloud_candle_field_none_passes_as_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing numeric fields become NaN — only un-coerceable values raise."""
    from fundcloud.data import fundcloud as fc_data

    payload = {
        "symbol": "AAPL",
        "candles": [
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "open": None,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
            },
        ],
    }
    monkeypatch.setattr(fc_data.FundCloudClient, "get", lambda self, path, params=None: payload)
    out = fc_data.FundCloud("AAPL").read()
    assert len(out) == 1


def test_data_fundcloud_columns_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass `columns=` to slice down which OHLCV fields are returned."""
    from fundcloud.data import fundcloud as fc_data

    payload = {
        "symbol": "AAPL",
        "candles": [
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
            },
        ],
    }
    monkeypatch.setattr(fc_data.FundCloudClient, "get", lambda self, path, params=None: payload)
    out = fc_data.FundCloud("AAPL").read(columns=["close"])
    fields = [c[0] for c in out.columns]
    assert fields == ["close"]


def test_data_fundcloud_non_dict_payload_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-dict OHLCV payload yields an empty frame, not a crash."""
    from fundcloud.data import fundcloud as fc_data

    monkeypatch.setattr(
        fc_data.FundCloudClient, "get", lambda self, path, params=None: ["not", "a", "dict"]
    )
    out = fc_data.FundCloud("AAPL").read()
    assert out.empty
