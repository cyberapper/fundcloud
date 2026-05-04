"""Tests for `fundcloud.accounts.IB` — schema, sign / FX conversion, defaults."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------- helpers


def _build_csv(
    *,
    nav_rows: list[tuple[str, str, str, str, str]] | None = None,
    cash_rows: list[tuple[str, str, str, str, str, str]] | None = None,
) -> str:
    """Compose a synthetic Flex CSV from row tuples.

    nav row tuple: (account, currency, yyyyMMdd, cash, total)
    cash row tuple: (account, currency, fx_rate, dt, amount, type)
    """
    parts = []
    if nav_rows:
        parts.append('"ClientAccountID","CurrencyPrimary","ReportDate","Cash","Total"')
        parts.extend(f'"{a}","{c}","{d}","{cash}","{total}"' for a, c, d, cash, total in nav_rows)
    if cash_rows:
        parts.append(
            '"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"'
        )
        parts.extend(
            f'"{a}","{c}","{fx}","{dt}","{amt}","{ty}"' for a, c, fx, dt, amt, ty in cash_rows
        )
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- construction


def test_construction_requires_exactly_one_source() -> None:
    from fundcloud.accounts import IB

    with pytest.raises(ValueError, match="exactly one"):
        IB()  # zero sources

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    with pytest.raises(ValueError, match="exactly one"):
        IB(path="x.csv", text=csv)  # two sources


def test_from_string_factory() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    src = IB.from_string(csv)
    nav = src.nav()
    assert len(nav) == 1
    assert nav["aum"].iloc[0] == 1000


# ---------------------------------------------------------------- list_funds / list_accounts


def test_list_funds_one_row_per_account_id() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0001", "USD", "20240103", "100", "1010"),
            ("U_TEST_0002", "USD", "20240102", "200", "2000"),
        ],
    )
    src = IB(text=csv)
    funds = src.list_funds()
    assert sorted(funds["fund_id"].tolist()) == ["U_TEST_0001", "U_TEST_0002"]
    # AUM is the LATEST per account.
    row1 = funds[funds["fund_id"] == "U_TEST_0001"].iloc[0]
    assert row1["aum"] == 1010
    assert row1["total_shares"] == 1.0
    assert row1["currency"] == "USD"


def test_list_accounts_mirrors_list_funds_for_ib() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    src = IB(text=csv)
    accounts = src.list_accounts()
    assert accounts.iloc[0]["account_id"] == "U_TEST_0001"
    assert accounts.iloc[0]["fund_id"] == "U_TEST_0001"  # single-tier
    assert accounts.iloc[0]["external_account_id"] == "U_TEST_0001"


def test_list_accounts_filter_by_unknown_fund_raises() -> None:
    from fundcloud.accounts import IB
    from fundcloud.errors import NotFoundError

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    src = IB(text=csv)
    with pytest.raises(NotFoundError):
        src.list_accounts(fund_id="U_DOES_NOT_EXIST")


# ---------------------------------------------------------------- nav


def test_nav_raw_matches_total_column() -> None:
    """adjust_for_flows=False just exposes the Total column verbatim."""
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0001", "USD", "20240103", "100", "1010"),
            ("U_TEST_0001", "USD", "20240104", "100", "1025"),
        ],
    )
    src = IB(text=csv)
    nav = src.nav(adjust_for_flows=False)
    assert nav["aum"].tolist() == [1000.0, 1010.0, 1025.0]
    assert (nav["shares"] == 1.0).all()
    assert (nav["fill_type"] == "actual").all()


def test_nav_adjust_for_flows_smooths_deposit() -> None:
    """A deposit causes a raw AUM jump; adjust_for_flows replays without it."""
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0001", "USD", "20240103", "100", "1010"),
            # +500 deposited on day 4; raw AUM jumps to 1525 (1010 base + 500 deposit
            # + 15 of investment growth).
            ("U_TEST_0001", "USD", "20240104", "100", "1525"),
            ("U_TEST_0001", "USD", "20240105", "100", "1540"),
        ],
        cash_rows=[("U_TEST_0001", "USD", "1", "20240104", "500", "Deposits/Withdrawals")],
    )
    src = IB(text=csv)

    raw = src.nav(adjust_for_flows=False)
    adj = src.nav(adjust_for_flows=True)

    # Raw shows the jump (~50% in one day).
    assert raw["aum"].iloc[2] / raw["aum"].iloc[1] > 1.4
    # Adjusted curve removes the deposit; day-4 should reflect only the
    # investment return (~1% on the previous balance).
    assert 0.99 < adj["aum"].iloc[2] / adj["aum"].iloc[1] < 1.05


# ---------------------------------------------------------------- capital_flows sign + FX


def test_capital_flows_positive_amount_maps_to_injection() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")],
        cash_rows=[
            ("U_TEST_0001", "USD", "1", "20240102", "500", "Deposits/Withdrawals"),
        ],
    )
    src = IB(text=csv)
    flows = src.capital_flows()
    assert flows.iloc[0]["flow_type"] == "INJECTION"
    assert flows.iloc[0]["amount"] == 500.0


def test_capital_flows_negative_amount_maps_to_withdrawal() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")],
        cash_rows=[
            ("U_TEST_0001", "USD", "1", "20240102", "-300", "Deposits/Withdrawals"),
        ],
    )
    src = IB(text=csv)
    flows = src.capital_flows()
    assert flows.iloc[0]["flow_type"] == "WITHDRAWAL"
    assert flows.iloc[0]["amount"] == 300.0  # always positive


def test_capital_flows_fx_to_base() -> None:
    """Non-USD flow with FXRateToBase != 1 is multiplied through."""
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")],
        cash_rows=[
            # 50,000 HKD * 0.13 fx = $6,500 USD-base
            ("U_TEST_0001", "HKD", "0.13", "20240102", "50000", "Deposits/Withdrawals"),
        ],
    )
    src = IB(text=csv)
    flows = src.capital_flows()
    assert flows.iloc[0]["amount"] == pytest.approx(6500.0)
    # Currency on the output is the BASE currency (USD), not the native HKD.
    assert flows.iloc[0]["currency"] == "USD"


# ---------------------------------------------------------------- to_portfolio defaults


def test_to_portfolio_defaults_to_aum_modified_dietz() -> None:
    """IB overrides FundCloud's nav_per_share + total_return defaults."""
    from fundcloud.accounts import IB
    from fundcloud.portfolio import Portfolio

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0001", "USD", "20240103", "100", "1010"),
            ("U_TEST_0001", "USD", "20240104", "100", "1525"),
        ],
        cash_rows=[("U_TEST_0001", "USD", "1", "20240104", "500", "Deposits/Withdrawals")],
    )
    src = IB(text=csv)
    pf = src.to_portfolio()
    assert isinstance(pf, Portfolio)
    # The deposit must NOT show up as a return — Modified Dietz on day 4 should
    # be a small positive number (~1.5%), not the +50% raw pct_change implies.
    assert -0.05 < pf.returns.iloc[-1] < 0.05


def test_to_portfolio_basis_method_validation() -> None:
    """Inherited from BaseAccountProvider — incompatible pairs raise."""
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0001", "USD", "20240103", "100", "1010"),
        ],
    )
    src = IB(text=csv)
    with pytest.raises(ValueError, match="aum"):
        src.to_portfolio(basis="aum", method="total_return")


# ---------------------------------------------------------------- account resolution


def test_resolve_account_single_account_auto_picks() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")],
    )
    src = IB(text=csv)
    assert src._resolve_account(None, None) == "U_TEST_0001"


def test_resolve_account_ambiguous_raises() -> None:
    from fundcloud.accounts import IB
    from fundcloud.errors import AmbiguousError

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0002", "USD", "20240102", "200", "2000"),
        ],
    )
    src = IB(text=csv)
    with pytest.raises(AmbiguousError):
        src._resolve_account(None, None)


def test_account_id_kwarg_resolves_directly() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0002", "USD", "20240102", "200", "2000"),
        ],
    )
    src = IB(text=csv)
    nav = src.nav(account_id="U_TEST_0002")
    assert (nav.shape[0] == 1) and (nav["aum"].iloc[0] == 2000)


def test_constructor_account_id_default() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(
        nav_rows=[
            ("U_TEST_0001", "USD", "20240102", "100", "1000"),
            ("U_TEST_0002", "USD", "20240102", "200", "2000"),
        ],
    )
    src = IB(text=csv, account_id="U_TEST_0001")
    nav = src.nav()  # no kwargs — uses constructor default
    assert nav["aum"].iloc[0] == 1000


# ---------------------------------------------------------------- positions / trades


def test_positions_empty_for_v01() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    src = IB(text=csv)
    positions = src.positions()
    assert positions.empty
    assert "symbol" in positions.columns
    assert "market_value" in positions.columns


def test_trades_empty_for_v01() -> None:
    from fundcloud.accounts import IB

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    src = IB(text=csv)
    trades = src.trades()
    assert trades.empty
    assert trades.index.name == "trade_date"


# ---------------------------------------------------------------- lazy registry


def test_ib_lazy_registry() -> None:
    from fundcloud.accounts import IB as LazyImported  # noqa: N811
    from fundcloud.accounts.ib import IB as DirectImported  # noqa: N811

    assert LazyImported is DirectImported


# ---------------------------------------------------------------- multi-file aggregation


def test_files_aggregate_per_section(tmp_path) -> None:
    """Two files, each with one account; result lists both."""
    from fundcloud.accounts import IB

    f1 = tmp_path / "year1.csv"
    f1.write_text(_build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")]))
    f2 = tmp_path / "year2.csv"
    f2.write_text(_build_csv(nav_rows=[("U_TEST_0002", "USD", "20240102", "200", "2000")]))

    src = IB(files=[f1, f2])
    funds = src.list_funds()
    assert sorted(funds["fund_id"].tolist()) == ["U_TEST_0001", "U_TEST_0002"]


# ---------------------------------------------------------------- regression: real sample


def test_synthetic_full_year_full_coverage_and_both_deposits(
    synthetic_ib_full_year_csv: str,
) -> None:
    """Lock in three correctness invariants on a synthetic full-year export:

    1. ``nav()`` reads the ``Total`` column, not ``Cash`` — verified on a
       date where the two columns differ significantly.
    2. ``capital_flows()`` returns BOTH HKD deposits, not just the one
       inside an arbitrary 1-year-back window.
    3. The default period covers the full CSV, not a 1-year-back slice.

    The fixture lives in ``tests/conftest.py`` and replaces a previous
    file-gated test that depended on a real anonymised broker export
    under ``temp/``.
    """
    from fundcloud.accounts import IB

    src = IB(text=synthetic_ib_full_year_csv)

    # 1. Full period — every NAV row in the CSV is visible by default.
    nav = src.nav(adjust_for_flows=False)
    assert len(nav) >= 260, f"expected ~262 NAV rows, got {len(nav)}"

    # 2. Both deposits visible — no arbitrary date filter.
    flows = src.capital_flows()
    assert len(flows) == 2, f"expected exactly 2 deposits in this sample, got {len(flows)}"
    # Both are INJECTIONs, both converted to USD-base (the NAV's currency).
    assert (flows["flow_type"] == "INJECTION").all()
    assert (flows["currency"] == "USD").all()
    # HKD 50,000 @ 0.12889 -> USD ~6,445; HKD 184,000 @ 0.12739 -> USD ~23,440.
    # Loose bounds tolerate any later fixture refresh.
    sorted_amounts = flows["amount"].sort_values().tolist()
    assert 6_000 < sorted_amounts[0] < 7_000  # HKD 50k  -> USD ~6,445
    assert 22_000 < sorted_amounts[1] < 24_000  # HKD 184k -> USD ~23,440

    # 3. Reads `Total`, not `Cash`. The synthetic late-year rows have
    # Cash deliberately negative while Total > 30k — proving the parser
    # picks the right column.
    last_aum = float(nav["aum"].iloc[-1])
    assert last_aum > 30_000, (
        f"AUM at end of year should be > $30k (the synthetic Total), got "
        f"${last_aum:.2f} - if this is negative, the parser is reading "
        f"Cash instead of Total."
    )


# ---------------------------------------------------------------- protocol satisfaction


def test_ib_satisfies_account_provider_protocol() -> None:
    from fundcloud.accounts import IB
    from fundcloud.accounts._base import AccountProvider

    csv = _build_csv(nav_rows=[("U_TEST_0001", "USD", "20240102", "100", "1000")])
    src = IB(text=csv)
    assert isinstance(src, AccountProvider)
