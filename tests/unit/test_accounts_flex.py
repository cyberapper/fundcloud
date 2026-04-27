"""Tests for the IB Flex Query CSV parser.

Every test uses synthetic ``ClientAccountID`` values (``U_TEST_0001`` /
``U_TEST_0002``) and round-number AUMs — no real broker data lands in
fixtures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------- helpers


def _nav_section(account: str = "U_TEST_0001") -> str:
    """Tiny NAV section: 3 daily rows, base USD."""
    return (
        '"ClientAccountID","CurrencyPrimary","ReportDate","Cash","Total"\n'
        f'"{account}","USD","20240102","100","1000"\n'
        f'"{account}","USD","20240103","100","1010"\n'
        f'"{account}","USD","20240104","100","1025"\n'
    )


def _cash_tx_section(
    account: str = "U_TEST_0001",
    *,
    extra_rows: list[str] | None = None,
) -> str:
    """Tiny cash-tx section: one deposit + one dividend (filtered out)."""
    rows = [
        f'"{account}","USD","1","20240102","500","Deposits/Withdrawals"',
        f'"{account}","USD","1","20240103","12.5","Dividends"',
    ]
    if extra_rows:
        rows.extend(extra_rows)
    body = "\n".join(rows)
    return (
        f'"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"\n{body}\n'
    )


# ---------------------------------------------------------------- order tolerance


def test_parser_handles_nav_then_cash_tx() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    text = _nav_section() + _cash_tx_section()
    result = parse_flex_csv(text)
    assert len(result.nav) == 3
    assert list(result.nav.columns) == ["account_id", "currency", "aum"]
    assert result.nav.index.name == "date"
    assert len(result.cash_transactions) == 1  # deposit only; dividend filtered
    assert result.cash_transactions["flow_type_raw"].iloc[0] == "Deposits/Withdrawals"


def test_parser_handles_cash_tx_then_nav() -> None:
    """Reversed section order must produce identical output."""
    from fundcloud.accounts._flex import parse_flex_csv

    forward = parse_flex_csv(_nav_section() + _cash_tx_section())
    reverse = parse_flex_csv(_cash_tx_section() + _nav_section())

    pd.testing.assert_frame_equal(forward.nav.sort_index(), reverse.nav.sort_index())
    pd.testing.assert_frame_equal(
        forward.cash_transactions.sort_index(),
        reverse.cash_transactions.sort_index(),
    )


# ---------------------------------------------------------------- robustness


def test_extra_columns_are_tolerated() -> None:
    """Unknown columns survive in `.sections["nav"]` but don't break normalization."""
    from fundcloud.accounts._flex import parse_flex_csv

    text = (
        '"ClientAccountID","CurrencyPrimary","ReportDate","Cash","Total","Crypto","FutureColumn"\n'
        '"U_TEST_0001","USD","20240102","100","1000","50","alpha"\n'
        '"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"\n'
        '"U_TEST_0001","USD","1","20240102","500","Deposits/Withdrawals"\n'
    )
    result = parse_flex_csv(text)
    assert "Crypto" in result.sections["nav"].columns
    assert "FutureColumn" in result.sections["nav"].columns
    # Normalized .nav still has only the canonical columns.
    assert list(result.nav.columns) == ["account_id", "currency", "aum"]


def test_missing_total_raises() -> None:
    """No `Total` → section can't be classified as NAV → require_nav fails.

    The path also emits a ``UserWarning`` for the unclassified section
    before raising; both are part of the contract.
    """
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    text = (
        '"ClientAccountID","CurrencyPrimary","ReportDate","Cash"\n'
        '"U_TEST_0001","USD","20240102","100"\n'
    )
    with pytest.warns(UserWarning, match="Unrecognised"), pytest.raises(MalformedDataError):
        parse_flex_csv(text)


def test_missing_datetime_raises() -> None:
    """No `Date/Time` → section can't be classified as cash-tx; with
    strict_unknown_sections=True the parser raises on the unknown
    section before checking require_cash_tx.
    """
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    text = _nav_section() + (
        '"ClientAccountID","CurrencyPrimary","FXRateToBase","Amount","Type"\n'
        '"U_TEST_0001","USD","1","500","Deposits/Withdrawals"\n'
    )
    with pytest.raises(MalformedDataError):
        parse_flex_csv(text, require_cash_tx=True, strict_unknown_sections=True)


def test_unknown_section_warns_by_default() -> None:
    """A future Flex section type warns rather than hard-erroring."""
    from fundcloud.accounts._flex import parse_flex_csv

    text = _nav_section() + (
        '"ClientAccountID","Symbol","Quantity","Side"\n"U_TEST_0001","AAPL","100","BUY"\n'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = parse_flex_csv(text)
    assert any("Unrecognised Flex CSV section" in str(w.message) for w in caught)
    assert "unknown_1" in result.sections


def test_unknown_section_strict_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    text = _nav_section() + (
        '"ClientAccountID","Symbol","Quantity","Side"\n"U_TEST_0001","AAPL","100","BUY"\n'
    )
    with pytest.raises(MalformedDataError, match="Unrecognised"):
        parse_flex_csv(text, strict_unknown_sections=True)


# ---------------------------------------------------------------- multi-account


def test_multiple_account_ids_visible() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    text = (
        '"ClientAccountID","CurrencyPrimary","ReportDate","Cash","Total"\n'
        '"U_TEST_0001","USD","20240102","100","1000"\n'
        '"U_TEST_0002","USD","20240102","200","2000"\n'
        '"U_TEST_0001","USD","20240103","100","1010"\n'
    )
    result = parse_flex_csv(text)
    assert sorted(result.nav["account_id"].unique().tolist()) == [
        "U_TEST_0001",
        "U_TEST_0002",
    ]


# ---------------------------------------------------------------- date / format variants


def test_datetime_with_time_component() -> None:
    """yyyyMMdd;HHmmss carries the time portion through."""
    from fundcloud.accounts._flex import parse_flex_csv

    text = _nav_section() + (
        '"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"\n'
        '"U_TEST_0001","USD","1","20240102;143015","500","Deposits/Withdrawals"\n'
    )
    result = parse_flex_csv(text)
    ts = result.cash_transactions.index[0]
    assert ts == pd.Timestamp("2024-01-02 14:30:15")


def test_unquoted_variant() -> None:
    """Flex export with quoting disabled still parses."""
    from fundcloud.accounts._flex import parse_flex_csv

    text = (
        "ClientAccountID,CurrencyPrimary,ReportDate,Cash,Total\n"
        "U_TEST_0001,USD,20240102,100,1000\n"
        "U_TEST_0001,USD,20240103,100,1010\n"
    )
    result = parse_flex_csv(text)
    assert len(result.nav) == 2
    assert result.nav["aum"].iloc[1] == 1010


def test_crlf_and_trailing_whitespace() -> None:
    """Windows line endings and stray spaces don't break the parser."""
    from fundcloud.accounts._flex import parse_flex_csv

    text = (
        '"ClientAccountID" , "CurrencyPrimary","ReportDate","Cash","Total"\r\n'
        '"U_TEST_0001","USD","20240102","100","1000"\r\n'
    )
    result = parse_flex_csv(text)
    assert len(result.nav) == 1
    assert result.nav["aum"].iloc[0] == 1000


# ---------------------------------------------------------------- requirements toggles


def test_nav_only_export_with_require_cash_tx_false() -> None:
    from fundcloud.accounts._flex import parse_flex_csv

    result = parse_flex_csv(_nav_section(), require_cash_tx=False)
    assert len(result.nav) == 3
    assert result.cash_transactions.empty
    assert list(result.cash_transactions.columns) == [
        "account_id",
        "currency",
        "fx_rate_to_base",
        "amount_native",
        "flow_type_raw",
        "description",
        "transaction_id",
    ]


def test_require_cash_tx_raises_when_absent() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    with pytest.raises(MalformedDataError, match="Cash Transactions"):
        parse_flex_csv(_nav_section(), require_cash_tx=True)


def test_require_nav_raises_when_absent() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    with pytest.raises(MalformedDataError, match="NAV in Base"):
        parse_flex_csv(_cash_tx_section())


def test_no_header_at_all_raises() -> None:
    from fundcloud.accounts._flex import parse_flex_csv
    from fundcloud.errors import MalformedDataError

    with pytest.raises(MalformedDataError, match="No 'ClientAccountID' header"):
        parse_flex_csv("garbage,more garbage\n1,2,3\n")


# ---------------------------------------------------------------- deposit-type variants


def test_deposit_type_phrasing_variants() -> None:
    """IB has used 'Deposits/Withdrawals', 'Deposits', and 'Withdrawals'."""
    from fundcloud.accounts._flex import parse_flex_csv

    text = _nav_section() + (
        '"ClientAccountID","CurrencyPrimary","FXRateToBase","Date/Time","Amount","Type"\n'
        '"U_TEST_0001","USD","1","20240102","100","Deposits/Withdrawals"\n'
        '"U_TEST_0001","USD","1","20240103","200","Deposits"\n'
        '"U_TEST_0001","USD","1","20240104","-50","Withdrawals"\n'
        '"U_TEST_0001","USD","1","20240105","5","Dividends"\n'
        '"U_TEST_0001","USD","1","20240106","-1","Broker Interest Paid"\n'
    )
    result = parse_flex_csv(text)
    # Three flow rows kept, two non-flow rows dropped.
    assert len(result.cash_transactions) == 3
    assert set(result.cash_transactions["flow_type_raw"]) == {
        "Deposits/Withdrawals",
        "Deposits",
        "Withdrawals",
    }


# ---------------------------------------------------------------- real-sample smoke


def test_real_sample_smoke() -> None:
    """Smoke test against the real anonymised export, gated by file presence."""
    sample = Path("temp/ib_fundcloud_example.csv")
    if not sample.exists():
        pytest.skip("temp/ib_fundcloud_example.csv not present (intentional)")
    from fundcloud.accounts._flex import parse_flex_csv

    result = parse_flex_csv(sample)
    assert len(result.nav) >= 200, "expected at least ~262 daily NAV rows"
    assert "account_id" in result.nav.columns
    # At least one deposit/withdrawal post-filter.
    assert len(result.cash_transactions) >= 1
