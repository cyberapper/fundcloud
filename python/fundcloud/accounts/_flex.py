"""Interactive Brokers Flex Query CSV parser.

A Flex Query CSV export is a single file containing one or more concatenated
sections, each prefixed with its own header row. Two sections matter for
:class:`fundcloud.accounts.IB`:

* **NAV in Base** — daily AUM rows. Header columns include ``ClientAccountID``,
  ``ReportDate``, ``Cash``, ``Total`` (and ~94 others).
* **Cash Transactions** — event rows. Header columns include
  ``ClientAccountID``, ``Date/Time``, ``Amount``, ``Type``, ``FXRateToBase``.
  Capital flows are rows where ``Type`` is one of ``"Deposits/Withdrawals"``,
  ``"Deposits"``, or ``"Withdrawals"``; everything else is investment income.

Sections are concatenated with no blank-line separator. The parser detects
section boundaries by scanning for header rows whose first cell is literally
``"ClientAccountID"``, then classifies each block by inspecting which columns
the header carries (NOT by ordinal position) — so the parser tolerates
either order, additional columns the user opted into, and forward-compat
section types it doesn't recognise yet.
"""

from __future__ import annotations

import csv
import io
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

import pandas as pd

from fundcloud.errors import MalformedDataError

__all__ = ["FlexExport", "parse_flex_csv"]


# ---------------------------------------------------------------------------
# Public API


@dataclass(frozen=True, slots=True)
class FlexExport:
    """Parsed result of a Flex Query CSV.

    Attributes
    ----------
    nav
        Normalized NAV-in-Base frame. ``DatetimeIndex`` named ``date``.
        Columns: ``account_id``, ``currency``, ``aum``. Empty frame with
        the right columns if no NAV section was present.
    cash_transactions
        Normalized capital-flow frame, **pre-filtered** to deposits /
        withdrawals only. ``DatetimeIndex`` named ``flow_date``.
        Columns: ``account_id``, ``currency`` (native), ``fx_rate_to_base``,
        ``amount_native`` (signed: positive = deposit, negative =
        withdrawal), ``flow_type_raw``, ``description``,
        ``transaction_id``. Empty frame with the right columns if no
        cash-transaction section was present.
    sections
        Raw, pre-normalized DataFrame for each detected section, keyed
        by classification (``"nav"``, ``"cash_tx"``, or
        ``"unknown_<n>"`` for unrecognised sections). Useful for
        accessing columns the normalized views drop (e.g.
        ``Stock``/``Bonds`` breakdowns in NAV, dividend rows in
        cash transactions).
    """

    nav: pd.DataFrame
    cash_transactions: pd.DataFrame
    sections: Mapping[str, pd.DataFrame] = field(default_factory=dict)


def parse_flex_csv(
    source: str | Path | IO[str] | bytes,
    *,
    require_nav: bool = True,
    require_cash_tx: bool = False,
    strict_unknown_sections: bool = False,
) -> FlexExport:
    """Parse an IB Flex Query CSV into a :class:`FlexExport`.

    Parameters
    ----------
    source
        Filesystem path (``str`` or :class:`pathlib.Path`), a text-mode
        file-like object, raw ``bytes``, or an inline CSV string
        (detected by the presence of a newline character).
    require_nav
        Raise :class:`MalformedDataError` if no NAV section is detected.
        Default ``True`` because every IB Flex Query the library
        consumes today carries the NAV section.
    require_cash_tx
        Raise :class:`MalformedDataError` if no Cash Transactions
        section is detected. Default ``False`` — some users export
        NAV-only.
    strict_unknown_sections
        If ``True``, raise :class:`MalformedDataError` when a section's
        header doesn't match the NAV or Cash Transactions fingerprint.
        Default ``False``: emit a :class:`UserWarning` and stash the
        raw frame under ``sections["unknown_<n>"]`` so a future Flex
        section type doesn't hard-break the parse.

    Returns
    -------
    FlexExport

    Raises
    ------
    MalformedDataError
        If the file is structurally invalid (no header found, missing
        essential column in a recognised section, unparseable date in a
        required column, required section absent).
    """
    text = _read_text(source)
    raw_sections = _split_sections(text)
    if not raw_sections:
        msg = "No 'ClientAccountID' header row found in Flex CSV."
        raise MalformedDataError(msg)

    sections: dict[str, pd.DataFrame] = {}
    nav_blocks: list[pd.DataFrame] = []
    cash_blocks: list[pd.DataFrame] = []
    unknown_count = 0

    for header, rows in raw_sections:
        kind = _classify_section(header)
        df = pd.DataFrame.from_records(rows, columns=header)
        if kind == "nav":
            nav_blocks.append(df)
            sections["nav"] = (
                df if "nav" not in sections else pd.concat([sections["nav"], df], ignore_index=True)
            )
        elif kind == "cash_tx":
            cash_blocks.append(df)
            sections["cash_tx"] = (
                df
                if "cash_tx" not in sections
                else pd.concat([sections["cash_tx"], df], ignore_index=True)
            )
        else:
            unknown_count += 1
            key = f"unknown_{unknown_count}"
            sections[key] = df
            if strict_unknown_sections:
                msg = (
                    f"Unrecognised Flex CSV section. Header columns: "
                    f"{list(header)!r}. Expected NAV (with 'ReportDate' + "
                    f"'Total' + 'Cash') or Cash Transactions (with 'Date/Time' "
                    f"+ 'Amount' + 'Type')."
                )
                raise MalformedDataError(msg)
            warnings.warn(
                f"Unrecognised Flex CSV section with {len(header)} columns "
                f"(first 5: {list(header)[:5]!r}); stored under "
                f"sections[{key!r}] but not normalized.",
                UserWarning,
                stacklevel=2,
            )

    if require_nav and not nav_blocks:
        msg = "Flex CSV did not contain a NAV in Base section."
        raise MalformedDataError(msg)
    if require_cash_tx and not cash_blocks:
        msg = "Flex CSV did not contain a Cash Transactions section."
        raise MalformedDataError(msg)

    nav_df = (
        _normalize_nav(pd.concat(nav_blocks, ignore_index=True)) if nav_blocks else _empty_nav()
    )
    cash_df = (
        _normalize_cash_tx(pd.concat(cash_blocks, ignore_index=True))
        if cash_blocks
        else _empty_cash_tx()
    )

    return FlexExport(nav=nav_df, cash_transactions=cash_df, sections=sections)


# ---------------------------------------------------------------------------
# Source loading


def _read_text(source: str | Path | IO[str] | bytes) -> str:
    """Load CSV text from any of the accepted source types."""
    if isinstance(source, bytes):
        return source.decode("utf-8-sig")  # tolerate BOM
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8-sig")
    if isinstance(source, str):
        # Inline CSV content if it looks multi-line; otherwise treat as a path.
        if "\n" in source:
            return source
        path = Path(source)
        if path.is_file():
            return path.read_text(encoding="utf-8-sig")
        msg = (
            f"`source` looks like a path but does not exist: {source!r}. "
            f"Pass a Path, a text-mode file object, raw bytes, or inline "
            f"CSV content with newlines."
        )
        raise FileNotFoundError(msg)
    # Anything with .read() — TextIO, StringIO, BytesIO-decoded, etc.
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            return data.decode("utf-8-sig")
        return data
    msg = f"unsupported source type: {type(source).__name__}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Section detection + classification


_NAV_FINGERPRINT = {"ReportDate", "Total", "Cash"}
_CASH_TX_FINGERPRINT = {"Date/Time", "Amount", "Type"}

_HEADER_FIRST_CELL = "ClientAccountID"


def _split_sections(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Walk the CSV once; partition rows into (header, body) per section.

    A row whose first non-whitespace cell equals ``"ClientAccountID"``
    starts a new section. Rows before the first header are silently
    dropped (defensive against ``Include header and trailer records=Yes``).
    """
    # ``skipinitialspace`` lets users stick spaces between the comma and the
    # next quoted cell without the parser keeping the quote chars as part of
    # the field. Tolerates Flex exports that have been hand-edited or that
    # round-tripped through a tool that injects whitespace.
    reader = csv.reader(io.StringIO(text), skipinitialspace=True)
    sections: list[tuple[list[str], list[list[str]]]] = []
    current_header: list[str] | None = None
    current_rows: list[list[str]] = []
    for row in reader:
        if not row:
            continue
        first = row[0].strip()
        if first == _HEADER_FIRST_CELL:
            if current_header is not None:
                sections.append((current_header, current_rows))
            current_header = [c.strip() for c in row]
            current_rows = []
        else:
            if current_header is None:
                # Pre-header garbage (e.g., trailer records). Skip silently.
                continue
            current_rows.append(row)
    if current_header is not None:
        sections.append((current_header, current_rows))
    return sections


def _classify_section(columns: Sequence[str]) -> Literal["nav", "cash_tx", "unknown"]:
    cols = set(columns)
    if _NAV_FINGERPRINT.issubset(cols):
        return "nav"
    if _CASH_TX_FINGERPRINT.issubset(cols):
        return "cash_tx"
    return "unknown"


# ---------------------------------------------------------------------------
# Schema normalization


_DEPOSIT_WITHDRAWAL_TYPES = frozenset({"Deposits/Withdrawals", "Deposits", "Withdrawals"})


def _empty_nav() -> pd.DataFrame:
    df = pd.DataFrame(columns=["account_id", "currency", "aum"])
    df.index = pd.DatetimeIndex([], name="date")
    return df


def _empty_cash_tx() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "account_id",
            "currency",
            "fx_rate_to_base",
            "amount_native",
            "flow_type_raw",
            "description",
            "transaction_id",
        ]
    )
    df.index = pd.DatetimeIndex([], name="flow_date")
    return df


def _require_columns(df: pd.DataFrame, required: Sequence[str], section: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = (
            f"{section} section missing required column(s) {missing!r}. "
            f"Found columns: {list(df.columns)!r}"
        )
        raise MalformedDataError(msg)


def _normalize_nav(raw: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        raw,
        ["ClientAccountID", "CurrencyPrimary", "ReportDate", "Total"],
        "NAV in Base",
    )
    date = pd.to_datetime(raw["ReportDate"], format="%Y%m%d", errors="coerce")
    bad = date.isna()
    if bad.any():
        bad_vals = raw.loc[bad, "ReportDate"].head(3).tolist()
        msg = f"Cannot parse NAV ReportDate values (expected '%Y%m%d'). First bad: {bad_vals!r}"
        raise MalformedDataError(msg)
    df = pd.DataFrame(
        {
            "account_id": raw["ClientAccountID"].astype(str).to_numpy(),
            "currency": raw["CurrencyPrimary"].astype(str).to_numpy(),
            "aum": pd.to_numeric(raw["Total"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(date.to_numpy(), name="date"),
    )
    return df.sort_index()


def _normalize_cash_tx(raw: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        raw,
        [
            "ClientAccountID",
            "CurrencyPrimary",
            "FXRateToBase",
            "Date/Time",
            "Amount",
            "Type",
        ],
        "Cash Transactions",
    )
    # Filter to deposits/withdrawals only — investment-income rows
    # (dividends, broker interest, withholding tax, …) are discarded.
    keep = raw["Type"].isin(_DEPOSIT_WITHDRAWAL_TYPES)
    raw = raw.loc[keep].copy()
    if raw.empty:
        return _empty_cash_tx()

    flow_date = _parse_flex_datetime(raw["Date/Time"])
    desc = (
        raw["Description"].fillna("").astype(str)
        if "Description" in raw.columns
        else pd.Series([""] * len(raw))
    )
    txid = (
        raw["TransactionID"].fillna("").astype(str)
        if "TransactionID" in raw.columns
        else pd.Series([""] * len(raw))
    )
    df = pd.DataFrame(
        {
            "account_id": raw["ClientAccountID"].astype(str).to_numpy(),
            "currency": raw["CurrencyPrimary"].astype(str).to_numpy(),
            "fx_rate_to_base": pd
            .to_numeric(raw["FXRateToBase"], errors="coerce")
            .fillna(1.0)
            .to_numpy(),
            "amount_native": pd.to_numeric(raw["Amount"], errors="coerce").to_numpy(),
            "flow_type_raw": raw["Type"].astype(str).to_numpy(),
            "description": desc.to_numpy(),
            "transaction_id": txid.to_numpy(),
        },
        index=pd.DatetimeIndex(flow_date.to_numpy(), name="flow_date"),
    )
    return df.sort_index()


def _parse_flex_datetime(series: pd.Series) -> pd.Series:
    """Parse a Flex ``Date/Time`` column tolerantly.

    The Flex export's ``Date/Time`` column is ``%Y%m%d;%H%M%S`` when a
    ``Date/Time Separator`` other than ``None`` is configured, and
    ``%Y%m%d`` otherwise. We try the longer format first and fall
    back to date-only on each cell that fails.
    """
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%Y%m%d;%H%M%S", errors="coerce")
    fallback_mask = parsed.isna() & s.ne("")
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            s.loc[fallback_mask], format="%Y%m%d", errors="coerce"
        )
    if parsed.isna().any():
        bad = s[parsed.isna() & s.ne("")].head(3).tolist()
        if bad:
            msg = (
                f"Cannot parse Flex Date/Time values "
                f"(tried '%Y%m%d;%H%M%S' then '%Y%m%d'). First bad: {bad!r}"
            )
            raise MalformedDataError(msg)
    return parsed
