"""Interactive Brokers (IB) account provider.

Reads NAV / capital-flow data from IB's Flex Query CSV export
(Reports → Flex Queries → Activity Flex Query → CSV) and exposes them
through the same :class:`fundcloud.accounts._base.AccountProvider`
contract :class:`fundcloud.accounts.fundcloud.FundCloud` satisfies. Once
constructed, the analytics surface is identical: ``src.to_portfolio()``
produces a regular :class:`fundcloud.portfolio.Portfolio`, and every
``.fc`` accessor / metric / Tearsheet renderer works without changes.

See ``docs/guides/accounts/ib.md`` for the IB-website setup
walkthrough (which Flex Query fields to tick).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pandas as pd

from fundcloud.accounts._base import BaseAccountProvider, Basis
from fundcloud.accounts._flex import FlexExport, parse_flex_csv
from fundcloud.errors import AmbiguousError, MalformedDataError, NotFoundError
from fundcloud.metrics.core import ReturnMethod
from fundcloud.portfolio import Portfolio

__all__ = ["IB"]


class IB(BaseAccountProvider):
    """NAV / capital-flow source backed by an IB Flex Query CSV.

    Parameters
    ----------
    path
        Filesystem path to a single Flex Query CSV. Mutually exclusive
        with ``files`` and ``text``.
    files
        A sequence of paths to concatenate (e.g., one file per year, or
        per sub-account). Each file is parsed independently and the
        per-section frames are concatenated. Mutually exclusive with
        ``path`` and ``text``.
    text
        Inline CSV content (useful in tests, notebooks, or when piping
        from another tool). Mutually exclusive with ``path`` and
        ``files``. Equivalent to :meth:`from_string`.
    account_id
        Default ``ClientAccountID`` to use when callers don't pass one.
        Convenient for the single-account case.

    Notes
    -----
    IB has no concept of "fund" — a Flex Query CSV is keyed by
    ``ClientAccountID`` directly. We surface the same id as both
    ``fund_id`` and ``account_id`` on the protocol surface, so single-
    account users don't need to think about the distinction
    (``IB("export.csv").to_portfolio()`` works zero-arg).

    Brokerage accounts also have no ``shares_outstanding``, so the
    "NAV-per-share + total-return" path FundCloud defaults to is
    inappropriate for IB. The :meth:`to_portfolio` default flips to
    ``basis="aum"`` + ``method="modified_dietz"`` — the GIPS-standard
    AUM TWR that's the natural analogue of investor return for a
    brokerage account.

    The :meth:`positions` and :meth:`trades` methods return empty
    frames: the Cash Transactions section is event data, not trades,
    and the NAV section gives only asset-class subtotals. To populate
    them, configure separate Open Positions / Trades Flex sections in
    Interactive Brokers.
    """

    name: ClassVar[str] = "ib"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        files: Sequence[str | Path] | None = None,
        text: str | None = None,
        account_id: str | None = None,
    ) -> None:
        provided = sum(1 for v in (path, files, text) if v is not None)
        if provided != 1:
            msg = (
                "Pass exactly one of `path=`, `files=`, or `text=` "
                "(got "
                f"path={path!r}, files={'…' if files else None}, "
                f"text={'…' if text else None})."
            )
            raise ValueError(msg)

        if files is not None:
            exports: list[FlexExport] = [parse_flex_csv(Path(f), require_nav=True) for f in files]
            self._nav_df: pd.DataFrame = pd.concat([e.nav for e in exports]).sort_index()
            self._tx_df: pd.DataFrame = pd.concat([
                e.cash_transactions for e in exports
            ]).sort_index()
        else:
            export = parse_flex_csv(path if path is not None else text)  # type: ignore[arg-type]
            self._nav_df = export.nav
            self._tx_df = export.cash_transactions

        self._default_account_id = account_id

    @classmethod
    def from_string(cls, text: str, *, account_id: str | None = None) -> IB:
        """Construct an :class:`IB` provider from inline CSV text."""
        return cls(text=text, account_id=account_id)

    # --------------------------------------------------------- discovery

    def list_funds(self) -> pd.DataFrame:
        """One row per unique ``ClientAccountID`` in the parsed export.

        For IB, fund and account are the same id (single-tier
        hierarchy). The columns match the ``AccountProvider`` protocol
        so generic UI / report code that walks ``list_funds()`` works
        identically across providers.
        """
        if self._nav_df.empty:
            return pd.DataFrame(
                columns=[
                    "fund_id",
                    "name",
                    "short_name",
                    "currency",
                    "inception_date",
                    "status",
                    "aum",
                    "total_shares",
                ],
            )

        rows: list[dict[str, object]] = []
        for acct_id, sub in self._nav_df.groupby("account_id"):
            sub_sorted = sub.sort_index()
            currencies = sub_sorted["currency"].unique().tolist()
            if len(currencies) > 1:
                msg = (
                    f"IB account {acct_id!r} reports NAV in multiple "
                    f"currencies {currencies!r}; one base currency per "
                    f"account is required. Re-run the Flex Query in a "
                    f"single base currency."
                )
                raise MalformedDataError(msg)
            rows.append({
                "fund_id": acct_id,
                "name": acct_id,
                "short_name": acct_id,
                "currency": currencies[0],
                "inception_date": sub_sorted.index.min(),
                "status": "ACTIVE",
                "aum": float(sub_sorted["aum"].iloc[-1]),
                "total_shares": 1.0,
            })
        return pd.DataFrame(rows)

    def list_accounts(self, fund_id: str | None = None) -> pd.DataFrame:
        """One row per linked account.

        For IB, there is one account per ``ClientAccountID`` and the
        fund id is the same id; the result matches :meth:`list_funds`
        with the column names normalized to the ``AccountProvider``
        account schema.
        """
        funds = self.list_funds()
        if fund_id is not None:
            funds = funds[funds["fund_id"] == fund_id]
            if funds.empty:
                msg = f"fund_id={fund_id!r} not present in this Flex CSV"
                raise NotFoundError(msg)
        if funds.empty:
            return pd.DataFrame(
                columns=[
                    "account_id",
                    "account_name",
                    "fund_id",
                    "fund_name",
                    "currency",
                    "external_account_id",
                    "latest_nav",
                    "latest_aum",
                ],
            )
        return pd.DataFrame({
            "account_id": funds["fund_id"],
            "account_name": funds["name"],
            "fund_id": funds["fund_id"],
            "fund_name": funds["name"],
            "currency": funds["currency"],
            "external_account_id": funds["fund_id"],
            "latest_nav": funds["aum"],
            "latest_aum": funds["aum"],
        }).reset_index(drop=True)

    # --------------------------------------------------------- timeseries

    def nav(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        adjust_for_flows: bool = True,
    ) -> pd.DataFrame:
        """Daily NAV / AUM rows from the Flex Query NAV section.

        Unlike network providers (e.g.,
        :class:`fundcloud.accounts.fundcloud.FundCloud`) this method
        does **not** apply a 1-year-back default to ``start``. The
        Flex Query CSV is already bounded by whatever period was
        requested at export time — returning the full file is
        almost always the right behaviour. Pass ``start=`` / ``end=``
        to narrow the window further.

        ``adjust_for_flows=True`` (default) returns a synthetic
        flow-adjusted AUM curve computed client-side via
        :func:`fundcloud.metrics.returns_from_nav`, replaying the
        return series as if no deposits or withdrawals had occurred.
        Pass ``False`` for the raw ``Total`` column straight from the
        CSV.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex named ``date``. Columns: ``nav`` (= AUM,
            since IB has no per-share concept), ``aum``, ``shares``
            (synthesised constant ``1.0``), ``daily_return``,
            ``fill_type``. ``daily_return`` reflects the chosen
            ``adjust_for_flows`` mode.
        """
        from fundcloud.metrics import returns_from_nav

        acct = self._resolve_account(fund_id, account_id)

        sub = self._nav_df[self._nav_df["account_id"] == acct].sort_index()
        if start is not None or end is not None:
            sub = sub.loc[
                pd.Timestamp(start) if start is not None else None : pd.Timestamp(end)
                if end is not None
                else None
            ]

        if sub.empty:
            return pd.DataFrame(
                columns=["nav", "aum", "shares", "daily_return", "fill_type"],
                index=pd.DatetimeIndex([], name="date"),
            )

        aum = sub["aum"].astype(float)

        if adjust_for_flows and len(aum) > 1:
            # Drop leading rows where AUM is non-positive (account is
            # pre-funding). Those are technically valid Total values
            # straight from IB but they break compounding (anchor of
            # 0 → flat-zero curve, division by zero in pct_change).
            funded_mask = aum > 0
            if funded_mask.any():
                first_funded = aum.index[funded_mask.argmax()]
                aum_funded = aum.loc[first_funded:]
                signed_flows = self._signed_base_flows(acct, aum_funded.index)
                try:
                    returns = returns_from_nav(
                        aum_funded, capital_flows=signed_flows, method="daily_twr"
                    )
                except ValueError:
                    returns = pd.Series(dtype=float)
                if not returns.empty:
                    # Replay AUM curve as if no flows had occurred — anchor at
                    # the first funded AUM, compound flow-adjusted returns.
                    anchor = float(aum_funded.iloc[0])
                    synthetic = pd.Series(index=aum_funded.index, dtype=float)
                    synthetic.iloc[0] = anchor
                    synthetic.iloc[1:] = anchor * (1.0 + returns).cumprod()
                    # Stitch leading non-funded rows back so the result still
                    # spans the user-visible NAV period.
                    leading = aum.loc[:first_funded].iloc[:-1]
                    aum = pd.concat([leading, synthetic])

        return pd.DataFrame(
            {
                "nav": aum.to_numpy(),
                "aum": aum.to_numpy(),
                "shares": 1.0,
                "daily_return": pd.Series(aum, dtype=float).pct_change().to_numpy(),
                "fill_type": "actual",
            },
            index=aum.index,
        )

    def capital_flows(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Capital flow events (deposits / withdrawals only).

        IB's Flex export ships signed amounts (positive = deposit,
        negative = withdrawal); we translate to the protocol's
        ``flow_type`` + always-positive ``amount`` convention and
        multiply by ``FXRateToBase`` so amounts are denominated in the
        account's base currency, ready to reconcile against
        base-currency NAV.

        Returns every flow row in the parsed CSV by default — no
        date-based default filter, since the export window is already
        fixed by what the user asked IB for. Pass ``start=`` /
        ``end=`` to narrow further.
        """
        acct = self._resolve_account(fund_id, account_id)

        sub = self._tx_df[self._tx_df["account_id"] == acct].sort_index()
        if start is not None or end is not None:
            sub = sub.loc[
                pd.Timestamp(start) if start is not None else None : pd.Timestamp(end)
                if end is not None
                else None
            ]

        if sub.empty:
            return _empty_capital_flows()

        signed_native = sub["amount_native"].astype(float)
        fx = sub["fx_rate_to_base"].astype(float)
        signed_base = signed_native * fx
        flow_type = signed_base.where(signed_base < 0, other="INJECTION").where(
            signed_base >= 0, other="WITHDRAWAL"
        )
        # Resolve base currency from the NAV section if we have it; otherwise
        # fall back to the native currency.
        base_ccy = self._account_base_currency(acct) or sub["currency"].iloc[0]

        return pd.DataFrame(
            {
                "flow_type": flow_type.to_numpy(),
                "amount": signed_base.abs().to_numpy(),
                "currency": base_ccy,
                "account_id": sub["account_id"].to_numpy(),
                "notes": sub["description"].to_numpy(),
            },
            index=sub.index,
        )

    # --------------------------------------------------------- snapshots

    def positions(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        asof: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Returns an empty frame.

        The Flex Query "Net Asset Value (NAV) in Base" section provides
        asset-class subtotals (`Stock`, `Bonds`, `Crypto`, …) but not
        per-symbol position rows. Configure a separate "Open Positions"
        Flex section in Interactive Brokers to populate this.
        """
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "asset_type",
                "quantity",
                "avg_cost",
                "current_price",
                "market_value",
                "currency",
                "weight",
                "unrealized_pnl",
                "unrealized_pnl_percent",
                "account_id",
                "info",
            ],
        )

    def trades(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Returns an empty frame.

        The Flex Query "Cash Transactions" section is event data, not
        trades. Configure a separate "Trades" Flex section in
        Interactive Brokers to populate this.
        """
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "quantity",
                "price",
                "amount",
                "currency",
                "fee",
                "broker",
                "status",
                "account_id",
            ],
            index=pd.DatetimeIndex([], name="trade_date"),
        )

    # --------------------------------------------------------- to_portfolio

    def to_portfolio(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        basis: Basis = "aum",
        method: ReturnMethod = "modified_dietz",
        benchmark: pd.Series | None = None,
        name: str | None = None,
    ) -> Portfolio:
        """Build a :class:`Portfolio` from the IB account.

        Defaults differ from the FundCloud provider: ``basis="aum"`` +
        ``method="modified_dietz"``. Brokerage accounts have no
        ``shares_outstanding``, so the "NAV-per-share + total-return"
        path doesn't apply — AUM-basis Modified Dietz is the
        GIPS-standard analogue of investor return for a brokerage book.
        Override either kwarg to switch (e.g., ``method="daily_twr"``
        for daily-precision flow timing).
        """
        return super().to_portfolio(
            fund_id,
            account_id=account_id,
            start=start,
            end=end,
            basis=basis,
            method=method,
            benchmark=benchmark,
            name=name,
        )

    # --------------------------------------------------------- internals

    def _resolve_account(self, fund_id: str | None, account_id: str | None) -> str:
        """Resolve which ``ClientAccountID`` to operate on.

        Priority: explicit ``account_id`` → explicit ``fund_id`` →
        constructor default → unique account in the parsed CSV.
        Raises :class:`AmbiguousError` if none are set and the CSV
        contains multiple accounts.
        """
        if account_id is not None:
            return account_id
        if fund_id is not None:
            return fund_id
        if self._default_account_id is not None:
            return self._default_account_id

        accounts: list[str] = (
            self._nav_df["account_id"].unique().tolist() if not self._nav_df.empty else []
        )
        if len(accounts) == 0:
            msg = "No accounts found in the Flex CSV."
            raise NotFoundError(msg)
        if len(accounts) == 1:
            return str(accounts[0])
        msg = (
            f"Multiple accounts in this Flex CSV: {sorted(accounts)!r}. "
            f"Pass account_id= (or fund_id=) to select one, or set "
            f"`account_id=` on the constructor for a default."
        )
        raise AmbiguousError(msg)

    def _account_base_currency(self, account_id: str) -> str | None:
        sub = self._nav_df[self._nav_df["account_id"] == account_id]
        if sub.empty:
            return None
        return str(sub["currency"].iloc[0])

    def _signed_base_flows(self, account_id: str, nav_index: pd.DatetimeIndex) -> pd.Series:
        """Return signed base-currency flow per NAV date for one account.

        Used by the synthetic ``adjust_for_flows=True`` path of
        :meth:`nav`. Flows after the last NAV date or before the first
        are dropped — they cannot be applied within the visible NAV
        window.
        """
        sub = self._tx_df[self._tx_df["account_id"] == account_id]
        if sub.empty:
            return pd.Series(0.0, index=nav_index)
        signed = sub["amount_native"].astype(float) * sub["fx_rate_to_base"].astype(float)
        # Group same-day flows; reindex to nav dates (forward-attribute via searchsorted)
        per_date = signed.groupby(signed.index.normalize()).sum()
        nav_sorted = pd.DatetimeIndex(sorted(nav_index.unique()))
        positions = nav_sorted.searchsorted(per_date.index, side="left")
        mask = positions < len(nav_sorted)
        if not mask.any():
            return pd.Series(0.0, index=nav_index)
        target = nav_sorted[positions[mask]]
        aligned = pd.Series(per_date.values[mask], index=target)
        aggregated = aligned.groupby(aligned.index).sum()
        return aggregated.reindex(nav_index).fillna(0.0).astype(float)

    def _display_name(self, fund_id: str | None, account_id: str | None) -> str:
        acct = account_id or fund_id or self._default_account_id
        if acct:
            return f"IB / {acct}"
        return "IB"


# -------------------------------------------------------------------- helpers


def _empty_capital_flows() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=["flow_type", "amount", "currency", "account_id", "notes"],
    )
    df.index = pd.DatetimeIndex([], name="flow_date")
    return df
