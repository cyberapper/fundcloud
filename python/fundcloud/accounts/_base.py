"""``AccountProvider`` protocol + ``BaseAccountProvider`` ABC.

Every provider in :mod:`fundcloud.accounts` satisfies the same six-method
contract (``list_funds`` / ``list_accounts`` / ``nav`` / ``positions`` /
``trades`` / ``capital_flows``) plus a default ``to_portfolio`` composition
defined on :class:`BaseAccountProvider`. The composition wires NAV + flows
through :meth:`fundcloud.portfolio.Portfolio.from_nav`, applying the
right sign convention per ``basis`` — so IB / Plaid will reuse the
same correct-return logic without re-implementing anything.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd

from fundcloud.metrics.core import ReturnMethod
from fundcloud.portfolio import Portfolio

__all__ = [
    "AccountProvider",
    "BaseAccountProvider",
    "Basis",
]

Basis = Literal["nav_per_share", "aum"]


@runtime_checkable
class AccountProvider(Protocol):
    """Unified protocol for account-level data providers.

    Concrete implementations: :class:`fundcloud.accounts.FundCloud`
    and :class:`fundcloud.accounts.IB`. Future providers (e.g., Plaid)
    follow the same contract.

    Every method returns a :class:`pd.DataFrame` (or :class:`pd.Series`
    where natural) — typed entities stay in docstrings as documented
    schemas, not in code, to preserve the ecosystem feel of the rest
    of the library.
    """

    name: ClassVar[str]

    # --- discovery --------------------------------------------------------
    def list_funds(self) -> pd.DataFrame:
        """One row per fund visible to this credential.

        Columns (minimum): ``fund_id``, ``name``, ``short_name``
        (where available), ``currency``, ``inception_date``, ``status``,
        ``aum``, ``total_shares``. Extra provider-specific fields may
        appear in a trailing ``info`` column.
        """
        ...

    def list_accounts(self, fund_id: str | None = None) -> pd.DataFrame:
        """One row per linked account, optionally filtered to one fund.

        Columns (minimum): ``account_id``, ``account_name``, ``fund_id``,
        ``fund_name``, ``currency``, ``external_account_id`` (broker-side
        id, when available), ``latest_nav``, ``latest_aum``.
        """
        ...

    # --- timeseries -------------------------------------------------------
    def nav(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        adjust_for_flows: bool = True,
    ) -> pd.DataFrame:
        """Historical NAV timeseries.

        DatetimeIndex. Columns: ``nav`` (per-share), ``aum`` (total),
        ``shares``, ``daily_return`` (if reported by the provider).
        When ``account_id=None`` the provider returns the fund-level
        aggregate; when set, a single-account curve.

        ``adjust_for_flows=True`` (default) requests a flow-smoothed
        NAV — implementation-defined per provider (server-side query
        flag for FundCloud; client-side fallback for IB / Plaid
        when those land). Pass ``False`` for the raw, unadjusted
        NAV series.
        """
        ...

    def capital_flows(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Capital flow events (injections, withdrawals, distributions).

        DatetimeIndex (``flow_date``). Columns: ``flow_type``
        (``INJECTION`` / ``WITHDRAWAL`` / ``DISTRIBUTION``), ``amount``
        (always positive — direction is encoded in ``flow_type``, not in
        sign), ``currency``, ``account_id``, ``notes``.
        """
        ...

    # --- snapshots --------------------------------------------------------
    def positions(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        asof: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Current open positions.

        Columns: ``symbol``, ``name``, ``asset_type``, ``quantity``,
        ``avg_cost``, ``current_price``, ``market_value``, ``currency``,
        ``weight``, ``unrealized_pnl``, ``unrealized_pnl_percent``,
        ``account_id``.
        """
        ...

    def trades(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Executed trades, one row per fill.

        DatetimeIndex (``trade_date``). Columns: ``symbol``, ``side``,
        ``quantity``, ``price``, ``amount``, ``currency``, ``fee``,
        ``broker``, ``status``, ``account_id``.
        """
        ...

    # --- convenience ------------------------------------------------------
    def to_portfolio(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        basis: Basis = "nav_per_share",
        method: ReturnMethod = "total_return",
        benchmark: pd.Series | None = None,
        name: str | None = None,
    ) -> Portfolio:
        """Fetch NAV (+ flows) and return a ready-to-analyse Portfolio.

        ``basis`` + ``method`` pick the return-computation convention:

        - ``basis='nav_per_share'`` + ``method='total_return'`` (default):
          per-share NAV with ``DISTRIBUTION`` flows added back;
          injections and withdrawals are ignored. Matches how funds
          report investor return.
        - ``basis='aum'`` + ``method='modified_dietz'`` (or
          ``'daily_twr'``): AUM TWR, all flow types signed and
          aggregated.

        Delegates return computation to
        :func:`fundcloud.metrics.returns_from_nav`.
        """
        ...


class BaseAccountProvider(ABC):
    """Default implementations shared by every concrete provider.

    Subclasses must implement all six fact methods
    (:meth:`list_funds`, :meth:`list_accounts`, :meth:`nav`,
    :meth:`capital_flows`, :meth:`positions`, :meth:`trades`).
    :meth:`to_portfolio` is provided here and composes them with
    :meth:`Portfolio.from_nav` using the correct sign convention for
    each basis.
    """

    name: ClassVar[str]

    @abstractmethod
    def list_funds(self) -> pd.DataFrame: ...

    @abstractmethod
    def list_accounts(self, fund_id: str | None = None) -> pd.DataFrame: ...

    @abstractmethod
    def nav(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        adjust_for_flows: bool = True,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def capital_flows(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def positions(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        asof: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame: ...

    @abstractmethod
    def trades(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame: ...

    # --------------------------------------------------------- to_portfolio
    def to_portfolio(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        basis: Basis = "nav_per_share",
        method: ReturnMethod = "total_return",
        benchmark: pd.Series | None = None,
        name: str | None = None,
    ) -> Portfolio:
        """Fetch NAV (+ flows) and build a Portfolio.

        See :class:`AccountProvider.to_portfolio` for the parameter
        semantics. The implementation:

        1. Fetches NAV via :meth:`nav`.
        2. Fetches capital flows via :meth:`capital_flows` (skipped
           when ``method='none'``).
        3. Converts flows to the shape expected by
           :func:`fundcloud.metrics.returns_from_nav` for the chosen
           ``basis`` (per-share distributions for ``nav_per_share``;
           signed AUM flows for ``aum``).
        4. Delegates to :meth:`Portfolio.from_nav`.
        """
        # Always fetch raw NAV here. ``to_portfolio`` applies its own
        # canonical client-side TWR (or ``total_return`` add-back) below
        # via ``returns_from_nav`` / ``Portfolio.from_nav``; passing
        # server-adjusted NAV would double-count the flow correction.
        nav_df = self.nav(
            fund_id,
            account_id=account_id,
            start=start,
            end=end,
            adjust_for_flows=False,
        )
        if nav_df.empty:
            msg = (
                f"No NAV data returned for fund_id={fund_id!r} "
                f"account_id={account_id!r} — cannot build Portfolio."
            )
            raise ValueError(msg)

        display = name or self._display_name(fund_id, account_id)

        if method == "none":
            # No flow adjustment requested.
            nav_series = nav_df["nav"] if basis == "nav_per_share" else nav_df["aum"]
            return Portfolio.from_nav(
                nav_series,
                method="none",
                benchmark=benchmark,
                name=display,
            )

        flows = self.capital_flows(fund_id, account_id=account_id, start=start, end=end)

        if basis == "nav_per_share":
            if method not in ("total_return", "none"):
                msg = (
                    f"basis='nav_per_share' is only valid with method='total_return' "
                    f"or 'none'; got method={method!r}. Use basis='aum' for "
                    f"modified_dietz / daily_twr."
                )
                raise ValueError(msg)
            distributions = _flows_to_per_share_distributions(flows, nav_df["shares"])
            return Portfolio.from_nav(
                nav_df["nav"],
                distributions=distributions,
                method="total_return",
                benchmark=benchmark,
                name=display,
            )

        if basis == "aum":
            if method not in ("modified_dietz", "daily_twr", "none"):
                msg = (
                    f"basis='aum' is only valid with method='modified_dietz', "
                    f"'daily_twr', or 'none'; got method={method!r}. Use "
                    f"basis='nav_per_share' for total_return."
                )
                raise ValueError(msg)
            signed = _flows_to_signed_aum_series(flows, nav_df.index)
            return Portfolio.from_nav(
                nav_df["aum"],
                capital_flows=signed,
                method=method,
                benchmark=benchmark,
                name=display,
            )

        msg = f"unknown basis {basis!r}; expected 'nav_per_share' or 'aum'"
        raise ValueError(msg)

    # --------------------------------------------------------- helpers
    def _display_name(self, fund_id: str | None, account_id: str | None) -> str:
        """Default Portfolio name for ``to_portfolio``; subclasses can override."""
        parts = [self.name]
        if fund_id:
            parts.append(f"fund={fund_id}")
        if account_id:
            parts.append(f"account={account_id}")
        return " ".join(parts)


# -------------------------------------------------------------------- helpers


def _flows_to_per_share_distributions(
    flows: pd.DataFrame,
    shares: pd.Series,
) -> pd.Series:
    """Convert DISTRIBUTION flows to a per-share add-back series.

    Only ``DISTRIBUTION`` flows affect NAV-per-share (injections and
    withdrawals issue / redeem shares at current NAV, leaving
    NAV-per-share invariant). Each distribution is divided by the
    outstanding shares on the attributed NAV date to give a per-share
    add-back aligned to ``shares.index``.

    Flows whose date doesn't fall on a NAV date are attributed to the
    next NAV date on or after the flow date. Flows past the last NAV
    date are dropped.
    """
    idx = pd.DatetimeIndex(shares.index)
    if flows.empty:
        return pd.Series(0.0, index=idx, name="distributions")

    distribs = flows[flows["flow_type"] == "DISTRIBUTION"]
    if distribs.empty:
        return pd.Series(0.0, index=idx, name="distributions")

    # Total distribution amount per flow_date (sum same-day distributions).
    per_date = distribs.groupby(distribs.index)["amount"].sum()
    aligned = _attribute_to_next_index_date(per_date, idx)
    # Divide by shares on the attributed NAV date. Shares of 0 → NaN → 0.
    shares_safe = shares.astype(float).replace(0.0, float("nan"))
    per_share = (aligned / shares_safe.reindex(aligned.index)).fillna(0.0)
    per_share.name = "distributions"
    return per_share.reindex(idx).fillna(0.0)


def _flows_to_signed_aum_series(
    flows: pd.DataFrame,
    index: pd.DatetimeIndex | pd.Index,
) -> pd.Series:
    """Sign flows by type and aggregate to a daily series.

    ``INJECTION`` → positive (money into the fund).
    ``WITHDRAWAL`` → negative (money out to investors).
    ``DISTRIBUTION`` → negative (cash distributed out of the fund pool).

    Unknown flow types are treated as zero (defensive).
    """
    idx = pd.DatetimeIndex(index)
    if flows.empty:
        return pd.Series(0.0, index=idx, name="capital_flows")

    sign_map = {"INJECTION": 1.0, "WITHDRAWAL": -1.0, "DISTRIBUTION": -1.0}
    signs = flows["flow_type"].map(sign_map).fillna(0.0)
    signed = flows["amount"].astype(float) * signs

    # Sum same-day flows.
    per_date = signed.groupby(signed.index).sum()
    aligned = _attribute_to_next_index_date(per_date, idx)
    aligned.name = "capital_flows"
    return aligned.reindex(idx).fillna(0.0)


def _attribute_to_next_index_date(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Attribute each entry in ``series`` to the next date in ``target_index``.

    If an entry's date is already in ``target_index``, it stays there.
    Otherwise it's attributed to the smallest target date strictly
    greater than the entry's date. Entries after the last target date
    are dropped. Multiple entries attributed to the same target date
    are summed.
    """
    if len(series) == 0 or len(target_index) == 0:
        return pd.Series(dtype=float)
    target_sorted = pd.DatetimeIndex(sorted(target_index.unique()))
    # For each entry date, find its insertion point in target_sorted.
    # side='left' means an exact match slots into its own position.
    positions = target_sorted.searchsorted(series.index, side="left")
    mask = positions < len(target_sorted)
    if not mask.any():
        return pd.Series(dtype=float)
    target_dates = target_sorted[positions[mask]]
    aligned = pd.Series(series.values[mask], index=target_dates)
    return aligned.groupby(aligned.index).sum()
