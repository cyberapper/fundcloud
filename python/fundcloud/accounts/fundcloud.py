"""FundCloud account provider — NAV, positions, trades, capital flows.

Implements :class:`fundcloud.accounts._base.AccountProvider` against the
FundCloud platform REST API. Shares the HTTP + auth + retry stack with
:class:`fundcloud.data.fundcloud.FundCloud` via a common
:class:`fundcloud._clients.fundcloud.FundCloudClient`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from fundcloud._clients.fundcloud import FUNDCLOUD_BASE_URL, FundCloudClient
from fundcloud.accounts._base import BaseAccountProvider
from fundcloud.data._defaults import default_start_one_year_back
from fundcloud.errors import AmbiguousError, NotFoundError

__all__ = ["FundCloud"]


class FundCloud(BaseAccountProvider):
    """NAV / positions / trades / capital-flows source for FundCloud.

    Parameters
    ----------
    fund_id
        Default fund id to use when callers don't pass one. Convenient
        for the single-fund case. If omitted, each call that needs a
        fund id auto-resolves: an explicit ``account_id`` is mapped to
        its parent fund via :meth:`list_accounts`; otherwise the single
        visible fund is used. :class:`fundcloud.errors.AmbiguousError`
        surfaces only when neither hint is available and the credential
        sees more than one fund.
    api_key
        Falls back to the ``FUNDCLOUD_API_KEY`` env var.
    base_url
        Override the API base URL (useful in tests).
    timeout
        Per-request timeout in seconds.

    Notes
    -----
    Every method accepts ``fund_id`` and ``account_id`` as keywords. With
    ``account_id=None`` the provider returns the fund-level aggregate;
    with ``account_id`` set it drills into a single linked account. You
    can pass ``account_id`` *without* ``fund_id`` — the provider resolves
    the parent fund automatically (lazy, cached for the provider's
    lifetime).

    :meth:`nav` requests server-side flow-adjusted NAV by default
    (``adjust_for_flows=True``); :meth:`to_portfolio` always opts out
    and applies the canonical client-side TWR in
    :func:`fundcloud.metrics.returns_from_nav`, which keeps results
    comparable across providers (IB, Plaid) that don't offer the same
    server-side flag.
    """

    name: ClassVar[str] = "fundcloud"

    def __init__(
        self,
        fund_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str = FUNDCLOUD_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._default_fund_id = fund_id
        self._client = FundCloudClient(api_key=api_key, base_url=base_url, timeout=timeout)
        # Cache for fund metadata (filled on first list_funds call), used
        # to enrich list_accounts output with fund names.
        self._funds_cache: pd.DataFrame | None = None
        # Lazy-built map of account_id → fund_id, populated when a caller
        # passes only ``account_id=`` and we need to resolve its parent
        # fund. Cached for the provider's lifetime — drop the provider
        # to refresh.
        self._account_to_fund: dict[str, str] | None = None

    # --------------------------------------------------------- discovery

    def list_funds(self) -> pd.DataFrame:
        """All funds visible to this credential, as a DataFrame."""
        rows = list(self._client.get_paginated("/funds"))
        if not rows:
            empty = pd.DataFrame(
                columns=[
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
                ],
            )
            self._funds_cache = empty
            return empty
        df = pd.DataFrame({
            "fund_id": [r.get("id") for r in rows],
            "name": [r.get("name") for r in rows],
            "short_name": [r.get("short_name") for r in rows],
            "currency": [r.get("currency") for r in rows],
            "inception_date": pd.to_datetime(
                [r.get("inception_date") for r in rows], errors="coerce"
            ),
            "status": [r.get("status") for r in rows],
            "aum": [_as_float(r.get("aum")) for r in rows],
            "total_shares": [_as_float(r.get("total_shares")) for r in rows],
            "fund_type": [r.get("fund_type") for r in rows],
            "info": [r.get("info") for r in rows],
        })
        self._funds_cache = df
        return df

    def list_accounts(self, fund_id: str | None = None) -> pd.DataFrame:
        """Linked accounts under ``fund_id`` (or all funds if None).

        Discovered via ``NAVEntry.account_breakdown`` on one recent
        aggregated NAV entry per fund — the FundCloud public API has
        no dedicated ``/accounts`` endpoint.
        """
        funds_df = self._funds_cache if self._funds_cache is not None else self.list_funds()
        if fund_id is not None:
            funds_df = funds_df[funds_df["fund_id"] == fund_id]
            if funds_df.empty:
                msg = f"fund_id={fund_id!r} not visible to this credential"
                raise NotFoundError(msg)

        rows: list[dict[str, Any]] = []
        for _, fund in funds_df.iterrows():
            fid = fund["fund_id"]
            payload = self._client.get(
                f"/funds/{fid}/nav",
                params={
                    "aggregation": "daily",
                    "page_size": 1,
                    "page": 1,
                    "adjust_for_flows": "false",
                    "sort": "-date",
                },
            )
            entries = payload.get("data", []) if isinstance(payload, dict) else []
            if not entries:
                continue
            entry = entries[0]
            for acc in entry.get("account_breakdown", []) or []:
                rows.append({
                    "account_id": acc.get("account_id"),
                    "account_name": acc.get("account_name"),
                    "fund_id": fid,
                    "fund_name": fund["name"],
                    "currency": fund["currency"],
                    "external_account_id": acc.get("account_id"),
                    "latest_nav": _as_float(acc.get("nav")),
                    "latest_aum": _as_float(acc.get("nav")),  # same field on breakdown
                })
        if not rows:
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
        return pd.DataFrame(rows)

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
        """Historical NAV, fund-aggregated (default) or per-account.

        ``start`` defaults to one year before ``end`` (or today − 1 year
        when ``end`` is also ``None``) — same convention as the network
        market-data backends. Pass ``start=`` explicitly for longer
        history. ``adjust_for_flows=True`` (default) requests the
        server-side flow-smoothed NAV; pass ``False`` for raw values.
        Both modes always use ``aggregation=daily`` (one row per date),
        which is the only mode compatible with the API's
        ``adjust_for_flows=true`` query.
        """
        fid = self._resolve_fund(fund_id, account_id=account_id)
        start = default_start_one_year_back(start, end)
        params: dict[str, Any] = {
            "aggregation": "daily",
            "adjust_for_flows": "true" if adjust_for_flows else "false",
            "sort": "date",
            "start_date": _as_date_str(start),
        }
        if account_id is not None:
            params["account_id"] = account_id
        if end is not None:
            params["end_date"] = _as_date_str(end)

        rows = list(self._client.get_paginated(f"/funds/{fid}/nav", params=params))
        if not rows:
            return pd.DataFrame(
                columns=["nav", "aum", "shares", "daily_return", "fill_type"],
                index=pd.DatetimeIndex([], name="date"),
            )
        df = pd.DataFrame(
            {
                "nav": [_as_float(r.get("nav")) for r in rows],
                "aum": [_as_float(r.get("aum")) for r in rows],
                "shares": [_as_float(r.get("shares")) for r in rows],
                "daily_return": [_as_float(r.get("daily_return")) for r in rows],
                "fill_type": [r.get("fill_type") for r in rows],
            },
            index=pd.DatetimeIndex(pd.to_datetime([r.get("date") for r in rows]), name="date"),
        ).sort_index()
        return df

    def capital_flows(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Capital flow events — amount is positive; direction is in flow_type.

        ``start`` defaults to one year before ``end`` (today − 1 year
        when ``end`` is also ``None``).
        """
        fid = self._resolve_fund(fund_id, account_id=account_id)
        start = default_start_one_year_back(start, end)
        params: dict[str, Any] = {
            "sort": "flow_date",
            "start_date": _as_date_str(start),
        }
        if account_id is not None:
            params["account_id"] = account_id
        if end is not None:
            params["end_date"] = _as_date_str(end)

        rows = list(self._client.get_paginated(f"/funds/{fid}/capital-flows", params=params))
        if not rows:
            return pd.DataFrame(
                columns=["flow_type", "amount", "currency", "account_id", "notes"],
                index=pd.DatetimeIndex([], name="flow_date"),
            )
        df = pd.DataFrame(
            {
                "flow_type": [r.get("flow_type") for r in rows],
                "amount": [_as_float(r.get("amount")) for r in rows],
                "currency": [r.get("currency") for r in rows],
                "account_id": [r.get("account_id") for r in rows],
                "notes": [r.get("notes") for r in rows],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime([r.get("flow_date") for r in rows]), name="flow_date"
            ),
        ).sort_index()
        return df

    # --------------------------------------------------------- snapshots

    def positions(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        asof: pd.Timestamp | str | None = None,  # reserved for future API support
    ) -> pd.DataFrame:
        """Current open positions."""
        fid = self._resolve_fund(fund_id, account_id=account_id)
        params: dict[str, Any] = {}
        if account_id is not None:
            params["account_id"] = account_id

        rows = list(self._client.get_paginated(f"/funds/{fid}/positions", params=params))
        if not rows:
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
        return pd.DataFrame({
            "symbol": [r.get("symbol") for r in rows],
            "name": [r.get("name") for r in rows],
            "asset_type": [r.get("asset_type") for r in rows],
            "quantity": [_as_float(r.get("quantity")) for r in rows],
            "avg_cost": [_as_float(r.get("avg_cost")) for r in rows],
            "current_price": [_as_float(r.get("current_price")) for r in rows],
            "market_value": [_as_float(r.get("market_value")) for r in rows],
            "currency": [r.get("currency") for r in rows],
            "weight": [_as_float(r.get("weight")) for r in rows],
            "unrealized_pnl": [_as_float(r.get("unrealized_pnl")) for r in rows],
            "unrealized_pnl_percent": [_as_float(r.get("unrealized_pnl_percent")) for r in rows],
            "account_id": [r.get("account_name") or r.get("external_account_id") for r in rows],
            "info": [r.get("info") for r in rows],
        })

    def trades(
        self,
        fund_id: str | None = None,
        *,
        account_id: str | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Executed trades, one row per fill.

        ``start`` defaults to one year before ``end`` (today − 1 year
        when ``end`` is also ``None``).
        """
        fid = self._resolve_fund(fund_id, account_id=account_id)
        start = default_start_one_year_back(start, end)
        params: dict[str, Any] = {
            "sort": "trade_date",
            "start_date": _as_date_str(start),
        }
        if account_id is not None:
            params["account_id"] = account_id
        if end is not None:
            params["end_date"] = _as_date_str(end)

        rows = list(self._client.get_paginated(f"/funds/{fid}/trades", params=params))
        if not rows:
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
        df = pd.DataFrame(
            {
                "symbol": [r.get("symbol") for r in rows],
                "side": [r.get("side") for r in rows],
                "quantity": [_as_float(r.get("quantity")) for r in rows],
                "price": [_as_float(r.get("price")) for r in rows],
                "amount": [_as_float(r.get("amount")) for r in rows],
                "currency": [r.get("currency") for r in rows],
                "fee": [_as_float(r.get("fee")) for r in rows],
                "broker": [r.get("broker") for r in rows],
                "status": [r.get("status") for r in rows],
                "account_id": [r.get("account_name") or r.get("external_account_id") for r in rows],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime([r.get("trade_date") for r in rows]),
                name="trade_date",
            ),
        ).sort_index()
        return df

    # --------------------------------------------------------- internals

    def _resolve_fund(
        self,
        fund_id: str | None,
        account_id: str | None = None,
    ) -> str:
        """Resolve ``fund_id`` via explicit arg → constructor default →
        ``account_id`` lookup → single-fund auto-pick.

        When ``account_id`` is supplied without an explicit ``fund_id``,
        we look up the parent fund via :meth:`_account_to_fund_map` (which
        lazily fetches and caches :meth:`list_accounts`). This lets users
        say ``src.nav(account_id=X)`` directly even with multiple funds
        visible — no need to pass ``fund_id=`` for each call.
        """
        if fund_id is not None:
            return fund_id
        if self._default_fund_id is not None:
            return self._default_fund_id
        if account_id is not None:
            mapping = self._account_to_fund_map()
            resolved = mapping.get(account_id)
            if resolved is not None:
                return resolved
            msg = (
                f"account_id={account_id!r} not visible to this credential. "
                f"Pass fund_id= explicitly if the account is known to be valid."
            )
            raise NotFoundError(msg)
        funds = self.list_funds()
        if len(funds) == 0:
            msg = "No funds visible to this credential."
            raise NotFoundError(msg)
        if len(funds) == 1:
            resolved = funds.iloc[0]["fund_id"]
            return str(resolved)
        names = ", ".join(f"{row['name']!r} ({row['fund_id']})" for _, row in funds.iterrows())
        msg = (
            f"Multiple funds visible: {names}. Pass fund_id= to the "
            f"constructor or as a keyword argument to this call."
        )
        raise AmbiguousError(msg)

    def _account_to_fund_map(self) -> dict[str, str]:
        """Lazy-built ``account_id → fund_id`` lookup, cached for life of
        the provider. First call iterates :meth:`list_accounts` across
        every visible fund; subsequent calls hit the cache.
        """
        if self._account_to_fund is None:
            accounts = self.list_accounts()
            self._account_to_fund = {
                str(row["account_id"]): str(row["fund_id"])
                for _, row in accounts.iterrows()
                if row.get("account_id") and row.get("fund_id")
            }
        return self._account_to_fund

    def _display_name(self, fund_id: str | None, account_id: str | None) -> str:
        """Use fund + account name when available; fall back to ids."""
        fid = fund_id if fund_id is not None else self._default_fund_id
        # If the caller passed only account_id, try the lazy map (already
        # populated by an upstream nav() call inside to_portfolio).
        if fid is None and account_id is not None and self._account_to_fund is not None:
            fid = self._account_to_fund.get(account_id)
        if fid is None or self._funds_cache is None:
            return super()._display_name(fund_id, account_id)
        match = self._funds_cache[self._funds_cache["fund_id"] == fid]
        if match.empty:
            return super()._display_name(fund_id, account_id)
        base = str(match.iloc[0]["short_name"] or match.iloc[0]["name"] or fid)
        if account_id:
            base = f"{base} / {account_id}"
        return base


# -------------------------------------------------------------------- helpers


def _as_float(x: object) -> float:
    """Coerce API values (possibly strings or ``None``) to a float, NaN on failure."""
    if x is None:
        return float("nan")
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_date_str(x: pd.Timestamp | str) -> str:
    """Convert a timestamp-like to the ``YYYY-MM-DD`` string the API expects."""
    return str(pd.Timestamp(x).date())
