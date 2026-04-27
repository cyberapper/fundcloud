"""Account-level data providers — unified source of NAV / positions / trades / flows.

Parallel to :mod:`fundcloud.data` (public per-symbol market data), this
subpackage hosts providers of private, per-account data: fund NAV
history, current positions, executed trades, and capital flows. Every
provider satisfies the :class:`AccountProvider` protocol, so the
analysis surface is the same regardless of the backing source.

v0.1 ships one provider — :class:`fundcloud.accounts.FundCloud`, over
the FundCloud platform API. Future providers (IBKR, Plaid) plug into
the same protocol without any changes to downstream code.

The typical flow is::

    import fundcloud as fc

    src = fc.accounts.FundCloud()           # reads FUNDCLOUD_API_KEY
    src.list_funds()                        # DataFrame of visible funds
    pf = src.to_portfolio(fund_id="…")      # analytics-ready Portfolio
    fc.reports.Tearsheet(pf).render_html("my_fund.html")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fundcloud.accounts._base import (
    AccountProvider,
    BaseAccountProvider,
)

__all__ = [
    "AccountProvider",
    "BaseAccountProvider",
    "FundCloud",
]


_LAZY: dict[str, tuple[str, str]] = {
    "FundCloud": ("fundcloud.accounts.fundcloud", "FundCloud"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        from importlib import import_module

        module = import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'fundcloud.accounts' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise lazy-loaded providers to IDEs / REPL tab-completion."""
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # pragma: no cover — static-analysis only
    from fundcloud.accounts.fundcloud import FundCloud
