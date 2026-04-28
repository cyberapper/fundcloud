"""Shared FundCloud HTTP client — auth, pagination, typed error mapping.

Used by both :class:`fundcloud.data.fundcloud.FundCloud` (market data)
and :class:`fundcloud.accounts.fundcloud.FundCloud` (NAV / positions /
trades / capital flows). One ``X-API-Key``, one retry policy, one
pagination loop.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from types import TracebackType
from typing import Any

from fundcloud.data._http import HttpClient, TransientHttpError, require_httpx
from fundcloud.errors import (
    AuthError,
    MalformedDataError,
    NotFoundError,
    QuotaError,
    TransientError,
)

__all__ = ["FUNDCLOUD_BASE_URL", "FUNDCLOUD_ENV_VAR", "FundCloudClient"]

FUNDCLOUD_BASE_URL = "https://api.fundcloud.com/api/v1"
FUNDCLOUD_ENV_VAR = "FUNDCLOUD_API_KEY"

# OpenAPI spec advertises ``page_size`` max = 100, but the server accepts
# larger values in practice (verified up to 5000). The library is for
# data-pipeline use, not paginated UI display, so we default aggressively
# to cut round-trips: most funds fit in a single request, larger ones
# still drain via the pagination loop in :meth:`get_paginated`.
_DEFAULT_PAGE_SIZE = 1000

# Hard cap on pagination iterations. At ``_DEFAULT_PAGE_SIZE = 1000``
# rows per page, this is 10M items — far beyond any realistic FundCloud
# response. The cap exists purely as a runaway guard against a server
# returning broken pagination metadata (e.g., has_next=True forever, or
# total_pages exceeding actual page count). Hitting it raises
# :class:`fundcloud.errors.TransientError` so callers can decide whether
# to retry or surface the failure.
_MAX_PAGES = 10_000


class FundCloudClient:
    """Auth + pagination wrapper around :class:`HttpClient`.

    Parameters
    ----------
    api_key
        FundCloud API key (``fc_live_…``). Falls back to the
        ``FUNDCLOUD_API_KEY`` env var. Raises :class:`AuthError` when
        neither is set.
    base_url
        Override for the API base URL (useful in tests).
    timeout
        Per-request timeout in seconds (default 30).

    Notes
    -----
    Retry policy (4 attempts, exponential backoff on 5xx / 429 /
    ``httpx.ConnectError`` / ``httpx.ReadTimeout``) is inherited from
    the shared :class:`HttpClient`. 4xx responses map to typed
    exceptions from :mod:`fundcloud.errors`:

    * 401 → :class:`fundcloud.errors.AuthError`
    * 403 → :class:`fundcloud.errors.QuotaError`
    * 404 → :class:`fundcloud.errors.NotFoundError`
    * 5xx / 429 (retries exhausted) → :class:`fundcloud.errors.TransientError`
    * other 4xx → ``httpx.HTTPStatusError`` reraised unchanged

    The underlying ``httpx.Client`` is kept open for the lifetime of the
    :class:`FundCloudClient`; call :meth:`close` (or use ``with`` as a
    context manager) for explicit cleanup. Python's finalizer will also
    close the session on garbage collection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = FUNDCLOUD_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get(FUNDCLOUD_ENV_VAR)
        if not key:
            msg = (
                "FundCloud requires an API key. Pass `api_key=` or set the "
                f"{FUNDCLOUD_ENV_VAR} environment variable."
            )
            raise AuthError(msg)
        self._base_url = base_url
        self._http = HttpClient(
            base_url=base_url,
            headers={"X-API-Key": key, "Accept": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FundCloudClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --------------------------------------------------------------- GET
    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """GET ``path`` and return the decoded JSON body.

        Maps FundCloud HTTP error codes to the typed hierarchy in
        :mod:`fundcloud.errors`. See class docstring for the full table.
        """
        httpx = require_httpx()
        try:
            return self._http.get_json(path, params=params)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                msg = f"FundCloud authentication failed ({status}) for {path}"
                raise AuthError(msg) from e
            if status == 403:
                msg = f"FundCloud access denied or quota exhausted ({status}) for {path}"
                raise QuotaError(msg) from e
            if status == 404:
                msg = f"FundCloud resource not found ({status}) for {path}"
                raise NotFoundError(msg) from e
            raise
        except TransientHttpError as e:
            msg = f"FundCloud transient error (retries exhausted): {e}"
            raise TransientError(msg) from e

    # --------------------------------------------------------------- pagination
    def get_paginated(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over ``data[]`` across pages until exhausted.

        Uses the API's 1-indexed ``page`` + ``page_size`` query parameters
        (defaults: ``page=1``, ``page_size`` = ``_DEFAULT_PAGE_SIZE``).
        Stop condition prefers ``meta.has_next`` when present and falls
        back to ``meta.page < meta.total_pages`` — the live API
        currently omits ``has_next`` from some responses, so the
        fallback is what actually drains the pages in practice.
        """
        current: dict[str, Any] = dict(params or {})
        current.setdefault("page", 1)
        current.setdefault("page_size", _DEFAULT_PAGE_SIZE)

        for _ in range(_MAX_PAGES):
            payload = self.get(path, params=current)
            if not isinstance(payload, dict):
                # Fail closed: returning here would yield partial data
                # silently, which is much worse than a typed error for
                # data-pipeline consumers.
                msg = (
                    f"FundCloud returned non-object payload for {path!r}: "
                    f"{type(payload).__name__}. Expected JSON object with "
                    f"`data` and `meta` keys."
                )
                raise MalformedDataError(msg)
            data = payload.get("data", [])
            if isinstance(data, list):
                yield from data
            meta = payload.get("meta")
            if meta is not None and not isinstance(meta, Mapping):
                msg = (
                    f"FundCloud returned non-mapping `meta` for {path!r}: "
                    f"{type(meta).__name__}. Cannot determine pagination state."
                )
                raise MalformedDataError(msg)
            if _has_more_pages(meta or {}, current["page"]):
                current["page"] = int(current["page"]) + 1
            else:
                return
        # Loop exited via for-else: page cap hit without natural termination,
        # which means the server's pagination metadata is broken or we're
        # iterating something pathological. Bail out loudly rather than
        # accumulating data forever.
        msg = (
            f"FundCloud pagination did not terminate within {_MAX_PAGES} pages "
            f"for path={path!r}. The server likely returned a malformed "
            f"`meta` block (e.g., has_next=True forever or total_pages exceeding "
            f"real page count). Aborting to avoid runaway memory."
        )
        raise TransientError(msg)


def _has_more_pages(meta: Mapping[str, Any], current_page: int) -> bool:
    """Decide whether to fetch the next page.

    Prefers explicit ``has_next`` (present in the OpenAPI spec). Falls
    back to ``page < total_pages`` when the server omits ``has_next``,
    which the FundCloud public API currently does for several listing
    endpoints. As a final fallback, returns False — better to stop
    early on a malformed response than to loop forever.
    """
    has_next = meta.get("has_next")
    if isinstance(has_next, bool):
        return has_next
    page = meta.get("page", current_page)
    total_pages = meta.get("total_pages")
    if isinstance(page, int) and isinstance(total_pages, int):
        return page < total_pages
    return False
