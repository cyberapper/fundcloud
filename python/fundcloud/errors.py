"""Library-wide typed error hierarchy.

Every module in :mod:`fundcloud` that raises a "this is a Fundcloud problem,
not a generic Python problem" error raises from one of the classes in this
module. Users catch ``fundcloud.errors.AuthError`` / ``TransientError`` /
etc. uniformly regardless of which source or component produced the error.

Mapping conventions (not enforced by the types themselves — each raising
site documents its own mapping):

* **Auth / credentials** — bad or missing API key, expired token, 401 from
  a REST API: raise :class:`AuthError`.
* **Quota / rate limit** — 403 where the body / headers indicate a quota
  or rate-limit reason: raise :class:`QuotaError`. For true 429 rate-limit
  responses that the transport layer is already retrying, raise
  :class:`TransientError`.
* **Not found** — 404, missing fund/account id, unknown dataset: raise
  :class:`NotFoundError`.
* **Ambiguous lookup** — a method that expected to resolve a single entity
  (e.g., ``to_portfolio()`` with ``fund_id=None`` on a credential that
  sees multiple funds) finds more than one match: raise
  :class:`AmbiguousError`.
* **Transient** — 5xx responses, connection errors, read timeouts, 429s:
  raise :class:`TransientError`. The shared ``HttpClient`` already retries
  these automatically; raising is reserved for when retries are exhausted.
* **Malformed source data** — the source payload is structurally invalid:
  unknown section, missing essential column, unparseable date, etc.: raise
  :class:`MalformedDataError`. Distinct from :class:`NotFoundError`
  (the resource exists at the source but wasn't found by id) and
  :class:`AmbiguousError` (a lookup matched multiple).
"""

from __future__ import annotations

__all__ = [
    "AmbiguousError",
    "AuthError",
    "FundcloudError",
    "MalformedDataError",
    "NotFoundError",
    "QuotaError",
    "TransientError",
]


class FundcloudError(Exception):
    """Root of the library's typed error hierarchy.

    Catching :class:`FundcloudError` matches every library-raised error.
    """


class AuthError(FundcloudError):
    """Authentication or authorization failure.

    Typical causes: missing API key, invalid API key, expired token,
    401 response from a REST API.
    """


class QuotaError(FundcloudError):
    """Rate limit or quota exhausted.

    Raised when a 403 response indicates quota exhaustion. Transient 429
    responses that the transport retries automatically are surfaced as
    :class:`TransientError` once retries are exhausted.
    """


class NotFoundError(FundcloudError):
    """Requested resource does not exist.

    Typical causes: 404 from a REST API, unknown fund/account id,
    missing dataset key.
    """


class AmbiguousError(FundcloudError, ValueError):
    """A lookup expecting a single match found multiple.

    Also a :class:`ValueError` so callers who catch ``ValueError`` (the
    natural Python choice for "you gave me bad inputs") still see it.
    """


class TransientError(FundcloudError):
    """A transient failure that persisted after the transport's retries.

    Typical causes: repeated 5xx, connection errors, read timeouts, or
    sustained 429 rate-limit responses. The shared HTTP client retries
    these automatically; this exception surfaces only when the retry
    budget is exhausted.
    """


class MalformedDataError(FundcloudError, ValueError):
    """Source data is structurally invalid.

    Typical causes: unknown section in a CSV / JSON payload, missing
    essential column, unparseable timestamp, mismatched section header.
    Distinct from :class:`NotFoundError` (resource exists at the source
    but isn't found by id) and :class:`AmbiguousError` (a lookup
    matched multiple).

    Also a :class:`ValueError` so callers catching ``ValueError`` (the
    natural Python choice for "you gave me bad inputs") still see it.
    """
