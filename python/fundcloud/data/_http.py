"""Shared HTTP client for network data backends.

Thin wrapper around :mod:`httpx` with tenacity-based retry for transient
failures (5xx, connection errors, rate-limit 429). Every network backend in
:mod:`fundcloud.data` uses this instead of hand-rolling its own client, so
retry policy stays consistent.

``httpx`` is pulled in via the ``fundcloud[data-fmp]`` / ``fundcloud[data-av]``
extras. Backends that don't ship with httpx bail at construction time with a
friendly :class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

__all__ = ["HttpClient", "require_httpx"]


def require_httpx() -> Any:
    """Import and return the ``httpx`` module, with a helpful error if absent."""
    try:
        import httpx
    except ImportError as e:
        msg = (
            "httpx is required for network data backends. "
            "Install with: uv add 'fundcloud[data]' "
            "(or the specific provider extra, e.g. 'fundcloud[data-fmp]')."
        )
        raise ImportError(msg) from e
    return httpx


class HttpClient:
    """Blocking HTTP client with automatic retry."""

    def __init__(
        self,
        *,
        base_url: str = "",
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        httpx = require_httpx()
        self._httpx = httpx
        self._client = httpx.Client(
            base_url=base_url,
            params=dict(params or {}),
            headers=dict(headers or {}),
            timeout=timeout,
        )
        self._max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None) -> Any:
        """GET ``url`` and return the JSON body with transient retry."""
        httpx = self._httpx

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                TransientHttpError,
            )),
        )
        def _do() -> Any:
            resp = self._client.get(url, params=dict(params or {}))
            if resp.status_code >= 500 or resp.status_code == 429:
                msg = f"transient HTTP {resp.status_code} for {url}"
                raise TransientHttpError(msg)
            resp.raise_for_status()
            return resp.json()

        return _do()


class TransientHttpError(RuntimeError):
    """Raised when a 5xx or 429 response is received. Retried automatically."""
