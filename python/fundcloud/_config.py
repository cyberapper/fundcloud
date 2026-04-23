"""Runtime configuration.

A minimal, process-local config object for library defaults (annualisation
factor, risk-free rate, etc.). Users can mutate it at import time or scope
changes with the ``config()`` context manager.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from threading import Lock

__all__ = ["Config", "config", "get_config", "set_config"]


@dataclass(frozen=True, slots=True)
class Config:
    """Library-wide defaults.

    Attributes
    ----------
    periods_per_year
        Default annualisation factor (252 for business-day equities).
    risk_free_rate
        Default continuously-compounded risk-free rate per period, in the same
        units as the returns the metric is applied to.
    tol
        Default floating-point tolerance for equality checks.
    """

    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    tol: float = 1e-12
    # Reserved for future use without breaking callers that pass kwargs.
    extra: dict[str, object] = field(default_factory=dict)


_LOCK = Lock()
_CURRENT = Config()


def get_config() -> Config:
    """Return the current library config."""
    return _CURRENT


def set_config(**changes: object) -> Config:
    """Replace fields on the current config and return the new one."""
    global _CURRENT
    with _LOCK:
        _CURRENT = replace(_CURRENT, **changes)  # type: ignore[arg-type]
        return _CURRENT


@contextmanager
def config(**changes: object) -> Iterator[Config]:
    """Temporarily override config fields.

    Example
    -------
    >>> from fundcloud._config import config
    >>> with config(periods_per_year=365):
    ...     ...  # computations see 365 here
    """
    global _CURRENT
    with _LOCK:
        previous = _CURRENT
        _CURRENT = replace(_CURRENT, **changes)  # type: ignore[arg-type]
    try:
        yield _CURRENT
    finally:
        with _LOCK:
            _CURRENT = previous
