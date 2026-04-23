"""Technical indicators wrapped as sklearn-compatible transformers.

Every TA-Lib function is available as a class named after the function
(``SMA``, ``RSI``, ``MACD``, …). Under the hood we auto-generate these at
import time from :mod:`fundcloud.features.indicators._talib_autogen`.

When TA-Lib is not installed (the ``fundcloud[ta]`` extra is missing),
imports like ``from fundcloud.features.indicators import SMA`` raise a
helpful :class:`ImportError`. Custom indicators registered via
:func:`fundcloud.features.indicators.base.register_indicator` remain
available either way.
"""

from __future__ import annotations

from typing import Any

from fundcloud.features.indicators import _talib_autogen
from fundcloud.features.indicators.base import (
    IndicatorSpec,
    register_indicator,
    registered_indicators,
)

__all__ = [
    "GROUPS",
    "IndicatorSpec",
    "list_indicators",
    "register_indicator",
    "registered_indicators",
]

GROUPS: dict[str, list[str]] = _talib_autogen.GROUPS


def list_indicators() -> list[str]:
    """All indicator names resolvable via ``from fundcloud.features.indicators import X``."""
    return sorted(_talib_autogen.GENERATED.keys()) + sorted(registered_indicators().keys())


def __getattr__(name: str) -> Any:
    gen = _talib_autogen.GENERATED
    if name in gen:
        return gen[name]
    custom = registered_indicators()
    if name in custom:
        return custom[name]
    if not _talib_autogen.TALIB_AVAILABLE and name.isupper():
        msg = (
            f"Indicator {name!r} requires TA-Lib. Install the C library "
            "(e.g. `brew install ta-lib`), then `uv add 'fundcloud[ta]'`."
        )
        raise ImportError(msg)
    raise AttributeError(f"module 'fundcloud.features.indicators' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose every auto-generated TA-Lib indicator + custom registrations to the REPL / IDE."""
    return sorted(
        set(__all__)
        | set(globals())
        | set(_talib_autogen.GENERATED.keys())
        | set(registered_indicators().keys())
    )
