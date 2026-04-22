"""Auto-generate :class:`IndicatorSpec` subclasses for every TA-Lib function.

At import time we introspect :func:`talib.get_functions` and
:func:`talib.get_function_groups` to create a wrapper class per function
(``SMA``, ``RSI``, ``MACD``, …). Each generated class:

* sets ``talib_name`` so the base :meth:`_compute` dispatches correctly,
* copies TA-Lib's default parameters into ``default_params`` (so
  ``get_params`` returns sensible values),
* sets ``inputs`` / ``outputs`` from TA-Lib's ``input_names`` /
  ``output_names``.

If TA-Lib is not installed the module degrades gracefully: no classes are
produced, and an ``ImportError`` is attached to ``TALIB_ERROR`` so callers
can surface a friendly message.
"""

from __future__ import annotations

from typing import Any

from fundcloud.features.indicators.base import IndicatorSpec

__all__ = ["GENERATED", "GROUPS", "TALIB_AVAILABLE", "TALIB_ERROR"]


TALIB_ERROR: Exception | None = None
GENERATED: dict[str, type[IndicatorSpec]] = {}
GROUPS: dict[str, list[str]] = {}


try:
    import talib  # type: ignore[import-not-found]
    from talib import abstract  # type: ignore[import-not-found]

    TALIB_AVAILABLE = True
except ImportError as e:  # pragma: no cover - only hit when TA-Lib absent
    TALIB_AVAILABLE = False
    TALIB_ERROR = e


def _normalise_name(raw: str) -> str:
    """TA-Lib uses lowercase fields like ``open``; use the same."""
    return raw.strip().lower().replace(" ", "_")


def _make_class(function_name: str) -> type[IndicatorSpec]:
    info = abstract.Function(function_name).info
    defaults = dict(info.get("parameters", {})) if isinstance(info, dict) else {}
    inputs_raw = info.get("input_names", {}) if isinstance(info, dict) else {}
    # TA-Lib's ``input_names`` is a dict like {"price": "close"} or
    # {"prices": ["high", "low", "close"]} — flatten to a tuple.
    flat_inputs: list[str] = []
    for val in inputs_raw.values():
        if isinstance(val, (list, tuple)):
            flat_inputs.extend(_normalise_name(str(v)) for v in val)
        else:
            flat_inputs.append(_normalise_name(str(val)))
    inputs = tuple(flat_inputs) if flat_inputs else ("close",)

    outputs_raw = info.get("output_names", []) if isinstance(info, dict) else []
    outputs = tuple(_normalise_name(str(o)) for o in outputs_raw) or ("value",)

    cls = type(
        function_name,
        (IndicatorSpec,),
        {
            "__module__": "fundcloud.features.indicators",
            "__qualname__": function_name,
            "talib_name": function_name,
            "inputs": inputs,
            "outputs": outputs,
            "default_params": dict(defaults),
            "__doc__": _make_docstring(function_name, info),
        },
    )
    return cls  # type: ignore[return-value]


def _make_docstring(name: str, info: Any) -> str:
    group = info.get("group", "?") if isinstance(info, dict) else "?"
    display_name = info.get("display_name", name) if isinstance(info, dict) else name
    inputs = info.get("input_names", {}) if isinstance(info, dict) else {}
    outputs = info.get("output_names", []) if isinstance(info, dict) else []
    params = info.get("parameters", {}) if isinstance(info, dict) else {}
    return (
        f"{display_name} ({name}) — TA-Lib group: {group}.\n\n"
        f"Inputs: {inputs!r}\n"
        f"Outputs: {outputs!r}\n"
        f"Parameters (with defaults): {params!r}\n"
    )


if TALIB_AVAILABLE:
    GROUPS = {group: list(names) for group, names in talib.get_function_groups().items()}
    for _fn in talib.get_functions():
        try:
            GENERATED[_fn] = _make_class(_fn)
        except Exception:  # pragma: no cover — skip unintrospectable functions
            continue
