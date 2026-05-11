"""Run every registered pattern detector in one call.

The "TA-Lib idiom" for this kind of bulk scan is::

    candle_names = talib.get_function_groups()['Pattern Recognition']
    for name in candle_names:
        df[name] = getattr(talib, name)(open_, high, low, close)

TA-Lib stops there — it returns one integer column per pattern. This
library has richer events (pivots, breakout level, quality score,
variant), so :func:`scan_all_patterns` returns the unified events frame
across every registered :class:`PatternIndicator`, ready for
:func:`fundcloud.metrics.feature_quality.evaluate` /
:func:`~fundcloud.metrics.feature_quality.per_pattern`.

Discovery is registry-driven — anything registered via
:func:`fundcloud.features.indicators.base.register_indicator` whose class
extends :class:`PatternIndicator` is included automatically. Plug-in
detectors get the same treatment as built-ins.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from fundcloud.features.indicators.base import registered_indicators
from fundcloud.features.patterns._base import PatternIndicator
from fundcloud.features.patterns._condition import PatternCondition
from fundcloud.features.patterns._enums import Pattern
from fundcloud.features.patterns._events import EVENTS_COLUMNS

__all__ = ["registered_pattern_indicators", "scan_all_patterns"]


def registered_pattern_indicators() -> dict[str, type[PatternIndicator]]:
    """Subset of :func:`registered_indicators` that detects patterns.

    Returns the registry filtered to :class:`PatternIndicator` subclasses,
    keyed by registered name (the same string the ``@register_indicator``
    decorator was called with — e.g. ``"head_and_shoulders"``).

    Built-ins and plug-in detectors share this registry, so anything
    written by users that extends :class:`PatternIndicator` and registers
    a name shows up here automatically.
    """
    return {
        name: cls
        for name, cls in registered_indicators().items()
        if isinstance(cls, type) and issubclass(cls, PatternIndicator)
    }


def scan_all_patterns(
    bars: pd.DataFrame,
    *,
    patterns: Iterable[Pattern | str] | None = None,
    conditions: Mapping[Pattern | str, PatternCondition] | None = None,
    params: Mapping[Pattern | str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run every registered pattern detector and return a unified events frame.

    Parameters
    ----------
    bars
        MultiIndex ``Bars`` frame with ``(field, asset)`` columns and a
        :class:`pd.DatetimeIndex`. Same shape every
        :class:`PatternIndicator` consumes.
    patterns
        Optional subset. Iterable of :class:`Pattern` enum values or
        their string names. ``None`` (default) runs every registered
        pattern.
    conditions
        Optional per-pattern :class:`PatternCondition` overrides. Keys
        are :class:`Pattern` enum values or names; missing keys use the
        detector's class-level default.
    params
        Optional per-pattern parameter overrides forwarded to the
        indicator constructor (e.g. ``{Pattern.DOUBLE_TOP:
        {"peak_tolerance": 0.01}}``). Missing keys use defaults.

    Returns
    -------
    pd.DataFrame
        Concatenated events table with columns :data:`EVENTS_COLUMNS`.
        The ``pattern`` column distinguishes each row's source detector.
        Empty if no detector found any formation.

    Raises
    ------
    KeyError
        If a name in ``patterns``/``conditions``/``params`` is not a
        registered pattern.
    """
    registry = registered_pattern_indicators()
    if not registry:
        return pd.DataFrame(columns=list(EVENTS_COLUMNS))

    selected = _select_patterns(registry, patterns)
    cond_map = _normalise_keyed(conditions)
    param_map = _normalise_keyed(params)
    _validate_keys(cond_map, registry, kind="conditions")
    _validate_keys(param_map, registry, kind="params")

    frames: list[pd.DataFrame] = []
    for name in selected:
        cls = registry[name]
        kwargs: dict[str, Any] = dict(param_map.get(name, {}))
        if name in cond_map:
            kwargs["condition"] = cond_map[name]
        indicator = cls(**kwargs)
        events = indicator.events(bars)
        if not events.empty:
            frames.append(events)

    if not frames:
        return pd.DataFrame(columns=list(EVENTS_COLUMNS))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- helpers


def _select_patterns(
    registry: Mapping[str, type[PatternIndicator]],
    patterns: Iterable[Pattern | str] | None,
) -> list[str]:
    if patterns is None:
        return sorted(registry.keys())
    selected: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        name = _pattern_to_name(p)
        if name not in registry:
            msg = (
                f"pattern {name!r} is not registered. "
                f"Registered patterns: {sorted(registry.keys())}"
            )
            raise KeyError(msg)
        if name not in seen:
            selected.append(name)
            seen.add(name)
    return selected


def _normalise_keyed(
    mapping: Mapping[Pattern | str, Any] | None,
) -> dict[str, Any]:
    if mapping is None:
        return {}
    return {_pattern_to_name(k): v for k, v in mapping.items()}


def _validate_keys(
    keyed: Mapping[str, Any],
    registry: Mapping[str, type[PatternIndicator]],
    *,
    kind: str,
) -> None:
    unknown = sorted(k for k in keyed if k not in registry)
    if unknown:
        msg = (
            f"{kind} contains unregistered patterns: {unknown}. "
            f"Registered patterns: {sorted(registry.keys())}"
        )
        raise KeyError(msg)


def _pattern_to_name(value: Pattern | str) -> str:
    """Normalise a user-supplied identifier to the registry key.

    Accepts the :class:`Pattern` enum (built-ins) or any string (built-in
    or plug-in name). Validation against the registry happens later — we
    deliberately don't coerce through :class:`Pattern` here so plug-in
    names register without being members of the static enum.
    """
    if isinstance(value, Pattern):
        return value.value
    return str(value)
