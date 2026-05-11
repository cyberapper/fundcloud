"""Empirical per-pattern direction inference from outcome data.

Detection is direction-agnostic — every formation comes out of the
scanner as pure geometry. Whether a pattern resolves long or short is
an empirical question, not a textbook prior, and this module answers
it: given an events frame (typically from
:func:`fundcloud.features.patterns.scan_all_patterns`) and the OHLCV
bars they came from, it computes a per-pattern direction map suitable
for :class:`fundcloud.strategies.PatternStrategy` or
:func:`fundcloud.features.patterns.apply_condition`.

The classification rule is intentionally simple: for each pattern, take
the **mean forward return** at ``horizon`` bars across every event of
that pattern. A positive mean → :attr:`Direction.BULLISH` (go long);
a negative mean → :attr:`Direction.BEARISH` (go short). Patterns with
fewer than ``min_samples`` events fall back to the user-supplied
``default`` rather than committing to a noisy estimate.

The "mean forward return" choice is deliberately *not* a Sharpe-like
signal-to-noise ratio: a pattern that produces a small but consistent
edge gets the right sign even if its volatility is high. Replace this
function with an MFE/MAE-skew or ML-scored variant when you've earned
the right to (and have somewhere to put it on the roadmap).

Workflow::

    from fundcloud.features.patterns import scan_all_patterns
    from fundcloud.metrics import pattern_direction as pd_

    events = scan_all_patterns(bars)
    direction_map = pd_.direction_map_from_outcomes(events, bars, horizon=20)
    strat = PatternStrategy(indicator, direction_map=direction_map)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fundcloud.features.patterns._enums import Direction, Pattern

__all__ = ["direction_map_from_outcomes", "mean_forward_returns"]


DEFAULT_HORIZON = 20
DEFAULT_MIN_SAMPLES = 30
# Mean forward returns whose absolute value falls below this floor are
# treated as "too close to zero to commit" and fall back to ``default``.
# 0.0 means "any non-zero mean wins"; a positive floor adds a safety
# margin against noise. Set conservatively low so the gate doesn't
# swallow real-but-small edges.
DEFAULT_NULL_THRESHOLD = 0.0


def direction_map_from_outcomes(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    default: Direction = Direction.BULLISH,
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
) -> dict[str, Direction]:
    """Build a per-pattern direction lookup from realised outcomes.

    Parameters
    ----------
    events
        Canonical events frame (the schema emitted by detectors and
        :func:`scan_all_patterns`). Must include ``pattern``, ``asset``,
        ``breakout_ts``, ``breakout_level``.
    bars
        MultiIndex ``Bars`` frame the events came from. Same shape every
        :class:`PatternIndicator` consumes.
    horizon
        Forward-return window in bars. Defaults to 20 — the same
        canonical horizon :func:`feature_quality.evaluate` uses.
    min_samples
        Minimum event count per pattern to commit to an empirical
        direction. Below this, falls back to ``default``. Defaults to 30
        — the smallest count where the sign of the mean forward return
        is reliable on noisy real-world series.
    default
        Direction returned for patterns with fewer than ``min_samples``
        events or whose mean return is within ``null_threshold`` of
        zero. Defaults to :attr:`Direction.BULLISH` (go long), matching
        the rest of the library's "assume long until proven otherwise"
        convention.
    null_threshold
        Absolute mean-return floor. Means with ``|mean| <= null_threshold``
        are treated as undecided and use ``default``. Defaults to 0.0
        ("any non-zero mean wins"); raise for a more conservative gate.

    Returns
    -------
    dict[str, Direction]
        Pattern name → direction. Keys are the registered pattern names
        (matching :class:`Pattern` enum values). Patterns absent from
        ``events`` are absent from the output — callers that consume
        this map (``apply_condition`` / ``PatternStrategy``) already
        fall back to their own default for missing keys.

    Examples
    --------
    >>> events = scan_all_patterns(bars)                                 # doctest: +SKIP
    >>> dmap = direction_map_from_outcomes(events, bars, horizon=20)     # doctest: +SKIP
    >>> dmap["double_top"]                                               # doctest: +SKIP
    <Direction.BEARISH: 'bearish'>
    """
    if horizon <= 0:
        msg = "horizon must be > 0"
        raise ValueError(msg)
    if min_samples <= 0:
        msg = "min_samples must be > 0"
        raise ValueError(msg)
    if null_threshold < 0:
        msg = "null_threshold must be non-negative"
        raise ValueError(msg)

    if events.empty:
        return {}

    means = mean_forward_returns(events, bars, horizon=horizon)

    out: dict[str, Direction] = {}
    for pattern_name, stats in means.items():
        if stats["count"] < min_samples:
            out[pattern_name] = default
            continue
        mean_ret = stats["mean"]
        if not np.isfinite(mean_ret) or abs(mean_ret) <= null_threshold:
            out[pattern_name] = default
            continue
        out[pattern_name] = Direction.BULLISH if mean_ret > 0 else Direction.BEARISH
    return out


def mean_forward_returns(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, dict[str, float]]:
    """Per-pattern mean and event-count of forward close-to-close returns.

    Returned shape is ``{pattern_name: {"count": int, "mean": float}}``.
    Empty events → empty dict. Events whose breakout falls off the bar
    grid or whose horizon exceeds the available lookahead are dropped
    silently — same convention as :func:`feature_quality.evaluate`.
    """
    if horizon <= 0:
        msg = "horizon must be > 0"
        raise ValueError(msg)
    if events.empty:
        return {}
    if not isinstance(bars.columns, pd.MultiIndex):
        msg = "bars must have MultiIndex (field, asset) columns"
        raise TypeError(msg)

    asset_close: dict[str, pd.Series] = {}

    per_pattern: dict[str, list[float]] = {}
    for _, ev in events.iterrows():
        pattern_name = _pattern_value(ev.get("pattern"))
        asset = str(ev.get("asset"))
        ts = ev.get("breakout_ts")
        if pattern_name is None or pd.isna(ts):
            continue

        if asset not in asset_close:
            try:
                series = bars.xs(asset, level=-1, axis=1)["close"].dropna()
            except KeyError:
                asset_close[asset] = pd.Series(dtype=np.float64)
                continue
            asset_close[asset] = series
        close = asset_close[asset]
        if close.empty:
            continue

        try:
            pos = close.index.get_loc(ts)
        except KeyError:
            continue
        if isinstance(pos, slice):
            pos = pos.start
        if pos + horizon >= len(close):
            continue

        entry = float(close.iloc[pos])
        if entry == 0 or not np.isfinite(entry):
            continue
        future = float(close.iloc[pos + horizon])
        if not np.isfinite(future):
            continue
        per_pattern.setdefault(pattern_name, []).append((future - entry) / entry)

    return {
        name: {"count": len(returns), "mean": float(np.mean(returns))}
        for name, returns in per_pattern.items()
    }


def _pattern_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Pattern):
        return value.value
    return str(value)
