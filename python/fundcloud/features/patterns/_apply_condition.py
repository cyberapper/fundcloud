"""Fill the events table's ``target_price`` and ``stop_price`` columns.

The detector emits geometric pivots and an entry / breakout level. Those
are sufficient to grade the *feature*, but a *strategy* needs explicit
target and stop levels — and those depend on user choice
(:class:`PatternCondition`'s :class:`TargetMethod` and
:class:`StopMethod`).

``apply_condition(events, condition, bars)`` returns a copy of the
events table with the target / stop columns filled per the condition,
ready for downstream backtesting via :class:`PatternStrategy` or
target-aware grading via :func:`feature_quality.evaluate`.

**Pattern height** is the load-bearing measurement. We compute it as:

* Bullish events: ``entry - min(low_pivot_prices)``
* Bearish events: ``max(high_pivot_prices) - entry``

The pivot list is read from the events table's ``pivots`` column (a
``list[dict]`` populated by :func:`build_events_frame`). When that list
lacks usable extremes for the direction (e.g., a malformed event), we
fall back to ``1 × ATR`` as the height — keeps the pipeline robust.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fundcloud.features.patterns._condition import PatternCondition
from fundcloud.features.patterns._enums import Direction, StopMethod, TargetMethod

__all__ = ["apply_condition"]


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Wilder's ATR. Returns ``np.nan`` for the first ``window`` bars.

    Mirrors the helper in ``fundcloud.metrics.feature_quality``; kept
    inline here to avoid creating a layering dependency from
    ``features.patterns`` onto ``metrics``.
    """
    n = len(close)
    if n == 0 or window < 1:
        return np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ])
    atr = np.full(n, np.nan)
    if n < window:
        return atr
    atr[window - 1] = np.mean(tr[:window])
    for t in range(window, n):
        atr[t] = (atr[t - 1] * (window - 1) + tr[t]) / window
    return atr


def _select_asset(bars: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Pull a single asset's OHLCV columns from a MultiIndex bars frame.

    Drops rows missing any OHLC field — same convention as
    ``feature_quality._select_asset`` so leading-NaN listing history
    doesn't poison ATR.
    """
    if not isinstance(bars.columns, pd.MultiIndex):
        msg = "bars must have MultiIndex (field, asset) columns"
        raise TypeError(msg)
    return bars.xs(asset, level=-1, axis=1).dropna(subset=["open", "high", "low", "close"])  # type: ignore[call-overload, no-any-return]


def _direction_sign(direction: Direction | str) -> int:
    """+1 for bullish, -1 for bearish, 0 otherwise."""
    value = direction.value if isinstance(direction, Direction) else str(direction).lower()
    if value == "bullish":
        return 1
    if value == "bearish":
        return -1
    return 0


def _pivot_prices(pivots: list[dict[str, Any]], kind: str) -> list[float]:
    """Extract finite numeric prices for pivots of the given ``kind``.

    Tolerant of malformed entries: missing / non-numeric / non-finite
    ``price`` values are skipped rather than raised, so a single bad
    pivot can't abort the whole apply pass.
    """
    out: list[float] = []
    for p in pivots:
        if p.get("kind") != kind:
            continue
        v = p.get("price")
        if v is None:
            continue
        try:
            price = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price):
            continue
        out.append(price)
    return out


def _pattern_height(
    pivots: list[dict[str, Any]],
    entry: float,
    sign: int,
    fallback: float,
) -> float:
    """Distance from entry to the most adverse pivot in the formation.

    Bullish: ``entry - min(low_pivot_prices)`` — how far below entry
    the support reached.

    Bearish: ``max(high_pivot_prices) - entry`` — how far above entry
    the resistance reached.

    ``fallback`` (typically 1×ATR) is returned when the pivot list
    lacks usable extremes — e.g., an event that survived dedup with no
    pivots of the relevant kind.
    """
    if not pivots or sign == 0:
        return fallback
    if sign > 0:
        lows = _pivot_prices(pivots, "LOW")
        if not lows:
            return fallback
        height = entry - min(lows)
    else:
        highs = _pivot_prices(pivots, "HIGH")
        if not highs:
            return fallback
        height = max(highs) - entry
    if not np.isfinite(height) or height <= 0:
        return fallback
    return height


def _resolve_target(
    *,
    entry: float,
    sign: int,
    pattern_height: float,
    atr: float,
    method: TargetMethod,
    atr_multiple: float,
    fib_multiple: float,
) -> float:
    """Compute the target price per ``method``."""
    if method is TargetMethod.MEASURED_MOVE:
        return entry + sign * pattern_height
    if method is TargetMethod.FIB_1_618:
        return entry + sign * fib_multiple * pattern_height
    if method is TargetMethod.FIXED_ATR:
        return entry + sign * atr_multiple * atr
    msg = f"unsupported target method: {method!r}"
    raise ValueError(msg)


def _resolve_stop(
    *,
    entry: float,
    sign: int,
    pivots: list[dict[str, Any]],
    atr: float,
    method: StopMethod,
    atr_multiple: float,
    fixed_pct: float,
) -> float:
    """Compute the stop price per ``method``.

    BELOW_PIVOT for a bearish trade reads as "above the highest pivot"
    — the enum name is from the bullish-default perspective.
    """
    if method is StopMethod.BELOW_PIVOT:
        if sign > 0:
            lows = _pivot_prices(pivots, "LOW")
            return min(lows) if lows else entry - atr_multiple * atr
        highs = _pivot_prices(pivots, "HIGH")
        return max(highs) if highs else entry + atr_multiple * atr
    if method is StopMethod.ATR_MULTIPLE:
        return entry - sign * atr_multiple * atr
    if method is StopMethod.FIXED_PCT:
        return entry * (1 - sign * fixed_pct)
    msg = f"unsupported stop method: {method!r}"
    raise ValueError(msg)


def apply_condition(
    events: pd.DataFrame,
    condition: PatternCondition,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of ``events`` with ``target_price`` and ``stop_price``
    filled per the supplied :class:`PatternCondition`.

    Parameters
    ----------
    events
        Canonical events table (see :data:`EVENTS_COLUMNS`).
    condition
        Entry / exit / target / stop rules.
    bars
        OHLCV MultiIndex frame the events came from. Used to compute
        ATR at the breakout bar for ATR-relative target / stop methods,
        and to provide a fallback when the events table is missing
        useful pivots.

    Notes
    -----
    * Events whose ``breakout_ts`` isn't in the bar index keep their
      original NaN target / stop — same convention as
      :func:`feature_quality.evaluate`.
    * ATR is computed once per asset and cached, so the function is
      O(events) on the hot path.
    """
    if events.empty:
        return events.copy()  # type: ignore[no-any-return]

    out = events.copy()
    asset_cache: dict[str, pd.DataFrame | None] = {}
    atr_cache: dict[str, np.ndarray] = {}

    targets: list[float] = []
    stops: list[float] = []
    for _, ev in out.iterrows():
        asset = str(ev["asset"])
        sign = _direction_sign(ev["direction"])
        if sign == 0:
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue

        if asset not in asset_cache:
            try:
                asset_cache[asset] = _select_asset(bars, asset)
            except KeyError:
                asset_cache[asset] = None
        ab = asset_cache[asset]
        if ab is None:
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue

        ts = ev.get("breakout_ts")
        if ts is None or pd.isna(ts):
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue
        try:
            pos = ab.index.get_loc(ts)
        except KeyError:
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue
        if isinstance(pos, slice):
            pos = pos.start

        if asset not in atr_cache:
            atr_cache[asset] = _wilder_atr(
                ab["high"].to_numpy(np.float64),
                ab["low"].to_numpy(np.float64),
                ab["close"].to_numpy(np.float64),
                condition.atr_window,
            )
        atr = float(atr_cache[asset][pos])
        atr_valid = np.isfinite(atr) and atr > 0
        if not atr_valid and (
            condition.target_method is TargetMethod.FIXED_ATR
            or condition.stop_method is StopMethod.ATR_MULTIPLE
        ):
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue

        entry_raw = ev.get("breakout_price")
        if entry_raw is None or pd.isna(entry_raw):
            entry_raw = ev.get("entry_price")
        if entry_raw is None or pd.isna(entry_raw):
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue
        entry = float(entry_raw)

        pivots = ev.get("pivots") or []
        fallback_height = atr if atr_valid else float("nan")
        height = _pattern_height(pivots, entry, sign, fallback=fallback_height)
        if not np.isfinite(height) or height <= 0:
            targets.append(float("nan"))
            stops.append(float("nan"))
            continue

        targets.append(
            _resolve_target(
                entry=entry,
                sign=sign,
                pattern_height=height,
                atr=atr,
                method=condition.target_method,
                atr_multiple=condition.atr_multiple,
                fib_multiple=condition.fib_target_multiple,
            )
        )
        stops.append(
            _resolve_stop(
                entry=entry,
                sign=sign,
                pivots=pivots,
                atr=atr,
                method=condition.stop_method,
                atr_multiple=condition.atr_multiple,
                fixed_pct=condition.fixed_pct,
            )
        )

    out["target_price"] = targets
    out["stop_price"] = stops
    return out  # type: ignore[no-any-return]
