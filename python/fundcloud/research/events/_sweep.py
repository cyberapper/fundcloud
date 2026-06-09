"""Liquidity-sweep failure detector (``ev_sweep_fail``).

A *sweep failure* is a stop-run that immediately reverses: price pierces a known
swing level by a volatility-scaled margin, then closes back on the original side
within the same bar. The pierced level is liquidity; the close-back is the
rejection that names the bar a sweep, not a breakout.

* **bullish** (support sweep) — an eligible pivot-low support ``L`` is taken out
  below (``low[t] < L - eps * atr[t-1]``) but the bar closes back above it
  (``close[t] > L``). The reversal is up; the structural stop sits under the
  wick (``stop_ref_price = low[t]``).
* **bearish** (resistance sweep) — an eligible pivot-high resistance ``R`` is
  taken out above (``high[t] > R + eps * atr[t-1]``) but the bar closes back
  below it (``close[t] < R``). The reversal is down; the stop sits above the
  wick (``stop_ref_price = high[t]``).

Causality (``docs/guides/research/event-registry.md``): levels are
pivot-derived, so each neighbour-locks. A pivot of order ``k`` at index ``p`` is
only knowable at ``p + k``; the level it builds is *eligible* at bar ``t`` only
when ``p + k <= t``. The sweep margin uses the *prior* bar's ATR (``atr[t-1]``),
which is fully knowable at ``t``. ``confirmed_ts`` is therefore bar ``t`` with no
forward read; ``execution_ts`` / ``entry_ref_price`` legitimately reference the
next bar's open (NaT / NaN on the final bar). Bullish and bearish are separate
rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import confirmed_pivots, wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_sweep_fail"]

EVENT_ID = "ev_sweep_fail"
_TIMEFRAME = "1D"


def detect_sweep_fail(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    pivot_k: int = 3,
    eps: float = 0.10,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect liquidity-sweep failures on one asset's OHLCV bars.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame: lowercase ``open``/``high``/``low``/``close``/
        ``volume`` columns and a sorted, unique, tz-aware UTC ``DatetimeIndex``.
    asset
        Asset label copied into every emitted row.
    pivot_k
        Pivot order forwarded to
        :func:`fundcloud.research.events._causality.confirmed_pivots`. A pivot at
        index ``p`` becomes eligible only at bar ``p + pivot_k`` (neighbour-lock).
    eps
        Sweep margin in ATR units: a level is swept only when price pierces it by
        more than ``eps * atr[t-1]``.
    atr_n
        Wilder ATR length. The threshold at bar ``t`` uses ``atr[t-1]`` (known at
        ``t``); a bar with NaN prior ATR (warmup) can never fire.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. At most one
        row per ``(bar, direction)``, using the nearest swept level on ties.
        Empty input — or no sweep failures — yields an empty frame with those
        columns.
    """
    if bars.empty:
        return build_observations([])

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    index = bars.index
    n = len(index)

    atr = wilder_atr(high, low, close, atr_n)
    pivot_highs, pivot_lows = confirmed_pivots(high, low, pivot_k)

    params = {"pivot_k": int(pivot_k), "eps": float(eps), "atr_n": int(atr_n)}
    phash = params_hash(EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for t in range(n):
        prior_atr = atr[t - 1] if t >= 1 else np.nan
        if np.isnan(prior_atr):
            continue
        margin = eps * prior_atr

        bull = _nearest_support_sweep(pivot_lows, t, low[t], close[t], margin)
        if bull is not None:
            level = bull
            rows.append(
                _make_row(
                    asset=asset,
                    index=index,
                    t=t,
                    n=n,
                    open_=open_,
                    direction="bullish",
                    zone_lo=level - margin,
                    zone_hi=level,
                    stop_ref_price=low[t],
                    atr_at_confirm=atr[t],
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )

        bear = _nearest_resistance_sweep(pivot_highs, t, high[t], close[t], margin)
        if bear is not None:
            level = bear
            rows.append(
                _make_row(
                    asset=asset,
                    index=index,
                    t=t,
                    n=n,
                    open_=open_,
                    direction="bearish",
                    zone_lo=level,
                    zone_hi=level + margin,
                    stop_ref_price=high[t],
                    atr_at_confirm=atr[t],
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )

    return build_observations(rows)


def _nearest_support_sweep(
    pivot_lows: list[tuple[int, int, float]],
    t: int,
    low_t: float,
    close_t: float,
    margin: float,
) -> float | None:
    """Nearest eligible support level swept-and-reclaimed at bar ``t``, or ``None``.

    A support ``L`` qualifies when it is neighbour-locked (``p + k <= t``), pierced
    below by more than ``margin`` (``low_t < L - margin``) and reclaimed on close
    (``close_t > L``). Among matches the one nearest ``close_t`` (largest ``L``)
    is returned for determinism.
    """
    best: float | None = None
    for _p, confirm_idx, level in pivot_lows:
        if confirm_idx > t:
            continue
        if low_t < level - margin and close_t > level and (best is None or level > best):
            best = level
    return best


def _nearest_resistance_sweep(
    pivot_highs: list[tuple[int, int, float]],
    t: int,
    high_t: float,
    close_t: float,
    margin: float,
) -> float | None:
    """Nearest eligible resistance level swept-and-rejected at bar ``t``, or ``None``.

    A resistance ``R`` qualifies when it is neighbour-locked (``p + k <= t``),
    pierced above by more than ``margin`` (``high_t > R + margin``) and rejected on
    close (``close_t < R``). Among matches the one nearest ``close_t`` (smallest
    ``R``) is returned for determinism.
    """
    best: float | None = None
    for _p, confirm_idx, level in pivot_highs:
        if confirm_idx > t:
            continue
        if high_t > level + margin and close_t < level and (best is None or level < best):
            best = level
    return best


def _make_row(
    *,
    asset: str,
    index: pd.DatetimeIndex,
    t: int,
    n: int,
    open_: np.ndarray,
    direction: str,
    zone_lo: float,
    zone_hi: float,
    stop_ref_price: float,
    atr_at_confirm: float,
    params: dict[str, object],
    phash: str,
    logic_version: int,
) -> dict[str, object]:
    """Build one observation dict for a sweep failure confirmed at bar ``t``."""
    has_next = t + 1 < n
    execution_ts = index[t + 1] if has_next else None
    entry_ref_price = float(open_[t + 1]) if has_next else np.nan
    return {
        "event_id": EVENT_ID,
        "asset": asset,
        "timeframe": _TIMEFRAME,
        "formation_end_ts": index[t],
        "confirmed_ts": index[t],
        "execution_ts": execution_ts,
        "direction": direction,
        "params": params,
        "logic_version": int(logic_version),
        "params_hash": phash,
        "entry_ref_price": entry_ref_price,
        "stop_ref_price": float(stop_ref_price),
        "zone_lo": float(zone_lo),
        "zone_hi": float(zone_hi),
        "quality": np.nan,
        "atr_at_confirm": float(atr_at_confirm) if not np.isnan(atr_at_confirm) else np.nan,
    }
