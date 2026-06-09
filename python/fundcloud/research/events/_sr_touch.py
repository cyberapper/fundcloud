"""Support/resistance touch-and-hold detector (``ev_sr_bounce_up`` / ``ev_sr_bounce_dn``).

A *touch-and-hold* is a bar that reaches an established swing level inside a
volatility-scaled tolerance band but closes back on the level's holding side —
the level absorbs the test rather than breaking. Unlike a sweep failure
(:mod:`fundcloud.research.events._sweep`), the bar need not pierce the level:
brushing the band is enough, so long as the close respects the level.

* ``ev_sr_bounce_up`` (support bounce) — an eligible pivot-low support ``L`` is
  touched from above (``low[t] <= L + eps * atr[t-1]``) yet the bar closes above
  it (``close[t] > L``). The reaction is up; the structural stop sits under the
  wick (``stop_ref_price = low[t]``). The band straddles the level:
  ``zone_lo = L - eps * atr[t-1]``, ``zone_hi = L + eps * atr[t-1]``.
* ``ev_sr_bounce_dn`` (resistance bounce) — an eligible pivot-high resistance
  ``R`` is touched from below (``high[t] >= R - eps * atr[t-1]``) yet the bar
  closes below it (``close[t] < R``). The reaction is down; the stop sits above
  the wick (``stop_ref_price = high[t]``). The band straddles the level:
  ``zone_lo = R - eps * atr[t-1]``, ``zone_hi = R + eps * atr[t-1]``.

The two geometric branches are distinct events (their ``event_id`` *is* the
detection equation), emitted as separate rows that share one ``params_hash``
(keyed on the detector base id :data:`BASE_EVENT_ID`).

Causality (``docs/guides/research/event-registry.md``): levels are
pivot-derived, so each neighbour-locks. A pivot of order ``k`` at index ``p`` is
only knowable at ``p + k``; the level it builds is *eligible* at bar ``t`` only
when ``p + k <= t``. The tolerance band uses the *prior* bar's ATR (``atr[t-1]``),
which is fully knowable at ``t``. ``confirmed_ts`` is therefore bar ``t`` with no
forward read; ``execution_ts`` / ``entry_ref_price`` legitimately reference the
next bar's open (NaT / NaN on the final bar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import confirmed_pivots, wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_sr_touch_bounce"]

#: Event id for the support-bounce (bullish-geometry) branch.
EVENT_ID_UP = "ev_sr_bounce_up"
#: Event id for the resistance-bounce (bearish-geometry) branch.
EVENT_ID_DN = "ev_sr_bounce_dn"
#: Detector base id — used ONLY to key ``params_hash`` so both branches of one
#: call share a single hash. It never appears as an emitted ``event_id``.
BASE_EVENT_ID = "ev_sr_touch_bounce"
_TIMEFRAME = "1D"


def detect_sr_touch_bounce(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    pivot_k: int = 3,
    eps: float = 0.10,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect support/resistance touch-and-hold bounces on one asset's OHLCV bars.

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
        Touch tolerance in ATR units: a level is touched when price reaches within
        ``eps * atr[t-1]`` of it.
    atr_n
        Wilder ATR length. The tolerance at bar ``t`` uses ``atr[t-1]`` (known at
        ``t``); a bar with NaN prior ATR (warmup) can never fire.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. At most one
        row per ``(bar, branch)`` (``ev_sr_bounce_up`` / ``ev_sr_bounce_dn``),
        using the level nearest ``close[t]`` on ties. Empty input — or no
        bounces — yields an empty frame with those columns.
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
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for t in range(n):
        prior_atr = atr[t - 1] if t >= 1 else np.nan
        if np.isnan(prior_atr):
            continue
        margin = eps * prior_atr

        bull = _nearest_support_touch(pivot_lows, t, low[t], close[t], margin)
        if bull is not None:
            level = bull
            rows.append(
                _make_row(
                    asset=asset,
                    index=index,
                    t=t,
                    n=n,
                    open_=open_,
                    event_id=EVENT_ID_UP,
                    zone_lo=level - margin,
                    zone_hi=level + margin,
                    stop_ref_price=low[t],
                    atr_at_confirm=atr[t],
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )

        bear = _nearest_resistance_touch(pivot_highs, t, high[t], close[t], margin)
        if bear is not None:
            level = bear
            rows.append(
                _make_row(
                    asset=asset,
                    index=index,
                    t=t,
                    n=n,
                    open_=open_,
                    event_id=EVENT_ID_DN,
                    zone_lo=level - margin,
                    zone_hi=level + margin,
                    stop_ref_price=high[t],
                    atr_at_confirm=atr[t],
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )

    return build_observations(rows)


def _nearest_support_touch(
    pivot_lows: list[tuple[int, int, float]],
    t: int,
    low_t: float,
    close_t: float,
    margin: float,
) -> float | None:
    """Nearest eligible support level touched-and-held at bar ``t``, or ``None``.

    A support ``L`` qualifies when it is neighbour-locked (``p + k <= t``), reached
    within tolerance (``low_t <= L + margin``) and held on close (``close_t > L``).
    Among matches the one nearest ``close_t`` (largest ``L``) is returned for
    determinism.
    """
    best: float | None = None
    for _p, confirm_idx, level in pivot_lows:
        if confirm_idx > t:
            continue
        if low_t <= level + margin and close_t > level and (best is None or level > best):
            best = level
    return best


def _nearest_resistance_touch(
    pivot_highs: list[tuple[int, int, float]],
    t: int,
    high_t: float,
    close_t: float,
    margin: float,
) -> float | None:
    """Nearest eligible resistance level touched-and-held at bar ``t``, or ``None``.

    A resistance ``R`` qualifies when it is neighbour-locked (``p + k <= t``),
    reached within tolerance (``high_t >= R - margin``) and held on close
    (``close_t < R``). Among matches the one nearest ``close_t`` (smallest ``R``)
    is returned for determinism.
    """
    best: float | None = None
    for _p, confirm_idx, level in pivot_highs:
        if confirm_idx > t:
            continue
        if high_t >= level - margin and close_t < level and (best is None or level < best):
            best = level
    return best


def _make_row(
    *,
    asset: str,
    index: pd.DatetimeIndex,
    t: int,
    n: int,
    open_: np.ndarray,
    event_id: str,
    zone_lo: float,
    zone_hi: float,
    stop_ref_price: float,
    atr_at_confirm: float,
    params: dict[str, object],
    phash: str,
    logic_version: int,
) -> dict[str, object]:
    """Build one observation dict for a touch-and-hold bounce confirmed at bar ``t``."""
    has_next = t + 1 < n
    execution_ts = index[t + 1] if has_next else None
    entry_ref_price = float(open_[t + 1]) if has_next else np.nan
    return {
        "event_id": event_id,
        "asset": asset,
        "timeframe": _TIMEFRAME,
        "formation_end_ts": index[t],
        "confirmed_ts": index[t],
        "execution_ts": execution_ts,
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
