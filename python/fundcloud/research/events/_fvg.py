"""Three-candle fair-value-gap / imbalance detector (``ev_gap_up`` / ``ev_gap_dn``).

A fair-value gap (FVG) is a three-bar imbalance: the middle bar's range is so
wide that the outer two bars do not overlap, leaving an unfilled price void.
This detector is purely *bar-local* — every input (the ``t-2``, ``t-1`` and
``t`` bars) is closed at bar ``t``, so the event confirms at bar ``t`` with no
pivots and zero future bars. ``confirmed_ts`` is therefore ``index[t]`` (the
leak-free anchor) and ``execution_ts`` / ``entry_ref_price`` read the next bar's
open per the four-timestamp execution contract.

The gap is qualified two ways so noise gaps drop out:

* **Body fraction** — the middle bar's body must fill at least ``body_min`` of
  its own range, i.e. a decisive candle rather than a long-wicked doji.
* **Impulse** — the middle bar's range, measured in ATR units at ``t-1``, must
  reach ``z_imp``, i.e. an actual volatility expansion.

The two geometric branches are distinct events (their ``event_id`` *is* the
detection equation): the up-gap ``low[t] > high[t-2]`` emits ``ev_gap_up`` and
the down-gap ``high[t] < low[t-2]`` emits ``ev_gap_dn``, on separate rows that
share one ``params_hash`` (keyed on the detector base id :data:`BASE_EVENT_ID`).
See ``docs/guides/research/event-registry.md`` for the registry contract.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_fvg"]

#: Event id for the up-gap (bullish-geometry) branch.
EVENT_ID_UP = "ev_gap_up"
#: Event id for the down-gap (bearish-geometry) branch.
EVENT_ID_DN = "ev_gap_dn"
#: Detector base id — used ONLY to key ``params_hash`` so both branches of one
#: call share a single hash. It never appears as an emitted ``event_id``.
BASE_EVENT_ID = "ev_gap_imb_3c"


def detect_fvg(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    body_min: float = 0.6,
    z_imp: float = 1.0,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect three-candle fair-value gaps (imbalances) on one asset's bars.

    For every centre bar ``t >= max(2, atr_n)`` the middle bar ``t-1`` is scored
    by body fraction and ATR-relative impulse, and a gap fires when the outer
    bars (``t-2`` and ``t``) fail to overlap in the gap direction:

    * ``ev_gap_up`` — ``low[t] > high[t-2]`` (an up-gap); the void is
      ``zone_lo = high[t-2]`` to ``zone_hi = low[t]``.
    * ``ev_gap_dn`` — ``high[t] < low[t-2]`` (a down-gap); the void is
      ``zone_lo = high[t]`` to ``zone_hi = low[t-2]``.

    Each qualifying gap also requires ``body_frac >= body_min`` and
    ``impulse >= z_imp``, where ``body_frac = |close[t-1] - open[t-1]| / mid_rng``,
    ``impulse = mid_rng / atr[t-1]`` and ``mid_rng = high[t-1] - low[t-1]``
    (centres with ``mid_rng <= 0`` are skipped).

    The event confirms at bar ``t`` (``formation_end_ts == confirmed_ts ==
    index[t]``); ``execution_ts`` and ``entry_ref_price`` come from the next bar
    (``NaT`` / ``NaN`` when ``t`` is the last bar — the row is still emitted).
    ``quality`` and ``stop_ref_price`` are ``NaN`` (filled downstream);
    ``atr_at_confirm`` is ``atr[t]``.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns and a sorted, unique, tz-aware UTC
        DatetimeIndex.
    asset
        Asset label written into every emitted row.
    body_min
        Minimum fraction of the middle bar's range its body must occupy.
    z_imp
        Minimum middle-bar range in ATR units (impulse threshold).
    atr_n
        Wilder ATR length; also lifts the start index so ``atr[t-1]`` is defined.
    logic_version
        Formation/timestamp-rule version, embedded in ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input
        or no detected gaps yields an empty frame with those columns.
    """
    if bars.empty:
        return build_observations([])

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    index = bars.index
    n_bars = len(bars)

    atr = wilder_atr(high, low, close, atr_n)

    params: dict[str, Any] = {
        "body_min": body_min,
        "z_imp": z_imp,
        "atr_n": atr_n,
    }
    phash = params_hash(BASE_EVENT_ID, params, logic_version)
    timeframe = "1D"

    rows: list[dict[str, Any]] = []
    start = max(2, atr_n)
    for t in range(start, n_bars):
        mid_rng = high[t - 1] - low[t - 1]
        if mid_rng <= 0:
            continue
        body_frac = abs(close[t - 1] - open_[t - 1]) / mid_rng
        impulse = mid_rng / atr[t - 1]
        if body_frac < body_min or impulse < z_imp:
            continue

        if low[t] > high[t - 2]:
            event_id = EVENT_ID_UP
            zone_lo = float(high[t - 2])
            zone_hi = float(low[t])
        elif high[t] < low[t - 2]:
            event_id = EVENT_ID_DN
            zone_lo = float(high[t])
            zone_hi = float(low[t - 2])
        else:
            continue

        confirmed_ts = index[t]
        has_next = t + 1 < n_bars
        execution_ts = index[t + 1] if has_next else None
        entry_ref_price = float(open_[t + 1]) if has_next else float("nan")

        rows.append(
            {
                "event_id": event_id,
                "asset": asset,
                "timeframe": timeframe,
                "formation_end_ts": confirmed_ts,
                "confirmed_ts": confirmed_ts,
                "execution_ts": execution_ts,
                "params": params,
                "logic_version": logic_version,
                "params_hash": phash,
                "entry_ref_price": entry_ref_price,
                "stop_ref_price": float("nan"),
                "zone_lo": zone_lo,
                "zone_hi": zone_hi,
                "quality": float("nan"),
                "atr_at_confirm": float(atr[t]),
            }
        )

    return build_observations(rows)
