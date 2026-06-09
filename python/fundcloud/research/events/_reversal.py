"""Bar-local outside-bar key-reversal detector (``ev_keyrev_up`` / ``ev_keyrev_dn``).

A *key reversal* is a single outside bar that engulfs the prior bar's range and
then closes decisively beyond it: a bullish bar takes out yesterday's low yet
closes above yesterday's high near its own top, a bearish bar takes out
yesterday's high yet closes below yesterday's low near its own bottom. Both
conditions read only bars ``t-1`` and ``t`` — no pivots, no future bars — so the
event confirms *at* bar ``t``. ``confirmed_ts == formation_end_ts == index[t]``
(the leak-free anchor) and ``execution_ts`` / ``entry_ref_price`` read the next
bar's open per the four-timestamp execution contract
(``docs/guides/research/event-registry.md``).

For each bar ``t`` (with ``t >= 1`` so bar ``t-1`` exists, and finite ``atr[t]``
so a volatility unit can be recorded):

* outside bar — ``high[t] > high[t-1]`` and ``low[t] < low[t-1]``,
* ``rng = high[t] - low[t]``; bars with ``rng <= 0`` are skipped (no range to
  locate the close within),
* ``clv = (close[t] - low[t]) / rng`` — the close's location within the bar,
* up (``ev_keyrev_up``) when outside and ``close[t] > high[t-1]`` and
  ``clv >= clv_min``; the structural stop sits under the wick
  (``stop_ref_price = low[t]``),
* down (``ev_keyrev_dn``) when outside and ``close[t] < low[t-1]`` and
  ``clv <= 1 - clv_min``; the stop sits above the wick
  (``stop_ref_price = high[t]``).

The two geometric branches are mutually exclusive distinct events (their
``event_id`` *is* the detection equation), emitted as separate rows that share
one ``params_hash`` (keyed on the detector base id :data:`BASE_EVENT_ID`). The
event carries no zone (``zone_lo`` / ``zone_hi`` are ``NaN``); ``quality`` is
``NaN`` (filled downstream) and ``atr_at_confirm`` is ``atr[t]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_key_reversal"]

#: Event id for the up (bullish-geometry) branch.
EVENT_ID_UP = "ev_keyrev_up"
#: Event id for the down (bearish-geometry) branch.
EVENT_ID_DN = "ev_keyrev_dn"
#: Detector base id — used ONLY to key ``params_hash`` so both branches of one
#: call share a single hash. It never appears as an emitted ``event_id``.
BASE_EVENT_ID = "ev_outside_reversal"
_TIMEFRAME = "1D"


def detect_key_reversal(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    clv_min: float = 0.7,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect bar-local outside-bar key reversals in single-asset OHLCV.

    For every bar ``t >= 1`` an outside bar (``high[t] > high[t-1]`` and
    ``low[t] < low[t-1]``) with a finite ``atr[t]`` is scored by its
    close-location value ``clv = (close[t] - low[t]) / (high[t] - low[t])``
    (bars with non-positive range are skipped). A bullish reversal
    (``ev_keyrev_up``) fires when the bar closes above the prior high
    (``close[t] > high[t-1]``) with ``clv >= clv_min``; a bearish reversal
    (``ev_keyrev_dn``) fires when it closes below the prior low
    (``close[t] < low[t-1]``) with ``clv <= 1 - clv_min``. The two branches are
    mutually exclusive.

    The event confirms at bar ``t`` (``formation_end_ts == confirmed_ts ==
    index[t]``); ``execution_ts`` and ``entry_ref_price`` come from the next bar
    (``NaT`` / ``NaN`` when ``t`` is the last bar — the row is still emitted).
    ``stop_ref_price`` is ``low[t]`` for the up branch and ``high[t]`` for the
    down branch; ``zone_lo`` / ``zone_hi`` / ``quality`` are ``NaN`` and
    ``atr_at_confirm`` is ``atr[t]``.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        DatetimeIndex.
    asset
        Asset label written to every emitted row.
    clv_min
        Minimum close-location value for a bullish reversal (a bearish reversal
        requires ``clv <= 1 - clv_min``). ``0.7`` keeps closes in the top/bottom
        30% of the bar.
    atr_n
        Wilder ATR length. A bar only fires when ``atr[t]`` is finite, so warmup
        bars never confirm; the recorded vol unit ``atr_at_confirm`` is
        ``atr[t]``.
    logic_version
        Formation/timestamp-rule version stamped into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input
        or no detections yields an empty frame with those columns.
    """
    if bars.empty:
        return build_observations([])

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    index = bars.index
    n_bars = len(close)

    atr = wilder_atr(high, low, close, atr_n)

    params = {"clv_min": float(clv_min), "atr_n": int(atr_n)}
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for t in range(1, n_bars):
        atr_t = atr[t]
        if not np.isfinite(atr_t):
            continue
        outside = high[t] > high[t - 1] and low[t] < low[t - 1]
        if not outside:
            continue
        rng = high[t] - low[t]
        if rng <= 0:
            continue
        clv = (close[t] - low[t]) / rng

        is_bull = close[t] > high[t - 1] and clv >= clv_min
        is_bear = close[t] < low[t - 1] and clv <= 1.0 - clv_min
        if not (is_bull or is_bear):
            continue

        has_next = t + 1 < n_bars
        execution_ts = index[t + 1] if has_next else None
        entry_ref_price = float(open_[t + 1]) if has_next else float("nan")

        if is_bull:
            rows.append(
                _row(
                    asset=asset,
                    confirmed_ts=index[t],
                    execution_ts=execution_ts,
                    event_id=EVENT_ID_UP,
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                    entry_ref_price=entry_ref_price,
                    stop_ref_price=float(low[t]),
                    atr_at_confirm=float(atr_t),
                )
            )
        else:
            rows.append(
                _row(
                    asset=asset,
                    confirmed_ts=index[t],
                    execution_ts=execution_ts,
                    event_id=EVENT_ID_DN,
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                    entry_ref_price=entry_ref_price,
                    stop_ref_price=float(high[t]),
                    atr_at_confirm=float(atr_t),
                )
            )

    return build_observations(rows)


def _row(
    *,
    asset: str,
    confirmed_ts: pd.Timestamp,
    execution_ts: pd.Timestamp | None,
    event_id: str,
    params: dict[str, object],
    phash: str,
    logic_version: int,
    entry_ref_price: float,
    stop_ref_price: float,
    atr_at_confirm: float,
) -> dict[str, object]:
    """Assemble one observation dict for :func:`build_observations`."""
    return {
        "event_id": event_id,
        "asset": asset,
        "timeframe": _TIMEFRAME,
        "formation_end_ts": confirmed_ts,
        "confirmed_ts": confirmed_ts,
        "execution_ts": execution_ts,
        "params": params,
        "logic_version": int(logic_version),
        "params_hash": phash,
        "entry_ref_price": entry_ref_price,
        "stop_ref_price": stop_ref_price,
        "zone_lo": float("nan"),
        "zone_hi": float("nan"),
        "quality": float("nan"),
        "atr_at_confirm": atr_at_confirm,
    }
