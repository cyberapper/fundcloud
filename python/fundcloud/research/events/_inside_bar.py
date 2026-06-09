"""Bar-local inside-bar / compression detector (``ev_inside_bar``).

An inside bar is a contraction: bar ``t``'s range is wholly contained within the
*mother bar* ``t-1`` (``high[t] <= high[t-1]`` and ``low[t] >= low[t-1]``). It is
a volatility-coiling event, not a directional one — both extremes of bar ``t``
sit inside the prior range, so the geometry names no side. Every input is closed
at bar ``t`` (the mother bar ``t-1`` and the inside bar ``t``), so the event is
purely bar-local: it confirms *at* bar ``t`` with no pivots and zero future bars.
``confirmed_ts == formation_end_ts == index[t]`` (the leak-free anchor); there is
no neighbour-lock.

A contraction has no geometric side, so this detector is **neutral**: it emits a
single :data:`EVENT_ID` with no ``_up`` / ``_dn`` suffix (and :data:`BASE_EVENT_ID`
equals it). The evidence layer (``side="auto"``) decides direction from the
realised forward path, and the legacy ``to_events_frame`` -> ``feature_quality``
bridge intentionally drops the event: that bridge keys ``long_entry`` /
``short_entry`` off the ``_up`` / ``_dn`` suffix, so a suffix-less id lands ``NaN``
in both entry columns and is filtered out by ``evaluate``. See
``docs/guides/research/event-registry.md`` for the registry contract.

The mother-bar range is recorded as the zone (``zone_lo = low[t-1]``,
``zone_hi = high[t-1]``); ``stop_ref_price`` and ``quality`` are ``NaN`` (filled
downstream). ``atr_at_confirm`` is ``atr[t]``. ``execution_ts`` /
``entry_ref_price`` read the next bar's open per the four-timestamp execution
contract (``NaT`` / ``NaN`` when ``t`` is the last bar — the row is still emitted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_inside_bar"]

#: The neutral event id (no geometric branch).
EVENT_ID = "ev_inside_bar"
#: Detector base id — keys ``params_hash``. For a neutral detector it equals the
#: single emitted :data:`EVENT_ID`.
BASE_EVENT_ID = "ev_inside_bar"
_TIMEFRAME = "1D"


def detect_inside_bar(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    strict: bool = False,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect bar-local inside bars (compression) in single-asset OHLCV.

    For every bar ``t >= 1`` with a finite ``atr[t]`` (so a volatility unit can be
    recorded), bar ``t`` is an inside bar when its range is contained within the
    mother bar ``t-1``:

    * non-strict (default) — ``high[t] <= high[t-1]`` and ``low[t] >= low[t-1]``,
    * strict — ``high[t] < high[t-1]`` and ``low[t] > low[t-1]`` (a bar that merely
      equals the mother bar's high or low does not fire).

    The event is **neutral** (no geometric side): one :data:`EVENT_ID` row is
    emitted, with the mother-bar range as the zone (``zone_lo = low[t-1]``,
    ``zone_hi = high[t-1]``) and ``stop_ref_price`` / ``quality`` ``NaN``. It
    confirms at bar ``t`` (``formation_end_ts == confirmed_ts == index[t]``);
    ``execution_ts`` and ``entry_ref_price`` read the next bar's open (``NaT`` /
    ``NaN`` when ``t`` is the last bar — the row is still emitted).
    ``atr_at_confirm`` is ``atr[t]``.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        ``DatetimeIndex``.
    asset
        Asset label written to every emitted row.
    strict
        When ``True`` require strict containment (``<`` / ``>``) so a bar equal to
        the mother bar's high or low does not fire; when ``False`` (default) allow
        equality (``<=`` / ``>=``).
    atr_n
        Wilder ATR length. A bar with a NaN ``atr[t]`` (warmup) cannot fire, since
        no ``atr_at_confirm`` could be recorded.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input
        — or no inside bars — yields an empty frame with those columns.
    """
    if bars.empty:
        return build_observations([])

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    index = bars.index
    n_bars = len(index)

    atr = wilder_atr(high, low, close, atr_n)

    params = {"strict": bool(strict), "atr_n": int(atr_n)}
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for t in range(1, n_bars):
        atr_t = atr[t]
        if not np.isfinite(atr_t):
            continue

        if strict:
            is_inside = high[t] < high[t - 1] and low[t] > low[t - 1]
        else:
            is_inside = high[t] <= high[t - 1] and low[t] >= low[t - 1]
        if not is_inside:
            continue

        has_next = t + 1 < n_bars
        execution_ts = index[t + 1] if has_next else None
        entry_ref_price = float(open_[t + 1]) if has_next else float("nan")

        rows.append(
            {
                "event_id": EVENT_ID,
                "asset": asset,
                "timeframe": _TIMEFRAME,
                "formation_end_ts": index[t],
                "confirmed_ts": index[t],
                "execution_ts": execution_ts,
                "params": params,
                "logic_version": int(logic_version),
                "params_hash": phash,
                "entry_ref_price": entry_ref_price,
                "stop_ref_price": float("nan"),
                "zone_lo": float(low[t - 1]),
                "zone_hi": float(high[t - 1]),
                "quality": float("nan"),
                "atr_at_confirm": float(atr_t),
            }
        )

    return build_observations(rows)
