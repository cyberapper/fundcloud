"""Overnight opening-gap continuation detector (``ev_opengap_up`` / ``ev_opengap_dn``).

An opening gap is a discontinuity between yesterday's close and today's open
large enough to clear a volatility-scaled threshold, where today's bar then
*continues* in the gap's direction (closes through its own open). The void left
between ``close[t-1]`` and ``open[t]`` is the imbalance the event marks.

This detector is purely *bar-local*: the gap test reads ``open[t]`` against
``close[t-1]`` and the continuation test reads ``close[t]`` against ``open[t]``,
all of which are closed at bar ``t``. The buffer ATR uses ``atr[t-1]`` (known at
``t``), never ``atr[t]``, so the event confirms *at* bar ``t`` with zero future
bars and no neighbour-lock. ``confirmed_ts == formation_end_ts == index[t]``;
``execution_ts`` / ``entry_ref_price`` read the next bar's open per the
four-timestamp execution contract (``docs/guides/research/event-registry.md``).

For each bar ``t`` (with ``t >= 1`` so ``close[t-1]`` exists, and
``np.isfinite(atr[t-1])`` so the buffer is defined):

* up (``ev_opengap_up``) when ``open[t] > close[t-1] + k * atr[t-1]`` and
  ``close[t] > open[t]`` (gap up that holds and extends). The gap void runs
  ``zone_lo = close[t-1]`` to ``zone_hi = open[t]``; the structural stop sits
  under the bar (``stop_ref_price = low[t]``).
* down (``ev_opengap_dn``) when ``open[t] < close[t-1] - k * atr[t-1]`` and
  ``close[t] < open[t]`` (gap down that holds and extends). The gap void runs
  ``zone_lo = open[t]`` to ``zone_hi = close[t-1]``; the stop sits above the bar
  (``stop_ref_price = high[t]``).

The two geometric branches are distinct events (their ``event_id`` *is* the
detection equation), emitted as separate rows that share one ``params_hash``
(keyed on the detector base id :data:`BASE_EVENT_ID`). ``atr_at_confirm`` is
``atr[t]``; ``quality`` is ``NaN`` (filled downstream).

Gaps are split- and dividend-sensitive: a raw split or large dividend manifests
as an enormous synthetic open-to-close discontinuity. This detector therefore
**must** run on the adjusted / cleaned price panel, never on raw prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_opening_gap"]

#: Event id for the gap-up (bullish-geometry) branch.
EVENT_ID_UP = "ev_opengap_up"
#: Event id for the gap-down (bearish-geometry) branch.
EVENT_ID_DN = "ev_opengap_dn"
#: Detector base id — used ONLY to key ``params_hash`` so both branches of one
#: call share a single hash. It never appears as an emitted ``event_id``.
BASE_EVENT_ID = "ev_opening_gap"
_TIMEFRAME = "1D"


def detect_opening_gap(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    k: float = 0.5,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect overnight opening-gap continuations on one asset's OHLCV bars.

    For every bar ``t >= 1`` with a defined prior ATR, the open is compared to
    the previous close across a buffer of ``k * atr[t-1]``, and the bar must then
    close *through* its own open in the gap's direction:

    * ``ev_opengap_up`` — ``open[t] > close[t-1] + k * atr[t-1]`` and
      ``close[t] > open[t]``. The gap void is ``zone_lo = close[t-1]`` to
      ``zone_hi = open[t]``; ``stop_ref_price = low[t]``.
    * ``ev_opengap_dn`` — ``open[t] < close[t-1] - k * atr[t-1]`` and
      ``close[t] < open[t]``. The gap void is ``zone_lo = open[t]`` to
      ``zone_hi = close[t-1]``; ``stop_ref_price = high[t]``.

    The event confirms at bar ``t`` (``formation_end_ts == confirmed_ts ==
    index[t]``); ``execution_ts`` and ``entry_ref_price`` come from the next bar
    (``NaT`` / ``NaN`` when ``t`` is the last bar — the row is still emitted).
    ``quality`` is ``NaN`` (filled downstream); ``atr_at_confirm`` is ``atr[t]``.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        ``DatetimeIndex``. Prices MUST be split/dividend-adjusted (cleaned panel)
        — a raw split shows up as a giant synthetic gap.
    asset
        Asset label written into every emitted row.
    k
        Gap buffer in ATR units: the open must clear the prior close by more than
        ``k * atr[t-1]`` for the bar to count as gapped.
    atr_n
        Wilder ATR length. The buffer at bar ``t`` uses ``atr[t-1]`` (known at
        ``t``); a bar with NaN prior ATR (warmup) can never fire.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input
        — or no detected gaps — yields an empty frame with those columns.
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

    params = {"k": float(k), "atr_n": int(atr_n)}
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for t in range(1, n_bars):
        atr_prev = atr[t - 1]
        if not np.isfinite(atr_prev):
            continue
        buffer = k * atr_prev
        prev_close = close[t - 1]
        open_t = open_[t]
        close_t = close[t]

        is_up = open_t > prev_close + buffer and close_t > open_t
        is_dn = open_t < prev_close - buffer and close_t < open_t
        if not (is_up or is_dn):
            continue

        atr_at_confirm = float(atr[t]) if np.isfinite(atr[t]) else float("nan")

        if is_up:
            rows.append(
                _row(
                    asset=asset,
                    index=index,
                    t=t,
                    n_bars=n_bars,
                    open_=open_,
                    event_id=EVENT_ID_UP,
                    zone_lo=prev_close,
                    zone_hi=open_t,
                    stop_ref_price=low[t],
                    atr_at_confirm=atr_at_confirm,
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )
        else:
            rows.append(
                _row(
                    asset=asset,
                    index=index,
                    t=t,
                    n_bars=n_bars,
                    open_=open_,
                    event_id=EVENT_ID_DN,
                    zone_lo=open_t,
                    zone_hi=prev_close,
                    stop_ref_price=high[t],
                    atr_at_confirm=atr_at_confirm,
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )

    return build_observations(rows)


def _row(
    *,
    asset: str,
    index: pd.DatetimeIndex,
    t: int,
    n_bars: int,
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
    """Assemble one observation dict for an opening gap confirmed at bar ``t``."""
    has_next = t + 1 < n_bars
    execution_ts = index[t + 1] if has_next else None
    entry_ref_price = float(open_[t + 1]) if has_next else float("nan")
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
        "quality": float("nan"),
        "atr_at_confirm": atr_at_confirm,
    }
