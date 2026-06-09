"""N-day Donchian channel breakout detector (``ev_donchian_up`` / ``ev_donchian_dn``).

A Donchian breakout is a close that escapes the prior ``N``-bar price channel: the
upper boundary is the trailing ``N``-bar high, the lower boundary the trailing
``N``-bar low. Both boundaries are measured over a **trailing** half-open window
``[t-N, t)`` that *excludes* the current bar, so the channel is fully closed at
bar ``t`` and the event confirms at bar ``t`` with zero future bars.

This is the key causal contrast with the sweep detector: a trailing rolling
extreme is **not** a centred pivot. A pivot of order ``k`` at index ``p`` peeks
``k`` bars into the future and only becomes knowable at ``p + k`` (neighbour-lock).
A trailing max/min over ``[t-N, t)`` reads only bars strictly before ``t`` — there
is no forward read, so there is no neighbour-lock to respect. ``confirmed_ts`` is
simply ``index[t]``.

For each bar ``t`` (with ``t >= max(N, atr_n)`` so both the channel window and the
prior ATR exist):

* ``hi_prior = max(high[t-N : t])`` and ``lo_prior = min(low[t-N : t])`` over the
  trailing window that excludes ``t``,
* up break (``ev_donchian_up``) when ``close[t] > hi_prior + buf * atr[t-1]``; the
  structural stop sits at the opposite channel boundary
  (``stop_ref_price = lo_prior``),
* down break (``ev_donchian_dn``) when ``close[t] < lo_prior - buf * atr[t-1]``; the
  stop sits at the opposite boundary (``stop_ref_price = hi_prior``).

The breakout buffer uses the *prior* bar's ATR (``atr[t-1]``, known at ``t``),
never ``atr[t]``, inside the firing predicate. The two geometric branches are
distinct events (their ``event_id`` *is* the detection equation), emitted as
separate rows that share one ``params_hash`` (keyed on the detector base id
:data:`BASE_EVENT_ID`). The event carries no zone (``zone_lo``/``zone_hi`` are
``NaN``); ``atr_at_confirm`` is ``atr[t]``. ``confirmed_ts == formation_end_ts ==
index[t]``; ``execution_ts`` / ``entry_ref_price`` read the next bar's open per the
four-timestamp execution contract (``NaT`` / ``NaN`` when ``t`` is the last bar —
the row is still emitted). See ``docs/guides/research/event-registry.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_donchian"]

#: Event id for the up-break (bullish-geometry) branch.
EVENT_ID_UP = "ev_donchian_up"
#: Event id for the down-break (bearish-geometry) branch.
EVENT_ID_DN = "ev_donchian_dn"
#: Detector base id — used ONLY to key ``params_hash`` so both branches of one
#: call share a single hash. It never appears as an emitted ``event_id``.
BASE_EVENT_ID = "ev_donchian_break"
_TIMEFRAME = "1D"


def detect_donchian(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    N: int = 20,
    buf: float = 0.0,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect N-day Donchian channel breakouts on one asset's OHLCV bars.

    For every bar ``t >= max(N, atr_n)`` with a finite prior ATR, the trailing
    channel is measured over the half-open window ``[t-N, t)`` (excluding ``t``):

    * ``ev_donchian_up`` — ``close[t] > max(high[t-N : t]) + buf * atr[t-1]``; the
      stop sits at the opposite boundary ``stop_ref_price = min(low[t-N : t])``.
    * ``ev_donchian_dn`` — ``close[t] < min(low[t-N : t]) - buf * atr[t-1]``; the
      stop sits at the opposite boundary ``stop_ref_price = max(high[t-N : t])``.

    The trailing window reads only bars strictly before ``t``, so a breakout is a
    pure bar-local event (no centred pivot, no neighbour-lock, no forward read).
    The buffer scales the breakout margin in ATR units at ``t-1`` (known at ``t``).

    The event confirms at bar ``t`` (``formation_end_ts == confirmed_ts ==
    index[t]``); ``execution_ts`` and ``entry_ref_price`` come from the next bar
    (``NaT`` / ``NaN`` when ``t`` is the last bar — the row is still emitted).
    ``zone_lo`` / ``zone_hi`` / ``quality`` are ``NaN``; ``atr_at_confirm`` is
    ``atr[t]``.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        ``DatetimeIndex``.
    asset
        Asset label written into every emitted row.
    N
        Donchian channel length: the trailing window ``[t-N, t)`` whose high and
        low form the upper and lower channel boundaries.
    buf
        Breakout buffer in ATR units: the close must clear the boundary by more
        than ``buf * atr[t-1]``. ``0.0`` requires a strict break of the raw level.
    atr_n
        Wilder ATR length; also lifts the start index so ``atr[t-1]`` is defined.
        A bar at ``t`` is evaluated only once ``atr[t-1]`` is finite.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input —
        or no breakouts — yields an empty frame with those columns.
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

    params = {"N": int(N), "buf": float(buf), "atr_n": int(atr_n)}
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    start = max(N, atr_n)
    for t in range(start, n_bars):
        atr_prev = atr[t - 1]
        if not np.isfinite(atr_prev):
            continue

        hi_prior = float(np.max(high[t - N : t]))
        lo_prior = float(np.min(low[t - N : t]))
        margin = buf * atr_prev

        if close[t] > hi_prior + margin:
            rows.append(
                _row(
                    asset=asset,
                    index=index,
                    t=t,
                    n_bars=n_bars,
                    open_=open_,
                    event_id=EVENT_ID_UP,
                    stop_ref_price=lo_prior,
                    atr_at_confirm=atr[t],
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                )
            )
        elif close[t] < lo_prior - margin:
            rows.append(
                _row(
                    asset=asset,
                    index=index,
                    t=t,
                    n_bars=n_bars,
                    open_=open_,
                    event_id=EVENT_ID_DN,
                    stop_ref_price=hi_prior,
                    atr_at_confirm=atr[t],
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
    stop_ref_price: float,
    atr_at_confirm: float,
    params: dict[str, object],
    phash: str,
    logic_version: int,
) -> dict[str, object]:
    """Build one observation dict for a breakout confirmed at bar ``t``."""
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
        "zone_lo": float("nan"),
        "zone_hi": float("nan"),
        "quality": float("nan"),
        "atr_at_confirm": float(atr_at_confirm) if np.isfinite(atr_at_confirm) else float("nan"),
    }
