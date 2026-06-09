"""Order-block detector (``ev_ob_up`` / ``ev_ob_dn``).

An *order block* is the last opposing candle immediately before a decisive
displacement: the final bearish candle before a bullish impulse (a demand block),
or the final bullish candle before a bearish impulse (a supply block). The block's
body (or full range) marks the zone price is expected to revisit.

The detector is **impulse-bar-driven**: the loop iterates over the impulse-close
bar ``c`` and looks *backward* for the most-recent opposing candle ``j`` in
``[c - m, c - 1]``. Every input — the impulse bar ``c``, the opposing candle
``j`` and the clearance window ``j - r .. j`` — is closed at or before bar ``c``,
so the event confirms *at* bar ``c`` with zero future reads and no neighbour-lock.
Dating the event at the *impulse* bar (not the opposing candle) is what keeps it
online-causal: an opposing-candle-driven loop would scan forward from ``j`` to
confirm an event dated at ``j``, which leaks future bars and breaks prefix
invariance (``docs/guides/research/event-registry.md``).

For each impulse-close bar ``c`` (with ``c >= atr_n`` so ``atr[c-1]`` exists):

* **Bullish (``ev_ob_up``)** — a bullish displacement
  ``(close[c] - open[c]) > z_body * atr[c-1]``; ``j`` is the most-recent *bearish*
  candle (``close[j] < open[j]``) in the look-back window; clearance
  ``close[c] > max(high[j-r .. j])`` (window clamped to ``j - r >= 0``).
* **Bearish (``ev_ob_dn``)** — a bearish displacement
  ``(open[c] - close[c]) > z_body * atr[c-1]``; ``j`` is the most-recent *bullish*
  candle (``close[j] > open[j]``); clearance ``close[c] < min(low[j-r .. j])``.

The zone is candle ``j``'s body (``[min(open[j], close[j]), max(open[j],
close[j])]``) unless ``wick`` is set, in which case it is ``j``'s full range
(``[low[j], high[j]]``). ``stop_ref_price`` is ``low[j]`` for a bullish block and
``high[j]`` for a bearish block. The two geometric branches are distinct events
emitted as separate rows that share one ``params_hash`` (keyed on the detector
base id :data:`BASE_EVENT_ID`). ``confirmed_ts == formation_end_ts == index[c]``;
``execution_ts`` / ``entry_ref_price`` read the next bar's open per the
four-timestamp execution contract (``NaT`` / ``NaN`` on the final bar — the row
is still emitted). ``atr_at_confirm`` is ``atr[c]``; ``quality`` is ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_order_block"]

#: Event id for the bullish (demand-block) branch.
EVENT_ID_UP = "ev_ob_up"
#: Event id for the bearish (supply-block) branch.
EVENT_ID_DN = "ev_ob_dn"
#: Detector base id — used ONLY to key ``params_hash`` so both branches of one
#: call share a single hash. It never appears as an emitted ``event_id``.
BASE_EVENT_ID = "ev_ob_impulse_last_opp"
_TIMEFRAME = "1D"


def detect_order_block(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    m: int = 5,
    r: int = 1,
    z_body: float = 1.0,
    atr_n: int = 14,
    wick: bool = False,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect order blocks (last opposing candle before a displacement).

    The loop is impulse-bar-driven: for each impulse-close bar ``c`` it confirms
    at most one ``ev_ob_up`` and one ``ev_ob_dn``, looking backward for the
    nearest opposing candle. Everything read is ``<= c`` so the event is online-
    causal with no neighbour-lock.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        DatetimeIndex.
    asset
        Asset label written into every emitted row.
    m
        Look-back window length: the opposing candle ``j`` is searched in
        ``[c - m, c - 1]``.
    r
        Clearance look-back: the impulse must clear the extreme of the window
        ``[j - r, j]`` (clamped to ``j - r >= 0``).
    z_body
        Displacement threshold in ATR units: the impulse body must exceed
        ``z_body * atr[c-1]``.
    atr_n
        Wilder ATR length. A bar at ``c`` is only evaluated once ``atr[c-1]`` is
        defined, i.e. ``c >= atr_n`` with a finite prior ATR.
    wick
        When ``True`` the zone spans the opposing candle's full range
        (``[low[j], high[j]]``); otherwise it spans its body.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input
        — or no detected blocks — yields an empty frame with those columns.
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

    params = {
        "m": int(m),
        "r": int(r),
        "z_body": float(z_body),
        "atr_n": int(atr_n),
        "wick": bool(wick),
    }
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for c in range(atr_n, n_bars):
        atr_prev = atr[c - 1]
        if not np.isfinite(atr_prev):
            continue
        body_thresh = z_body * atr_prev
        lookback_start = max(0, c - m)

        if (close[c] - open_[c]) > body_thresh:
            j = _last_opposing(open_, close, lookback_start, c, bearish=True)
            if j is not None:
                clear_start = max(0, j - r)
                if close[c] > float(np.max(high[clear_start : j + 1])):
                    rows.append(
                        _make_row(
                            asset=asset,
                            index=index,
                            c=c,
                            n_bars=n_bars,
                            open_=open_,
                            event_id=EVENT_ID_UP,
                            zone=_zone(open_, close, low, high, j, wick=wick),
                            stop_ref_price=float(low[j]),
                            atr_at_confirm=atr[c],
                            params=params,
                            phash=phash,
                            logic_version=logic_version,
                        )
                    )

        if (open_[c] - close[c]) > body_thresh:
            j = _last_opposing(open_, close, lookback_start, c, bearish=False)
            if j is not None:
                clear_start = max(0, j - r)
                if close[c] < float(np.min(low[clear_start : j + 1])):
                    rows.append(
                        _make_row(
                            asset=asset,
                            index=index,
                            c=c,
                            n_bars=n_bars,
                            open_=open_,
                            event_id=EVENT_ID_DN,
                            zone=_zone(open_, close, low, high, j, wick=wick),
                            stop_ref_price=float(high[j]),
                            atr_at_confirm=atr[c],
                            params=params,
                            phash=phash,
                            logic_version=logic_version,
                        )
                    )

    return build_observations(rows)


def _last_opposing(
    open_: np.ndarray,
    close: np.ndarray,
    start: int,
    c: int,
    *,
    bearish: bool,
) -> int | None:
    """Index of the most-recent opposing candle in ``[start, c)``, or ``None``.

    A bearish candle satisfies ``close < open``; a bullish candle ``close > open``.
    The scan runs from ``c - 1`` down to ``start`` so the first match is the
    nearest (largest ``j``) — unique by construction.
    """
    for j in range(c - 1, start - 1, -1):
        if bearish:
            if close[j] < open_[j]:
                return j
        elif close[j] > open_[j]:
            return j
    return None


def _zone(
    open_: np.ndarray,
    close: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    j: int,
    *,
    wick: bool,
) -> tuple[float, float]:
    """``(zone_lo, zone_hi)`` for opposing candle ``j`` — full range if ``wick``."""
    if wick:
        return float(low[j]), float(high[j])
    return float(min(open_[j], close[j])), float(max(open_[j], close[j]))


def _make_row(
    *,
    asset: str,
    index: pd.DatetimeIndex,
    c: int,
    n_bars: int,
    open_: np.ndarray,
    event_id: str,
    zone: tuple[float, float],
    stop_ref_price: float,
    atr_at_confirm: float,
    params: dict[str, object],
    phash: str,
    logic_version: int,
) -> dict[str, object]:
    """Build one observation dict for an order block confirmed at bar ``c``."""
    has_next = c + 1 < n_bars
    execution_ts = index[c + 1] if has_next else None
    entry_ref_price = float(open_[c + 1]) if has_next else float("nan")
    zone_lo, zone_hi = zone
    return {
        "event_id": event_id,
        "asset": asset,
        "timeframe": _TIMEFRAME,
        "formation_end_ts": index[c],
        "confirmed_ts": index[c],
        "execution_ts": execution_ts,
        "params": params,
        "logic_version": int(logic_version),
        "params_hash": phash,
        "entry_ref_price": entry_ref_price,
        "stop_ref_price": stop_ref_price,
        "zone_lo": zone_lo,
        "zone_hi": zone_hi,
        "quality": float("nan"),
        "atr_at_confirm": float(atr_at_confirm) if np.isfinite(atr_at_confirm) else float("nan"),
    }
