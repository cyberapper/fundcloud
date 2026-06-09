"""Narrowest-range contraction detector (``ev_nr_squeeze``) — NRn squeeze.

An NRn bar is the *narrowest range* of the last ``n`` bars: a volatility
contraction where the current bar's high-to-low span is the smallest in the
trailing window (the classic NR7 / NR4 squeeze, generalised to ``n``). This is a
purely *bar-local* detector reading a backward window only — bar ``t`` and the
``n - 1`` bars before it — so the event confirms *at* bar ``t`` with zero future
bars and no neighbour-lock (the window never reaches forward). ``confirmed_ts``
is therefore ``index[t]`` (the leak-free anchor) and ``execution_ts`` /
``entry_ref_price`` read the next bar's open per the four-timestamp execution
contract.

For each bar ``t`` (with ``t >= n - 1`` so the trailing window is complete and
``t >= 1``):

* ``rng = high - low`` (elementwise),
* fire iff ``rng[t] <= min(rng[t - n + 1 .. t - 1])`` — narrowest vs the prior
  ``n - 1`` bars (the ``<=`` is inclusive so a tie with a prior narrow bar still
  fires).

A contraction carries no directional bias and no price geometry, so this is a
*neutral* single-id detector — the same convention as ``inside_bar``: one
:data:`EVENT_ID` with no ``_up`` / ``_dn`` suffix, and :data:`BASE_EVENT_ID`
equals it. ``zone_lo`` / ``zone_hi`` / ``stop_ref_price`` / ``quality`` are
``NaN``; ``atr_at_confirm`` is ``atr[t]`` (the recorded volatility unit, required
finite for a bar to fire). See ``docs/guides/research/event-registry.md`` for the
registry contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = ["detect_nr_squeeze"]

#: Event id for the neutral narrowest-range contraction (no geometric branch).
EVENT_ID = "ev_nr_squeeze"
#: Detector base id used to key ``params_hash``. For a neutral detector it equals
#: :data:`EVENT_ID` (there is no ``_up`` / ``_dn`` split to share a hash across).
BASE_EVENT_ID = "ev_nr_squeeze"
_TIMEFRAME = "1D"


def detect_nr_squeeze(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    n: int = 7,
    atr_n: int = 14,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect NRn narrowest-range contraction bars on one asset's OHLCV.

    For every bar ``t >= max(1, n - 1)`` the bar's range ``rng[t] = high[t] -
    low[t]`` is compared against the minimum range of the prior ``n - 1`` bars; a
    squeeze fires when ``rng[t] <= min(rng[t - n + 1 .. t - 1])`` (inclusive, so
    ties fire). The window is purely backward, so the event confirms at bar ``t``
    (``formation_end_ts == confirmed_ts == index[t]``) with no neighbour-lock.

    The detector is *neutral*: it emits a single :data:`EVENT_ID` row with no
    geometric branch (same convention as ``inside_bar``). ``zone_lo`` /
    ``zone_hi`` / ``stop_ref_price`` / ``quality`` are ``NaN``; ``atr_at_confirm``
    is ``atr[t]`` and must be finite for a bar to fire. ``execution_ts`` and
    ``entry_ref_price`` come from the next bar (``NaT`` / ``NaN`` when ``t`` is the
    last bar — the row is still emitted).

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        DatetimeIndex.
    asset
        Asset label written into every emitted row.
    n
        Squeeze window length: bar ``t`` must be the narrowest range among itself
        and the prior ``n - 1`` bars (NR7 is ``n = 7``).
    atr_n
        Wilder ATR length. ``atr[t]`` is recorded as ``atr_at_confirm`` and must
        be finite for a bar to fire, so a bar in the ATR warmup region is skipped.
    logic_version
        Formation/timestamp-rule version, folded into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly
        :data:`fundcloud.research.events.schema.OBSERVATION_COLUMNS`. Empty input
        — or no contraction bars — yields an empty frame with those columns.
    """
    if bars.empty:
        return build_observations([])

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    index = bars.index
    n_bars = len(index)

    rng = high - low
    atr = wilder_atr(high, low, close, atr_n)

    params = {"n": int(n), "atr_n": int(atr_n)}
    phash = params_hash(BASE_EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    start = max(1, n - 1)
    for t in range(start, n_bars):
        atr_t = atr[t]
        if not np.isfinite(atr_t):
            continue
        prior_min = float(np.min(rng[t - n + 1 : t]))
        if not (rng[t] <= prior_min):
            continue

        has_next = t + 1 < n_bars
        execution_ts = index[t + 1] if has_next else None
        entry_ref_price = float(open_[t + 1]) if has_next else float("nan")

        rows.append(
            _row(
                asset=asset,
                confirmed_ts=index[t],
                execution_ts=execution_ts,
                params=params,
                phash=phash,
                logic_version=logic_version,
                entry_ref_price=entry_ref_price,
                atr_at_confirm=float(atr_t),
            )
        )

    return build_observations(rows)


def _row(
    *,
    asset: str,
    confirmed_ts: pd.Timestamp,
    execution_ts: pd.Timestamp | None,
    params: dict[str, object],
    phash: str,
    logic_version: int,
    entry_ref_price: float,
    atr_at_confirm: float,
) -> dict[str, object]:
    """Assemble one observation dict for :func:`build_observations`."""
    return {
        "event_id": EVENT_ID,
        "asset": asset,
        "timeframe": _TIMEFRAME,
        "formation_end_ts": confirmed_ts,
        "confirmed_ts": confirmed_ts,
        "execution_ts": execution_ts,
        "params": params,
        "logic_version": int(logic_version),
        "params_hash": phash,
        "entry_ref_price": entry_ref_price,
        "stop_ref_price": float("nan"),
        "zone_lo": float("nan"),
        "zone_hi": float("nan"),
        "quality": float("nan"),
        "atr_at_confirm": atr_at_confirm,
    }
