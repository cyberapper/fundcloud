"""Bar-local displacement detector (``ev_disp_bar``).

A displacement bar is a single decisive candle: its body spans more than a
volatility-scaled threshold and it closes near the extreme it pushed toward.
Both conditions read only bar ``t`` itself plus an ATR known at ``t-1``, so the
event confirms *at* bar ``t`` with zero future bars — the simplest point on the
registry's causal contract (``docs/guides/research/event-registry.md``).

For each bar ``t`` (with ``t >= atr_n`` so ``atr[t-1]`` exists):

* ``rng = high - low``; bars with ``rng <= 0`` are skipped (no body to scale).
* ``clv = (close - low) / rng`` — the close's location within the bar's range.
* **bullish** when ``(close - open) > z_body * atr[t-1]`` and ``clv >= clv_min``,
* **bearish** when ``(open - close) > z_body * atr[t-1]`` and ``clv <= 1 - clv_min``.
* optional volume gate (when ``z_vol`` is not ``None``):
  ``volume[t] / mean(volume[t-atr_n .. t-1]) >= z_vol``.

Bullish and bearish are separate rows. The event carries no zone or stop
(``zone_lo``/``zone_hi``/``stop_ref_price``/``quality`` are ``NaN``);
``atr_at_confirm`` is ``atr[t]``. ``confirmed_ts == formation_end_ts == index[t]``;
``execution_ts``/``entry_ref_price`` read the next bar's open per the execution
contract (``NaT``/``NaN`` when ``t`` is the last bar — the row is still emitted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events._causality import wilder_atr
from fundcloud.research.events.schema import (
    build_observations,
    params_hash,
)

__all__ = ["detect_displacement"]

EVENT_ID = "ev_disp_bar"


def detect_displacement(
    bars: pd.DataFrame,
    *,
    asset: str = "",
    atr_n: int = 14,
    z_body: float = 1.0,
    clv_min: float = 0.7,
    z_vol: float | None = None,
    logic_version: int = 1,
) -> pd.DataFrame:
    """Detect bar-local displacement candles in single-asset OHLCV.

    Parameters
    ----------
    bars
        Single-asset OHLCV frame with lowercase ``open``/``high``/``low``/
        ``close``/``volume`` columns on a sorted, unique, tz-aware UTC
        DatetimeIndex.
    asset
        Asset label written to every emitted row.
    atr_n
        Wilder ATR length; also the look-back window for the optional volume
        average. A bar at ``t`` is only evaluated once ``atr[t-1]`` is defined,
        i.e. ``t >= atr_n``.
    z_body
        Body threshold in ATR units: the candle body must exceed
        ``z_body * atr[t-1]``.
    clv_min
        Minimum close-location value for a bullish bar (a bearish bar requires
        ``clv <= 1 - clv_min``). ``0.7`` keeps closes in the top/bottom 30%.
    z_vol
        Optional volume multiple. When set, the bar's volume must be at least
        ``z_vol`` times the trailing ``atr_n``-bar mean volume. ``None`` disables
        the gate.
    logic_version
        Formation/timestamp-rule version stamped into ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        Observation frame with exactly :data:`OBSERVATION_COLUMNS`. Empty input
        or no detections yields an empty frame with those columns.
    """
    if bars.empty:
        return build_observations([])

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    index = bars.index

    atr = wilder_atr(high, low, close, atr_n)
    n_bars = len(close)

    params = {
        "atr_n": int(atr_n),
        "z_body": float(z_body),
        "clv_min": float(clv_min),
        "z_vol": None if z_vol is None else float(z_vol),
    }
    phash = params_hash(EVENT_ID, params, logic_version)

    rows: list[dict[str, object]] = []
    for t in range(atr_n, n_bars):
        atr_prev = atr[t - 1]
        if not np.isfinite(atr_prev):
            continue
        rng = high[t] - low[t]
        if rng <= 0:
            continue

        if z_vol is not None:
            ref_vol = float(np.mean(volume[t - atr_n : t]))
            if not (ref_vol > 0 and volume[t] / ref_vol >= z_vol):
                continue

        clv = (close[t] - low[t]) / rng
        body_thresh = z_body * atr_prev

        is_bull = (close[t] - open_[t]) > body_thresh and clv >= clv_min
        is_bear = (open_[t] - close[t]) > body_thresh and clv <= 1.0 - clv_min
        if not (is_bull or is_bear):
            continue

        has_next = t + 1 < n_bars
        execution_ts = index[t + 1] if has_next else None
        entry_ref_price = float(open_[t + 1]) if has_next else float("nan")
        atr_at_confirm = float(atr[t]) if np.isfinite(atr[t]) else float("nan")

        if is_bull:
            rows.append(
                _row(
                    asset=asset,
                    confirmed_ts=index[t],
                    execution_ts=execution_ts,
                    direction="bullish",
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                    entry_ref_price=entry_ref_price,
                    atr_at_confirm=atr_at_confirm,
                )
            )
        if is_bear:
            rows.append(
                _row(
                    asset=asset,
                    confirmed_ts=index[t],
                    execution_ts=execution_ts,
                    direction="bearish",
                    params=params,
                    phash=phash,
                    logic_version=logic_version,
                    entry_ref_price=entry_ref_price,
                    atr_at_confirm=atr_at_confirm,
                )
            )

    return build_observations(rows)


def _row(
    *,
    asset: str,
    confirmed_ts: pd.Timestamp,
    execution_ts: pd.Timestamp | None,
    direction: str,
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
        "timeframe": "1D",
        "formation_end_ts": confirmed_ts,
        "confirmed_ts": confirmed_ts,
        "execution_ts": execution_ts,
        "direction": direction,
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
