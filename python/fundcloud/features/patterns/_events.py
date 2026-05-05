"""Events DataFrame schema + projection helpers.

The detector returns a list of dicts straight from the Rust scanner. Two
shapes derive from that list:

* The **events table** — a tidy ``pd.DataFrame`` with a fixed column
  schema so user code reads identically across all 9 detectors.
* The **per-bar signal panel** — a ``pd.DataFrame[index, asset]`` of
  floats produced by projecting the events onto the OHLCV time grid.
  Three projection modes mirror the :class:`SignalMode` enum.

Both shapes are stable contracts; the rest of the library (accessor,
:func:`feature_quality.evaluate`, :class:`PatternStrategy`) depends on
them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fundcloud.features.patterns._enums import Direction, Pattern, SignalMode, coerce

__all__ = [
    "EVENTS_COLUMNS",
    "build_events_frame",
    "empty_signal_series",
    "events_to_signal",
]


#: Canonical column order for the events table. Every detector returns the
#: same shape; only the row content differs.
EVENTS_COLUMNS: tuple[str, ...] = (
    "pattern",
    "asset",
    "direction",
    "formation_start",
    "formation_end",
    "breakout_ts",
    "entry_price",
    "breakout_price",
    "target_price",
    "stop_price",
    "quality",
    "variant",
    "pivots",
    "meta",
)


def build_events_frame(
    raw_events: list[dict[str, Any]],
    *,
    asset: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert raw `_core.scan_pattern` output into the canonical events frame.

    `raw_events` is the list of dicts returned by the PyO3 binding; `index`
    is the OHLCV timestamp index used to translate bar offsets into
    `pd.Timestamp` values.
    """
    if not raw_events:
        return pd.DataFrame(columns=EVENTS_COLUMNS)

    rows: list[dict[str, Any]] = []
    for ev in raw_events:
        formation_start_idx = int(ev["formation_start"])
        formation_end_idx = int(ev["formation_end"])
        rows.append({
            "pattern": Pattern(ev["name"] if "name" in ev else ev["pattern"]),
            "asset": asset,
            "direction": Direction(ev["direction"]),
            "formation_start": index[formation_start_idx],
            "formation_end": index[formation_end_idx],
            "breakout_ts": index[formation_end_idx],  # v1: breakout = right edge
            "entry_price": _nan_or_float(ev.get("entry_price")),
            "breakout_price": _nan_or_float(ev.get("breakout_price")),
            "target_price": float("nan"),
            "stop_price": float("nan"),
            "quality": float(ev.get("quality", float("nan"))),
            "variant": ev.get("variant"),
            "pivots": [
                {"ts": index[int(p["index"])], "price": float(p["price"]), "kind": p["kind"]}
                for p in ev.get("pivots", [])
            ],
            "meta": {
                "features": ev.get("features", {}),
                "trend_lines": ev.get("trend_lines", []),
                "scorer_version": ev.get("scorer_version"),
            },
        })
    return pd.DataFrame(rows, columns=list(EVENTS_COLUMNS))


def events_to_signal(
    events: pd.DataFrame,
    *,
    index: pd.DatetimeIndex,
    mode: SignalMode | str = SignalMode.BREAKOUT,
    decay_bars: int = 5,
) -> pd.Series:
    """Project an events table onto a per-bar float signal series.

    Parameters
    ----------
    events
        Events table built by :func:`build_events_frame`. May be empty.
    index
        Output index — the OHLCV timestamp index.
    mode
        Projection mode. ``BREAKOUT`` puts ``1.0`` on each breakout bar
        and ``0.0`` everywhere else. ``FORMATION`` puts ``1.0`` from
        ``formation_start`` to ``formation_end`` inclusive. ``DECAY``
        starts at ``1.0`` on the breakout bar and decays linearly to
        ``0.0`` over ``decay_bars`` bars.
    decay_bars
        Window for the ``DECAY`` mode.
    """
    mode = coerce(mode, SignalMode)
    out = pd.Series(0.0, index=index, dtype=np.float64)
    if events.empty:
        return out

    # Translate timestamps back to integer positions once.
    pos = pd.Series(np.arange(len(index)), index=index)

    if mode is SignalMode.BREAKOUT:
        for ts in events["breakout_ts"]:
            if pd.isna(ts):
                continue
            if ts in pos.index:
                out.iloc[int(pos[ts])] = 1.0  # type: ignore[call-overload]
        return out

    if mode is SignalMode.FORMATION:
        for _, ev in events.iterrows():
            start_ts = ev["formation_start"]
            end_ts = ev["formation_end"]
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            if start_ts not in pos.index or end_ts not in pos.index:
                continue
            i0 = int(pos[start_ts])
            i1 = int(pos[end_ts])
            out.iloc[i0 : i1 + 1] = 1.0  # type: ignore[call-overload]
        return out

    # DECAY
    if decay_bars <= 0:
        msg = "decay_bars must be positive when mode == SignalMode.DECAY"
        raise ValueError(msg)
    n = len(index)
    for ts in events["breakout_ts"]:
        if pd.isna(ts) or ts not in pos.index:
            continue
        i0 = int(pos[ts])
        for k in range(decay_bars):
            j = i0 + k
            if j >= n:
                break
            value = 1.0 - (k / decay_bars)
            if value > out.iloc[j]:
                out.iloc[j] = value  # type: ignore[call-overload]
    return out


def empty_signal_series(index: pd.DatetimeIndex) -> pd.Series:
    """Zero-filled per-bar signal — used when there are no events."""
    return pd.Series(0.0, index=index, dtype=np.float64)


def _nan_or_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)
