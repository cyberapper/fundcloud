"""Observation schema for the event-study engine + the reuse projection.

Every detector emits one tidy row per detected event. :data:`OBSERVATION_COLUMNS`
fixes that row's shape so user code reads identically across detectors, and
:func:`build_observations` assembles a list of detector dicts into a frame with
exactly those columns (missing keys filled with ``NaN`` / ``None``, all ``*_ts``
columns coerced to ``datetime64[ns, UTC]``).

:func:`to_events_frame` projects the observation frame onto the handful of fields
:func:`fundcloud.metrics.feature_quality.evaluate` actually reads — its
``_build_event_paths`` only ``.get()``s ``asset``, ``breakout_ts``,
``long_entry`` / ``short_entry``, ``stop_price`` and ``quality`` — so a one-line
rename feeds the existing evaluation engine with no engine change. ``confirmed_ts``
maps to ``breakout_ts``, ``entry_ref_price`` splits into ``long_entry`` /
``short_entry`` by the event_id orientation suffix (``_up`` → ``long_entry``,
``_dn`` → ``short_entry``), and ``stop_ref_price`` maps to ``stop_price``. That
suffix is a LEGACY geometric bridge only: it is the detection equation's branch,
**not** a trade mandate — the evidence layer decides the actual side from data.

:func:`params_hash` produces a short stable hex digest of ``(event_id, params,
logic_version)`` so re-definitions never pool incomparable samples.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

__all__ = [
    "OBSERVATION_COLUMNS",
    "build_observations",
    "params_hash",
    "to_events_frame",
]

#: Canonical column order for one detected event (the "known at confirmation"
#: half of the registry's observation schema). The forward-path metrics the
#: engine computes are *not* listed here — the detector never writes them.
OBSERVATION_COLUMNS: tuple[str, ...] = (
    "event_id",
    "asset",
    "timeframe",
    "formation_end_ts",
    "confirmed_ts",
    "execution_ts",
    "params",
    "logic_version",
    "params_hash",
    "entry_ref_price",
    "stop_ref_price",
    "zone_lo",
    "zone_hi",
    "quality",
    "atr_at_confirm",
)

#: The subset of :data:`OBSERVATION_COLUMNS` carrying tz-aware timestamps.
_TS_COLUMNS: tuple[str, ...] = ("formation_end_ts", "confirmed_ts", "execution_ts")


def build_observations(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble detector rows into the canonical observation frame.

    Parameters
    ----------
    rows
        One dict per detected event. Any key missing from a row is filled with
        ``None`` (which pandas renders as ``NaN`` for numeric columns); keys not
        in :data:`OBSERVATION_COLUMNS` are dropped.

    Returns
    -------
    pandas.DataFrame
        A frame whose columns are exactly :data:`OBSERVATION_COLUMNS` in order.
        Empty input yields an empty frame with those columns. Every ``*_ts``
        column is coerced to ``datetime64[ns, UTC]``.
    """
    if not rows:
        empty = pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
        for col in _TS_COLUMNS:
            empty[col] = pd.to_datetime(empty[col], utc=True)
        return empty

    normalized = [{col: row.get(col) for col in OBSERVATION_COLUMNS} for row in rows]
    frame = pd.DataFrame(normalized, columns=list(OBSERVATION_COLUMNS))
    for col in _TS_COLUMNS:
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def params_hash(event_id: str, params: Mapping[str, Any], logic_version: int) -> str:
    """Short stable hex digest identifying an event definition.

    Hashes the canonical JSON of ``(event_id, params, logic_version)`` with
    ``sort_keys=True`` so the digest is order-independent and reproducible across
    runs/processes. Returns the first 12 hex characters of the SHA-1 digest —
    enough to keep distinct parameterisations from pooling without bloating rows.

    Parameters
    ----------
    event_id
        Catalog id (e.g. ``"ev_gap_up"``). For the geometric-branch detectors
        the digest is keyed on the detector's *base* id (e.g. ``"ev_gap_imb_3c"``)
        so both ``_up`` / ``_dn`` branches of one call share one ``params_hash``.
    params
        Resolved detector parameters.
    logic_version
        Formation/timestamp-rule version; bump on any redefinition.
    """
    payload = json.dumps(
        {"event_id": event_id, "params": dict(params), "logic_version": int(logic_version)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def to_events_frame(obs: pd.DataFrame) -> pd.DataFrame:
    """Project observations onto the columns ``feature_quality.evaluate`` reads.

    One output row per observation. The engine's ``_build_event_paths`` anchors
    at ``breakout_ts`` and measures the forward path from the next bar against a
    per-direction entry, so this maps:

    * ``confirmed_ts`` → ``breakout_ts`` (the leak-free anchor),
    * ``entry_ref_price`` → ``long_entry`` where ``event_id`` ends in ``"_up"``,
      else ``NaN``; → ``short_entry`` where ``event_id`` ends in ``"_dn"``, else
      ``NaN`` (next-bar-open fill, per the execution contract),
    * ``stop_ref_price`` → ``stop_price``,
    * ``quality`` → ``quality``,
    * ``event_id`` → ``pattern`` (so per-event grouping survives).

    The ``_up`` / ``_dn`` split is a **LEGACY** compatibility bridge to
    ``feature_quality.evaluate`` (which reads ``long_entry`` for a long path and
    ``short_entry`` for a short path). The suffix is the detection equation's
    *geometric branch* — e.g. ``low[t] > high[t-2]`` is the up-gap branch — and is
    **not** a trade direction. Downstream trade side is data-driven
    (``side="auto"``) and the suffix must never drive a trade in the
    evidence / portfolio layer. An ``event_id`` carrying neither suffix yields
    ``NaN`` in both entry columns (such rows are dropped by ``evaluate``).

    Parameters
    ----------
    obs
        An observation frame shaped like :func:`build_observations` output.

    Returns
    -------
    pandas.DataFrame
        Columns ``asset``, ``breakout_ts``, ``long_entry``, ``short_entry``,
        ``stop_price``, ``quality``, ``pattern``.
    """
    columns = ["asset", "breakout_ts", "long_entry", "short_entry", "stop_price", "quality", "pattern"]
    if obs.empty:
        out = pd.DataFrame(columns=columns)
        out["breakout_ts"] = pd.to_datetime(out["breakout_ts"], utc=True)
        return out

    suffix = obs["event_id"].astype(str).str[-3:]
    is_long = suffix == "_up"
    is_short = suffix == "_dn"
    entry = obs["entry_ref_price"]
    return pd.DataFrame(
        {
            "asset": obs["asset"].to_numpy(),
            "breakout_ts": obs["confirmed_ts"].to_numpy(),
            "long_entry": entry.where(is_long),
            "short_entry": entry.where(is_short),
            "stop_price": obs["stop_ref_price"].to_numpy(),
            "quality": obs["quality"].to_numpy(),
            "pattern": obs["event_id"].to_numpy(),
        },
        columns=columns,
    )
