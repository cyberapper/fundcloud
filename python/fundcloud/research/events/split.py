"""Frozen chronological train / validation / holdout split for event studies.

Event mining is multiple testing in disguise: the more variants you score, the
more spurious "edges" you find. The discipline that keeps the research honest is
a **date-based, sealed split** fixed *before* any selection happens:

* **train** — events confirmed before ``train_end``; where a variant's behaviour
  is first observed.
* **validation** — events confirmed in ``[train_end, val_end)``; where survivors
  of train are re-checked.
* **holdout** — events confirmed at or after ``val_end``; **sealed** — touched
  exactly once, at the very end, to estimate the deflated out-of-sample edge.

Selection may only read :attr:`FrozenSplit.discovery` (train + validation). The
holdout never informs a choice; reading it more than once silently turns it into
another validation fold and the deflation guarantee evaporates.

The split partitions by a timestamp column (``confirmed_ts`` for the observation
schema, ``breakout_ts`` for the projected events frame) with **half-open**
intervals so every row lands in exactly one bucket. Bars are *not* split — a
detector needs full history for ATR warmup and pivot confirmation; only the
resulting events are partitioned.

**Purge + embargo (label leakage across the cut).** A naive ``confirmed_ts``
partition still leaks: a train event's forward-label window
``[execution_ts, execution_ts + horizon)`` reaches ``horizon`` bars past its
decision bar, so an event confirmed just before ``train_end`` can consume
validation-period bars in its label — the train and validation samples then
overlap in time. :func:`frozen_split` therefore takes a label ``horizon`` (bars)
and an ``embargo`` (extra bars of gap) and *purges* any event whose label window
would reach into the next split's region: a train event is dropped when its window
can touch ``train_end``, a validation event when its window can touch ``val_end``.
Purged rows are removed entirely (not reassigned) — they belong to neither side of
the boundary. The defaults ``horizon=0, embargo=0`` reproduce the plain
partition exactly (back-compatible). See López de Prado, *Advances in Financial
ML*, ch. 7 (purged k-fold) for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["FrozenSplit", "frozen_split"]


def _utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a cut point to a UTC timestamp (naive inputs are assumed UTC)."""
    parsed = pd.Timestamp(ts)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


@dataclass(frozen=True)
class FrozenSplit:
    """One immutable train / validation / holdout partition of an events frame.

    Attributes
    ----------
    train, val, holdout
        The three disjoint row subsets, each a copy carrying the original
        columns. With purge enabled (``horizon > 0``) some boundary-band rows are
        dropped, so the row counts sum to the input's only when no purge applies.
    train_end, val_end
        The two half-open cut timestamps (UTC). ``train`` is ``ts < train_end``,
        ``val`` is ``train_end <= ts < val_end``, ``holdout`` is ``ts >= val_end``.
    ts_col
        The timestamp column the split was computed on.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    ts_col: str

    @property
    def discovery(self) -> pd.DataFrame:
        """Train + validation — the only rows selection is allowed to read.

        The holdout is deliberately excluded: anything that informs a variant
        choice must come from here so the holdout stays a true out-of-sample test.
        """
        return pd.concat([self.train, self.val], ignore_index=True)

    @property
    def sizes(self) -> dict[str, int]:
        """Row count per bucket (``train`` / ``val`` / ``holdout``)."""
        return {"train": len(self.train), "val": len(self.val), "holdout": len(self.holdout)}


def frozen_split(
    events: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    ts_col: str = "confirmed_ts",
    horizon: int = 0,
    embargo: int = 0,
) -> FrozenSplit:
    """Partition an events frame into a frozen train / validation / holdout split.

    With ``horizon == 0`` (default) this is a plain half-open partition on
    ``ts_col``. With ``horizon > 0`` it additionally **purges** any train (resp.
    validation) event whose forward-label window could reach the next split's
    region, plus an ``embargo`` gap — see the module docstring.

    Parameters
    ----------
    events
        An observation frame (:func:`fundcloud.research.events.build_observations`
        output) or a projected events frame
        (:func:`fundcloud.research.events.to_events_frame` output). Must carry
        ``ts_col`` as a tz-aware column; row order is irrelevant.
    train_end, val_end
        The two cut points, parsed to UTC timestamps. Must satisfy
        ``train_end < val_end``. Intervals are half-open: ``train`` is strictly
        before ``train_end``, ``holdout`` is at or after ``val_end``.
    ts_col
        Timestamp column to split on. ``"confirmed_ts"`` for the observation
        schema (default), ``"breakout_ts"`` for the projected events frame.
    horizon
        Forward-label window length in **bars**. The label of an event confirmed
        at bar ``t`` spans ``[t + 1, t + 1 + horizon)`` (the next-bar-open fill
        plus ``horizon`` held bars), so its last label bar is ``t + horizon``. An
        event in train is purged when ``ts + (horizon + embargo)`` bars reaches at
        or past ``train_end`` (it would consume validation bars); a validation
        event is purged symmetrically against ``val_end``. ``0`` (default)
        disables purging. **Bars are approximated as business days** — the only
        feed in scope is daily (1D) US equities, where one bar ≈ one ``BDay``;
        frozen_split has no per-asset bar calendar, so the band is measured with a
        ``pandas.tseries.offsets.BDay`` offset. Holiday gaps make this an
        approximation (it can purge a row or two extra near a holiday cluster),
        deliberately erring toward *more* purge (no leak) over less.
    embargo
        Extra bars (business days) of gap added beyond ``horizon`` so a purged
        boundary band fully separates the two samples. ``0`` (default) adds none.

    Returns
    -------
    FrozenSplit
        The immutable partition. An empty input yields three empty frames with
        the input's columns. With purging, boundary-band rows are dropped from
        train / val entirely (they appear in no bucket).

    Raises
    ------
    KeyError
        If ``ts_col`` is absent.
    ValueError
        If ``train_end >= val_end``, or ``horizon`` / ``embargo`` is negative.
    """
    if ts_col not in events.columns:
        msg = f"events frame has no {ts_col!r} column to split on"
        raise KeyError(msg)

    train_ts = _utc(train_end)
    val_ts = _utc(val_end)
    if train_ts >= val_ts:
        msg = f"train_end ({train_ts}) must be strictly before val_end ({val_ts})"
        raise ValueError(msg)
    if horizon < 0 or embargo < 0:
        msg = f"horizon ({horizon}) and embargo ({embargo}) must be non-negative"
        raise ValueError(msg)

    ts = pd.to_datetime(events[ts_col], utc=True)
    train_mask = ts < train_ts
    val_mask = (ts >= train_ts) & (ts < val_ts)
    holdout_mask = ts >= val_ts

    if horizon > 0:
        band = pd.tseries.offsets.BDay(horizon + embargo)
        # Keep a train event only when its label window stays strictly before
        # train_end: ts + (horizon + embargo) bars < train_end, i.e.
        # ts < train_end - band. A window landing exactly on train_end already
        # consumes a validation bar (half-open: train_end is the first val bar),
        # so the comparison is strict. Validation is purged symmetrically.
        train_mask &= ts < (train_ts - band)
        val_mask &= ts < (val_ts - band)

    return FrozenSplit(
        train=events.loc[train_mask].copy(),
        val=events.loc[val_mask].copy(),
        holdout=events.loc[holdout_mask].copy(),
        train_end=train_ts,
        val_end=val_ts,
        ts_col=ts_col,
    )
