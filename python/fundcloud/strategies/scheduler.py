"""``Cadence`` + ``Scheduler`` — when does a strategy fire?

Strategies driven by the simulator express their rebalance / investment
rhythm through a :class:`Cadence`. Three named presets — ``daily`` /
``weekly`` / ``monthly`` — ship today, plus arbitrary
``pandas.Timedelta``-compatible step strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = ["Cadence", "Scheduler"]


HorizonName = Literal["daily", "weekly", "monthly"]


@dataclass(slots=True)
class Cadence:
    """An explicit cadence step. Use :meth:`Scheduler.from_horizon` for presets.

    ``step`` is any pandas-parseable offset like ``"7D"``, ``"14D"``, or
    ``"1D"``. Anchor is the first timestamp the cadence fires on; if
    unspecified the first trigger falls on the first timestamp in the passed
    index.
    """

    step: str = "1D"
    anchor: pd.Timestamp | None = None

    def triggers(
        self,
        index: pd.DatetimeIndex,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DatetimeIndex:
        """Return the subset of ``index`` that the cadence fires on."""
        if len(index) == 0:
            return pd.DatetimeIndex([])
        effective_start = pd.Timestamp(start) if start is not None else index[0]
        effective_end = pd.Timestamp(end) if end is not None else index[-1]
        anchor = self.anchor or effective_start

        step = pd.Timedelta(self.step)
        fires: list[pd.Timestamp] = []
        cursor = anchor
        while cursor <= effective_end:
            if cursor >= effective_start:
                # Snap to the next available bar in ``index`` (inclusive).
                pos = index.searchsorted(cursor, side="left")
                if pos >= len(index):
                    break
                snapped = index[pos]
                if snapped <= effective_end and (not fires or snapped > fires[-1]):
                    fires.append(snapped)
            cursor = cursor + step
        return pd.DatetimeIndex(fires)


class Scheduler:
    """Factory for :class:`Cadence` presets."""

    @staticmethod
    def from_horizon(
        horizon: HorizonName | Cadence | str,
        *,
        anchor: pd.Timestamp | None = None,
    ) -> Cadence:
        """Resolve a user-facing horizon into a concrete :class:`Cadence`.

        Accepted forms:

        * ``"daily"``  → every trading day (``1D`` step anchored at ``anchor``).
        * ``"weekly"`` → every 7 calendar days from ``anchor``
          (**not** ISO weekday 1 — matches PRD's "(7 days)" wording).
        * ``"monthly"`` → same day-of-month as ``anchor`` each month,
          falling back to the last trading day of the month if missing.
        * a :class:`Cadence` is returned as-is (with ``anchor`` merged in).
        * any pandas offset string (e.g. ``"30D"``, ``"3W"``) → :class:`Cadence`.
        """
        if isinstance(horizon, Cadence):
            return Cadence(step=horizon.step, anchor=anchor or horizon.anchor)
        if horizon == "daily":
            return Cadence(step="1D", anchor=anchor)
        if horizon == "weekly":
            return Cadence(step="7D", anchor=anchor)
        if horizon == "monthly":
            return _MonthlyCadence(anchor=anchor)  # type: ignore[return-value]
        # Treat anything else as a pandas-parseable offset.
        try:
            pd.Timedelta(horizon)
        except ValueError as e:  # pragma: no cover
            msg = f"unknown horizon: {horizon!r}"
            raise ValueError(msg) from e
        return Cadence(step=str(horizon), anchor=anchor)


class _MonthlyCadence(Cadence):
    """Monthly cadence that snaps to same-day-of-month.

    The rule is "same day-of-month as ``start``, or the last trading day
    of the month if that day is missing." We snap **backwards** to the
    most recent trading day ≤ the anchor day within the same month,
    falling back to the latest bar of the month only when every bar in
    the month is strictly after the anchor day.
    """

    def triggers(
        self,
        index: pd.DatetimeIndex,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DatetimeIndex:
        if len(index) == 0:
            return pd.DatetimeIndex([])
        effective_start = pd.Timestamp(start) if start is not None else index[0]
        effective_end = pd.Timestamp(end) if end is not None else index[-1]
        anchor = self.anchor or effective_start
        day = anchor.day
        fires: list[pd.Timestamp] = []

        cursor = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
        while cursor <= effective_end:
            month_end = cursor + pd.offsets.MonthEnd(0)
            try:
                target = pd.Timestamp(year=cursor.year, month=cursor.month, day=day)
                if target > month_end:
                    target = month_end
            except ValueError:
                target = month_end

            month_mask = (index >= cursor) & (index <= month_end)
            month_bars = index[month_mask]
            if len(month_bars) > 0:
                at_or_before = month_bars[month_bars <= target]
                snapped = at_or_before[-1] if len(at_or_before) > 0 else month_bars[-1]
                if effective_start <= snapped <= effective_end and (
                    not fires or snapped > fires[-1]
                ):
                    fires.append(snapped)
            cursor = cursor + pd.offsets.MonthBegin(1)
        return pd.DatetimeIndex(fires)
