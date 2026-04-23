"""Tests for the strategies.scheduler module."""

from __future__ import annotations

import pandas as pd
from fundcloud.strategies.scheduler import Cadence, Scheduler


def _idx(start: str, periods: int, freq: str = "B") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(start, periods=periods, freq=freq).values)


def test_daily_fires_every_bar() -> None:
    idx = _idx("2024-01-02", 5)
    cadence = Scheduler.from_horizon("daily")
    assert list(cadence.triggers(idx)) == list(idx)


def test_weekly_every_seven_calendar_days() -> None:
    idx = _idx("2024-01-01", 30)
    cadence = Scheduler.from_horizon("weekly")
    triggers = cadence.triggers(idx)
    # Anchor is the first bar. Expected: bars at offsets 0, 7, 14, 21, 28 from anchor.
    # Because ``B`` frequency skips weekends, check only that each trigger is >= anchor+7*k
    # and the spacing in calendar days is at least 7.
    assert len(triggers) >= 4
    for prev, curr in zip(triggers[:-1], triggers[1:]):
        assert (curr - prev).days >= 7


def test_monthly_snaps_to_anchor_day() -> None:
    idx = _idx("2024-01-15", 250)  # wide enough to cover ~12 months
    cadence = Scheduler.from_horizon("monthly", anchor=pd.Timestamp("2024-01-15"))
    triggers = cadence.triggers(idx)
    assert len(triggers) >= 10
    # Each trigger's day should be ≤ 15 (snaps to last trading day ≤ 15 if needed).
    for t in triggers:
        assert t.day <= 15 or (pd.Timestamp(t) + pd.offsets.MonthEnd(0)).day < 15


def test_custom_step_string() -> None:
    idx = _idx("2024-01-01", 30)
    cadence = Scheduler.from_horizon("3D")
    triggers = cadence.triggers(idx)
    assert len(triggers) > 0


def test_scheduler_preserves_cadence_when_given_one() -> None:
    c = Cadence(step="5D")
    out = Scheduler.from_horizon(c)
    assert out.step == "5D"


def test_empty_index() -> None:
    empty = pd.DatetimeIndex([])
    assert len(Scheduler.from_horizon("daily").triggers(empty)) == 0


def test_start_end_bound_triggers() -> None:
    idx = _idx("2024-01-01", 60)
    c = Cadence(step="7D", anchor=idx[0])
    triggers = c.triggers(idx, start=idx[5], end=idx[30])
    for t in triggers:
        assert idx[5] <= t <= idx[30]
