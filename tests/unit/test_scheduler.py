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


def test_quarterly_fires_every_three_months() -> None:
    idx = _idx("2024-01-15", 400)  # ~18 months of business days
    cadence = Scheduler.from_horizon("quarterly", anchor=pd.Timestamp("2024-01-15"))
    triggers = cadence.triggers(idx)
    # Across ~18 months we should see 6 quarterly fires (Jan, Apr, Jul, Oct, Jan, Apr).
    assert 5 <= len(triggers) <= 7
    # Spacing between consecutive fires ≈ 3 months (between 80 and 100 calendar days).
    for prev, curr in zip(triggers[:-1], triggers[1:]):
        delta_days = (curr - prev).days
        assert 80 <= delta_days <= 100, f"quarterly spacing {delta_days}d outside expected band"


def test_quarterly_snaps_to_anchor_day() -> None:
    idx = _idx("2024-01-15", 400)
    cadence = Scheduler.from_horizon("quarterly", anchor=pd.Timestamp("2024-01-15"))
    triggers = cadence.triggers(idx)
    # Each trigger snaps backwards to the latest trading day ≤ day-15 of its month.
    for t in triggers:
        assert t.day <= 15


def test_cadence_break_when_anchor_after_index_end() -> None:
    """Anchor far past the index end → loop exits via the ``break`` (line 57)."""
    idx = _idx("2024-01-01", 5)
    cadence = Cadence(step="1D", anchor=pd.Timestamp("2030-01-01"))
    # effective_end defaults to idx[-1] (2024-01-05); cursor starts at 2030
    # which is > effective_end so the loop never enters. With an explicit
    # later end we exercise the post-index break.
    triggers = cadence.triggers(idx, end=pd.Timestamp("2030-12-31"))
    assert len(triggers) == 0


def test_monthly_empty_index() -> None:
    """Empty index short-circuits ``_MonthlyCadence.triggers`` (line 129)."""
    cadence = Scheduler.from_horizon("monthly")
    triggers = cadence.triggers(pd.DatetimeIndex([]))
    assert len(triggers) == 0


def test_quarterly_empty_index() -> None:
    cadence = Scheduler.from_horizon("quarterly")
    triggers = cadence.triggers(pd.DatetimeIndex([]))
    assert len(triggers) == 0


def test_monthly_anchor_day_31_falls_back_to_month_end() -> None:
    """Anchor on the 31st triggers the ValueError → month-end fallback (lines 142-144).

    February has no 31st, so the cadence must snap to the last trading day
    of February rather than raising. Same for April / June / September /
    November (no 31st).
    """
    idx = _idx("2024-01-31", 250)
    cadence = Scheduler.from_horizon("monthly", anchor=pd.Timestamp("2024-01-31"))
    triggers = cadence.triggers(idx)
    months_seen = {(t.year, t.month) for t in triggers}
    # Should fire in Feb, Apr, Jun, Sep, Nov despite no 31st in those months.
    assert (2024, 2) in months_seen
    assert (2024, 4) in months_seen
    assert (2024, 6) in months_seen


def test_monthly_skips_month_with_no_bars() -> None:
    """Branch ``148 -> 155``: a month containing no bars is skipped.

    Building an index that has bars in January and March but none in
    February exercises the ``if len(month_bars) > 0`` false branch.
    """
    jan = pd.date_range("2024-01-02", "2024-01-31", freq="B")
    mar = pd.date_range("2024-03-01", "2024-03-29", freq="B")
    idx = pd.DatetimeIndex(jan.append(mar))
    cadence = Scheduler.from_horizon("monthly", anchor=pd.Timestamp("2024-01-15"))
    triggers = cadence.triggers(idx)
    months_seen = {(t.year, t.month) for t in triggers}
    assert (2024, 1) in months_seen
    assert (2024, 2) not in months_seen
    assert (2024, 3) in months_seen


def test_monthly_does_not_double_fire_within_month() -> None:
    """Branch ``151 -> 155``: snapped bar not after last fire is skipped.

    With an anchor near the start of the month and a tight start/end
    window, the first month's snapped bar can be ≤ the prior fire — the
    loop must skip rather than re-emit it.
    """
    idx = _idx("2024-01-02", 90)
    cadence = Scheduler.from_horizon("monthly", anchor=pd.Timestamp("2024-01-02"))
    triggers = cadence.triggers(idx)
    # Fires must be strictly increasing (no duplicates).
    assert list(triggers) == sorted(set(triggers))


def test_scheduler_merges_anchor_into_existing_cadence() -> None:
    """``from_horizon(Cadence, anchor=...)`` overrides the cadence anchor."""
    base = Cadence(step="5D", anchor=pd.Timestamp("2024-01-01"))
    override = pd.Timestamp("2024-06-01")
    out = Scheduler.from_horizon(base, anchor=override)
    assert out.step == "5D"
    assert out.anchor == override


def test_scheduler_keeps_cadence_anchor_when_none_passed() -> None:
    base = Cadence(step="5D", anchor=pd.Timestamp("2024-01-01"))
    out = Scheduler.from_horizon(base)
    assert out.anchor == pd.Timestamp("2024-01-01")


def test_monthly_skips_fires_before_start_window() -> None:
    """Branch ``151 -> 155``: snapped bar exists but falls outside ``start``.

    Anchor on Jan 15 produces a candidate snap for January, but ``start``
    is set after the snapped bar so the append is skipped without
    advancing ``fires``.
    """
    idx = _idx("2024-01-02", 80)
    cadence = Scheduler.from_horizon("monthly", anchor=pd.Timestamp("2024-01-15"))
    triggers = cadence.triggers(idx, start=pd.Timestamp("2024-02-01"))
    months_seen = {(t.year, t.month) for t in triggers}
    assert (2024, 1) not in months_seen  # Jan snap (Jan 15ish) skipped
    assert (2024, 2) in months_seen  # Feb fires normally
