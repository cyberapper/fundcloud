"""Tests for ``period_returns`` and ``runup_details``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.metrics import (
    drawdown_details,
    period_returns,
    runup_details,
    yearly_returns,
)


@pytest.fixture
def ten_year_series() -> pd.Series:
    idx = pd.date_range("2015-01-02", periods=2500, freq="B")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.012, 2500), index=idx, name="Strategy")


@pytest.fixture
def ten_year_benchmark() -> pd.Series:
    idx = pd.date_range("2015-01-02", periods=2500, freq="B")
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.0003, 0.010, 2500), index=idx, name="SPY")


# -------------------------------------------------------------------- period_returns


def test_period_returns_series_shape(ten_year_series: pd.Series) -> None:
    out = period_returns(ten_year_series)
    assert isinstance(out, pd.Series)
    assert list(out.index) == [
        "MTD",
        "3M",
        "6M",
        "YTD",
        "1Y",
        "3Y (ann.)",
        "5Y (ann.)",
        "10Y (ann.)",
        "All-time (ann.)",
    ]


def test_period_returns_with_benchmark_returns_dataframe(
    ten_year_series: pd.Series, ten_year_benchmark: pd.Series
) -> None:
    out = period_returns(ten_year_series, benchmark=ten_year_benchmark)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["SPY", "Strategy"]
    assert len(out) == 9


def test_period_returns_ytd_matches_manual_compound(
    ten_year_series: pd.Series,
) -> None:
    """YTD should equal (1+r).prod() - 1 restricted to the final year."""
    anchor = ten_year_series.index[-1]
    ytd_cutoff = anchor.normalize().replace(month=1, day=1)
    expected = float((1.0 + ten_year_series.loc[ytd_cutoff:]).prod() - 1.0)
    actual = float(period_returns(ten_year_series).loc["YTD"])
    assert actual == pytest.approx(expected, rel=1e-12)


def test_period_returns_all_time_equals_cagr(ten_year_series: pd.Series) -> None:
    out = period_returns(ten_year_series, periods_per_year=252)
    years = len(ten_year_series) / 252
    total = float((1.0 + ten_year_series).prod() - 1.0)
    expected = (1.0 + total) ** (1.0 / years) - 1.0
    assert out.loc["All-time (ann.)"] == pytest.approx(expected, rel=1e-10)


def test_period_returns_short_series_long_windows_collapse_to_all_time() -> None:
    """When the sample is shorter than a period window, the cutoff
    predates the first bar — so the window equals the full sample and
    all long-period CAGR values collapse to the All-time value."""
    idx = pd.date_range("2024-01-02", periods=120, freq="B")
    r = pd.Series(np.full(120, 0.001), index=idx, name="s")
    out = period_returns(r)
    all_time = float(out.loc["All-time (ann.)"])
    assert float(out.loc["3Y (ann.)"]) == pytest.approx(all_time)
    assert float(out.loc["5Y (ann.)"]) == pytest.approx(all_time)
    assert float(out.loc["10Y (ann.)"]) == pytest.approx(all_time)


def test_period_returns_dataframe_input_gives_dataframe() -> None:
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 0.01, 400), "b": rng.normal(0, 0.01, 400)}, index=idx)
    out = period_returns(df)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"a", "b"}


def test_period_returns_requires_datetime_index() -> None:
    r = pd.Series([0.01, -0.005, 0.02])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        period_returns(r)


# -------------------------------------------------------------------- runup_details


def test_runup_details_shape_columns(ten_year_series: pd.Series) -> None:
    ru = runup_details(ten_year_series)
    assert list(ru.columns) == [
        "start",
        "peak",
        "end",
        "max_runup",
        "duration_days",
        "days_after_peak",
    ]


def test_runup_details_sorted_descending_by_magnitude(
    ten_year_series: pd.Series,
) -> None:
    ru = runup_details(ten_year_series)
    assert not ru.empty
    vals = ru["max_runup"].to_numpy()
    assert (vals[:-1] >= vals[1:]).all()


def test_runup_details_all_positive_magnitudes(ten_year_series: pd.Series) -> None:
    ru = runup_details(ten_year_series)
    assert (ru["max_runup"] > 0).all()


def test_runup_details_episodes_are_complements_of_drawdowns(
    ten_year_series: pd.Series,
) -> None:
    """Runup episodes cannot overlap drawdown episodes — runup ``end``
    must match a drawdown ``start`` (or NaT for the trailing leg)."""
    ru = runup_details(ten_year_series).sort_values("start")
    dd = drawdown_details(ten_year_series).sort_values("start")
    dd_starts = set(dd["start"].dropna())
    for _, row in ru.iterrows():
        if pd.isna(row["end"]):
            continue
        assert row["end"] in dd_starts


def test_runup_details_empty_series() -> None:
    ru = runup_details(pd.Series(dtype=float))
    assert ru.empty


def test_runup_details_monotone_upward_series_produces_one_open_episode() -> None:
    """A strictly-upward return series has no drawdowns — one trailing runup."""
    idx = pd.date_range("2024-01-02", periods=30, freq="B")
    r = pd.Series(np.full(30, 0.001), index=idx, name="s")
    ru = runup_details(r)
    assert len(ru) == 1
    assert pd.isna(ru.iloc[0]["end"])
    assert ru.iloc[0]["max_runup"] > 0


def test_yearly_returns_output_shape(ten_year_series: pd.Series) -> None:
    """yearly_returns should indexing by year integers."""
    out = yearly_returns(ten_year_series)
    assert out.index.dtype.kind in {"i", "u"}
