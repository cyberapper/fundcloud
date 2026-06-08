"""Unit tests for fundcloud.research.quality (gate + cleaner)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.quality import check_quality, clean_panel


def _panel(n: int = 12) -> pd.DataFrame:
    """A clean two-asset OHLCV panel on a tz-aware UTC business-day index."""
    idx = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    data: dict[tuple[str, str], pd.Series] = {}
    for sym in ("AAA", "BBB"):
        close = pd.Series(100.0 + np.arange(n), index=idx)
        data[("open", sym)] = close - 0.5
        data[("high", sym)] = close + 1.0
        data[("low", sym)] = close - 1.0
        data[("close", sym)] = close
        data[("volume", sym)] = pd.Series(1_000_000.0, index=idx)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)


def test_clean_panel_passes_gate() -> None:
    report = check_quality(_panel(), min_days=5)
    assert report.ok
    assert report.high_lt_low == 0
    assert report.null_ohlc == 0
    assert report.n_symbols == 2
    assert report.symbols_below_min_days == ()


def test_high_lt_low_is_fatal_and_removed() -> None:
    df = _panel()
    bad_date = df.index[3]
    df.loc[bad_date, ("high", "AAA")] = df.loc[bad_date, ("low", "AAA")] - 5.0

    report = check_quality(df, min_days=5)
    assert report.high_lt_low >= 1
    assert not report.ok

    clean, clean_report = clean_panel(df)
    assert clean_report.ok
    # the offending AAA bar is gone, BBB on that date survives
    assert pd.isna(clean.xs("close", axis=1, level=0).loc[bad_date, "AAA"])
    assert not pd.isna(clean.xs("close", axis=1, level=0).loc[bad_date, "BBB"])


def test_partial_bar_is_null_ohlc_fatal() -> None:
    df = _panel()
    df.loc[df.index[2], ("open", "AAA")] = np.nan
    report = check_quality(df, min_days=5)
    assert report.null_ohlc >= 1
    assert not report.ok
    _clean, clean_report = clean_panel(df)
    assert clean_report.ok


def test_non_monotonic_index_is_fatal() -> None:
    df = _panel().iloc[::-1]
    report = check_quality(df, min_days=5)
    assert not report.monotonic_index
    assert not report.ok


def test_duplicate_dates_fatal_then_fixed_by_clean() -> None:
    df = _panel()
    dup = pd.concat([df, df.iloc[[0]]])
    report = check_quality(dup, min_days=5)
    assert report.duplicate_dates >= 1
    assert not report.ok
    _clean, clean_report = clean_panel(dup)
    assert clean_report.duplicate_dates == 0
    assert clean_report.ok


def test_zero_volume_is_warn_only() -> None:
    df = _panel()
    df.loc[df.index[1], ("volume", "AAA")] = 0.0
    report = check_quality(df, min_days=5)
    assert report.zero_volume >= 1
    assert report.ok  # volume is a warning, not fatal
    clean, _clean_report = clean_panel(df)
    # cleaner does not strip zero-volume bars
    assert clean.xs("volume", axis=1, level=0).loc[df.index[1], "AAA"] == 0.0


def test_nonpositive_price_is_fatal() -> None:
    df = _panel()
    df.loc[df.index[4], ("close", "BBB")] = -1.0
    report = check_quality(df, min_days=5)
    assert report.nonpositive_price >= 1
    assert not report.ok


def test_symbols_below_min_days_is_warn() -> None:
    df = _panel(n=10)
    report = check_quality(df, min_days=250)
    assert set(report.symbols_below_min_days) == {"AAA", "BBB"}
    assert report.ok  # short history is a warning, not fatal
