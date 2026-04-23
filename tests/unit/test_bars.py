"""Tests for ``fundcloud.data.bars`` conversions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import bars as B


def test_to_prices_with_multiindex(ohlcv_panel: pd.DataFrame) -> None:
    prices = B.to_prices(ohlcv_panel, field="close")
    assert isinstance(prices.columns, pd.Index)
    assert not isinstance(prices.columns, pd.MultiIndex)
    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.shape == (50, 2)


def test_to_prices_unknown_field_raises(ohlcv_panel: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        B.to_prices(ohlcv_panel, field="adjusted_close")  # type: ignore[arg-type]


def test_to_returns_simple_and_log_match_asymptotically() -> None:
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4], index=idx)
    simple = B.to_returns(prices, method="simple")
    log = B.to_returns(prices, method="log")
    # For small returns, simple ≈ log to a few decimals.
    assert np.allclose(simple.values, log.values, atol=1e-4)


def test_to_returns_requires_datetime_index() -> None:
    s = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        B.to_returns(s)


def test_align_inner(ohlcv_panel: pd.DataFrame) -> None:
    a = B.to_prices(ohlcv_panel)
    b = a.iloc[5:, :1]
    aligned_a, aligned_b = B.align(a, b, how="inner")
    assert aligned_a.index.equals(aligned_b.index)
    assert list(aligned_a.columns) == list(aligned_b.columns) == ["AAA"]


def test_resample_downsamples_weekly(ohlcv_panel: pd.DataFrame) -> None:
    weekly = B.resample(ohlcv_panel, "W-FRI")
    assert len(weekly) < len(ohlcv_panel)
    # Weekly close equals the last daily close in the period.
    closes_daily = ohlcv_panel[("close", "AAA")]
    expected_first = closes_daily.loc[: weekly.index[0]].iloc[-1]
    assert np.isclose(weekly[("close", "AAA")].iloc[0], expected_first)


def test_long_wide_roundtrip(ohlcv_panel: pd.DataFrame) -> None:
    prices = B.to_prices(ohlcv_panel)
    long = B.as_long(prices, value_name="price")
    wide = B.as_wide(long, value="price")
    pd.testing.assert_frame_equal(
        wide.reindex_like(prices).astype(float),
        prices.astype(float),
    )
