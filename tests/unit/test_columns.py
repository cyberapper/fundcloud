"""Tests for the canonical OHLCV column-normalisation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import (
    OHLCV_COLUMNS,
    canonicalize_ohlcv_order,
    normalize_field,
    normalize_ohlcv_columns,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Open", "open"),
        ("CLOSE", "close"),
        ("AdjClose", "adj_close"),
        ("Adj Close", "adj_close"),
        ("adj-close", "adj_close"),
        ("VWAP", "vwap"),
        ("unadjustedVolume", "unadjusted_volume"),
        ("changeOverTime", "change_over_time"),
        ("  open  ", "open"),
    ],
)
def test_normalize_field(raw: str, expected: str) -> None:
    assert normalize_field(raw) == expected


def test_normalize_ohlcv_columns_flat() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3))
    df = pd.DataFrame(
        {"Open": [1.0, 2, 3], "Close": [1.5, 2.5, 3.5], "AdjClose": [1.4, 2.4, 3.4]},
        index=idx,
    )
    out = normalize_ohlcv_columns(df.copy())
    assert list(out.columns) == ["open", "close", "adj_close"]


def test_normalize_ohlcv_columns_multiindex() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3))
    df = pd.DataFrame(
        np.zeros((3, 4)),
        index=idx,
        columns=pd.MultiIndex.from_tuples(
            [("Open", "SPY"), ("Close", "SPY"), ("AdjClose", "SPY"), ("Volume", "SPY")]
        ),
    )
    out = normalize_ohlcv_columns(df.copy())
    assert list(out.columns) == [
        ("open", "SPY"),
        ("close", "SPY"),
        ("adj_close", "SPY"),
        ("volume", "SPY"),
    ]


def test_canonicalize_ohlcv_order_flat_puts_ohlcv_first() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2))
    df = pd.DataFrame(
        {"adj_close": [0, 0], "volume": [0, 0], "close": [0, 0],
         "low": [0, 0], "high": [0, 0], "open": [0, 0]},
        index=idx,
    )
    out = canonicalize_ohlcv_order(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "adj_close"]


def test_canonicalize_ohlcv_order_multiindex_puts_ohlcv_first() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2))
    df = pd.DataFrame(
        np.zeros((2, 6)),
        index=idx,
        columns=pd.MultiIndex.from_tuples([
            ("volume", "SPY"), ("close", "SPY"), ("low", "SPY"),
            ("high", "SPY"), ("open", "SPY"), ("adj_close", "SPY"),
        ]),
    )
    out = canonicalize_ohlcv_order(df)
    assert list(out.columns) == [
        ("open", "SPY"), ("high", "SPY"), ("low", "SPY"),
        ("close", "SPY"), ("volume", "SPY"), ("adj_close", "SPY"),
    ]


def test_canonicalize_ohlcv_order_preserves_symbol_axis() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2))
    df = pd.DataFrame(
        np.zeros((2, 4)),
        index=idx,
        columns=pd.MultiIndex.from_tuples([
            ("close", "SPY"), ("open", "SPY"),
            ("close", "AAPL"), ("open", "AAPL"),
        ]),
    )
    out = canonicalize_ohlcv_order(df)
    assert list(out.columns) == [
        ("open", "SPY"), ("open", "AAPL"),
        ("close", "SPY"), ("close", "AAPL"),
    ]


def test_ohlcv_columns_constant() -> None:
    assert OHLCV_COLUMNS == ("open", "high", "low", "close", "volume")


def test_normalize_handles_empty_frame() -> None:
    empty = pd.DataFrame()
    assert normalize_ohlcv_columns(empty).empty
    assert canonicalize_ohlcv_order(empty).empty
