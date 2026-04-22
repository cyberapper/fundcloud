"""Tests for TA-Lib-backed indicator wrappers and the custom registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.indicators import (
    GROUPS,
    IndicatorSpec,
    _talib_autogen,
    list_indicators,
    register_indicator,
    registered_indicators,
)

talib_available = _talib_autogen.TALIB_AVAILABLE
pytestmark = pytest.mark.skipif(not talib_available, reason="TA-Lib not installed")


# --------------------------------------------------------------------- fixtures


def _ohlcv_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=100, freq="D")
    close_a = 100 + np.cumsum(rng.normal(0, 1, 100))
    close_b = 200 + np.cumsum(rng.normal(0, 1, 100))
    df = pd.DataFrame(
        {
            ("open", "A"): close_a - 0.1,
            ("high", "A"): close_a + 0.5,
            ("low", "A"): close_a - 0.5,
            ("close", "A"): close_a,
            ("volume", "A"): 1_000_000.0,
            ("open", "B"): close_b - 0.1,
            ("high", "B"): close_b + 0.5,
            ("low", "B"): close_b - 0.5,
            ("close", "B"): close_b,
            ("volume", "B"): 1_000_000.0,
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)


# ----------------------------------------------------------------- generation


def test_autogen_covers_all_talib_groups() -> None:
    assert len(GROUPS) > 0
    total = sum(len(v) for v in GROUPS.values())
    assert total == len(_talib_autogen.GENERATED)


def test_autogen_creates_sma_and_rsi() -> None:
    from fundcloud.features.indicators import RSI, SMA

    assert SMA.talib_name == "SMA"
    assert RSI.talib_name == "RSI"
    assert "timeperiod" in SMA.default_params
    assert "timeperiod" in RSI.default_params


def test_autogen_multioutput_macd() -> None:
    from fundcloud.features.indicators import MACD

    assert MACD.outputs == ("macd", "macdsignal", "macdhist")


# ------------------------------------------------------------------- transform


def test_sma_transform_shape() -> None:
    from fundcloud.features.indicators import SMA

    panel = _ohlcv_panel()
    sma = SMA(timeperiod=10)
    out = sma.fit_transform(panel)
    assert list(out.columns) == ["A", "B"]
    # First (timeperiod - 1) rows are NaN.
    assert out.iloc[:9].isna().all().all()
    assert not out.iloc[10:].isna().all().all()


def test_multioutput_indicator_prefixes_columns() -> None:
    from fundcloud.features.indicators import BBANDS

    panel = _ohlcv_panel()
    out = BBANDS(timeperiod=20).fit_transform(panel)
    assert set(out.columns) == {
        "upperband__A",
        "middleband__A",
        "lowerband__A",
        "upperband__B",
        "middleband__B",
        "lowerband__B",
    }


def test_sklearn_get_set_params() -> None:
    from fundcloud.features.indicators import RSI

    rsi = RSI(timeperiod=7)
    assert rsi.get_params()["timeperiod"] == 7
    rsi.set_params(timeperiod=21)
    assert rsi.timeperiod == 21


# -------------------------------------------------------------------- registry


def test_register_custom_indicator() -> None:
    @register_indicator("my_test_ind")
    class _MyInd(IndicatorSpec):
        default_params = {"window": 5}

        def _compute(self, series_by_field, index):
            close = series_by_field["close"]
            out = close.rolling(self.window).mean()
            return pd.DataFrame({"value": out})

    assert "my_test_ind" in registered_indicators()
    assert "my_test_ind" in list_indicators()
