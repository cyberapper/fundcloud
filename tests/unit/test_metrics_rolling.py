"""Tests for :mod:`fundcloud.metrics.rolling` — focused on calendar alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.metrics import rolling_alpha, rolling_beta


def _btc_like(rng: np.random.Generator, n: int) -> pd.Series:
    """Crypto-style 7-day/week series."""
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.002, 0.04, n), index=idx, name="BTC-USD")


def _nq_like(rng: np.random.Generator, n: int) -> pd.Series:
    """Futures-style 5-day/week series."""
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(rng.normal(0.0006, 0.011, n), index=idx, name="NQ=F")


def test_rolling_beta_handles_mixed_calendars() -> None:
    rng = np.random.default_rng(11)
    r = _btc_like(rng, 1_200)
    b = _nq_like(rng, 1_200)
    out = rolling_beta(r, b, window=63)
    finite = out.dropna()
    # With strict inner-join every non-NaN window produces a real number.
    # Without the fix this was < 50 (or zero) due to NaN propagation.
    assert len(finite) >= 500
    assert np.isfinite(finite).all()


def test_rolling_alpha_handles_mixed_calendars() -> None:
    rng = np.random.default_rng(12)
    r = _btc_like(rng, 1_200)
    b = _nq_like(rng, 1_200)
    out = rolling_alpha(r, b, window=63)
    finite = out.dropna()
    assert len(finite) >= 500
    assert np.isfinite(finite).all()


def test_rolling_beta_dataframe_input() -> None:
    rng = np.random.default_rng(13)
    bench = _nq_like(rng, 900)
    idx = bench.index
    df = pd.DataFrame(
        {
            "spy": rng.normal(0.0005, 0.010, len(idx)),
            "qqq": rng.normal(0.0008, 0.014, len(idx)),
        },
        index=idx,
    )
    out = rolling_beta(df, bench, window=63)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"spy", "qqq"}
    assert out.dropna().shape[0] >= 100
