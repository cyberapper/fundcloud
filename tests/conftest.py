"""Shared fixtures for the unit test suite."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def returns_series(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(datetime(2021, 1, 4), periods=252)
    return pd.Series(rng.normal(0.0005, 0.01, size=252), index=idx, name="strat")


@pytest.fixture
def returns_panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(datetime(2021, 1, 4), periods=252)
    cols = ["AAA", "BBB", "CCC"]
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(252, len(cols))),
        index=idx,
        columns=cols,
    )


@pytest.fixture
def ohlcv_panel(seed: int) -> pd.DataFrame:
    """Tiny two-asset OHLCV frame, MultiIndex on the columns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(datetime(2022, 1, 3), periods=50)
    data = {}
    for sym in ("AAA", "BBB"):
        close = 100 + np.cumsum(rng.normal(0, 1, size=len(idx)))
        data[("open", sym)] = close + rng.normal(0, 0.1, size=len(idx))
        data[("high", sym)] = close + np.abs(rng.normal(0, 0.5, size=len(idx)))
        data[("low", sym)] = close - np.abs(rng.normal(0, 0.5, size=len(idx)))
        data[("close", sym)] = close
        data[("volume", sym)] = rng.integers(1_000_000, 5_000_000, size=len(idx)).astype(float)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)
