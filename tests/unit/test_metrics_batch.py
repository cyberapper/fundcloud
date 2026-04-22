"""Tests for the batch metrics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.metrics import (
    batch_cvar,
    batch_max_drawdown,
    batch_sharpe,
    batch_sortino,
    batch_summary,
)


@pytest.fixture
def strategies() -> dict[str, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=250, freq="B")
    return {
        "slow": pd.Series(rng.normal(0.0003, 0.008, 250), index=idx),
        "fast": pd.Series(rng.normal(0.0008, 0.015, 250), index=idx),
    }


def test_batch_sharpe(strategies: dict[str, pd.Series]) -> None:
    out = batch_sharpe(strategies)
    assert set(out.index) == {"slow", "fast"}
    assert out.dtype == float


def test_batch_sortino_max_drawdown_cvar(strategies: dict[str, pd.Series]) -> None:
    assert set(batch_sortino(strategies).index) == {"slow", "fast"}
    assert set(batch_max_drawdown(strategies).index) == {"slow", "fast"}
    assert set(batch_cvar(strategies).index) == {"slow", "fast"}


def test_batch_summary_shape(strategies: dict[str, pd.Series]) -> None:
    table = batch_summary(strategies)
    assert set(table.index) == {"slow", "fast"}
    for col in ("sharpe", "sortino", "max_drawdown", "cvar"):
        assert col in table.columns


def test_batch_summary_empty() -> None:
    assert batch_summary({}).empty


def test_reduce_returns_accepts_panel() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-01", periods=50, freq="B")
    panel = pd.DataFrame(rng.normal(0, 0.01, (50, 3)), index=idx, columns=list("ABC"))
    out = batch_sharpe({"avg": panel})
    assert "avg" in out.index
    assert np.isfinite(out["avg"]) or np.isnan(out["avg"])
