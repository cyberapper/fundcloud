"""Tests for fallback optimisers (``EqualWeighted``, ``InverseVolatility``, ``MVO``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.optimize.fallback_mvo import MVO, EqualWeighted, InverseVolatility
from fundcloud.portfolio import Portfolio
from sklearn.base import BaseEstimator


@pytest.fixture
def returns_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (252, 4)),
        index=idx,
        columns=["A", "B", "C", "D"],
    )


# ------------------------------------------------------------ sklearn contract


@pytest.mark.parametrize("cls", [EqualWeighted, InverseVolatility, MVO])
def test_is_sklearn_estimator(cls) -> None:
    instance = cls()
    assert isinstance(instance, BaseEstimator)


@pytest.mark.parametrize("cls", [EqualWeighted, InverseVolatility, MVO])
def test_fit_predict_returns_portfolio(cls, returns_panel: pd.DataFrame) -> None:
    est = cls()
    est.fit(returns_panel)
    assert est.weights_ is not None
    assert np.isclose(np.sum(est.weights_), 1.0, atol=1e-6)
    p = est.predict(returns_panel)
    assert isinstance(p, Portfolio)


# -------------------------------------------------------------- specific maths


def test_equal_weighted_exact(returns_panel: pd.DataFrame) -> None:
    w = EqualWeighted().fit(returns_panel).weights_
    assert np.allclose(w, 0.25)


def test_inverse_volatility_descending_order() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    vols = [0.01, 0.02, 0.03, 0.04]
    cols = list("ABCD")
    data = np.column_stack([rng.normal(0, v, 500) for v in vols])
    returns = pd.DataFrame(data, index=idx, columns=cols)

    w = InverseVolatility().fit(returns).weights_
    # Lowest-volatility asset gets the largest weight.
    assert w.argmax() == 0
    assert w.argmin() == 3


def test_mvo_bounds_long_only_and_sum_to_one(returns_panel: pd.DataFrame) -> None:
    w = MVO(l2=0.001).fit(returns_panel).weights_
    assert np.all(w >= -1e-12)
    assert np.isclose(w.sum(), 1.0)


def test_predict_column_order_alignment(returns_panel: pd.DataFrame) -> None:
    est = EqualWeighted().fit(returns_panel)
    # Shuffle columns for predict — the optimiser should reorder.
    shuffled = returns_panel[["D", "A", "C", "B"]]
    p = est.predict(shuffled)
    assert isinstance(p, Portfolio)


def test_predict_before_fit_raises(returns_panel: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        EqualWeighted().predict(returns_panel)


def test_score_is_finite_sharpe(returns_panel: pd.DataFrame) -> None:
    est = EqualWeighted().fit(returns_panel)
    assert np.isfinite(est.score(returns_panel))
