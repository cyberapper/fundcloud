"""Tests for the skfolio-backed optimiser adapters (skipped without skfolio)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skfolio")


@pytest.fixture
def returns_panel() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (252, 5)),
        index=idx,
        columns=list("ABCDE"),
    )


def test_mean_risk_variance_round_trip(returns_panel: pd.DataFrame) -> None:
    from fundcloud.optimize import MeanRisk, RiskMeasure
    from fundcloud.portfolio import Portfolio

    est = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    est.fit(returns_panel)
    assert est.weights_ is not None
    p = est.predict(returns_panel)
    assert isinstance(p, Portfolio)
    assert np.isfinite(p.sharpe())


def test_hrp_produces_portfolio(returns_panel: pd.DataFrame) -> None:
    from fundcloud.optimize import HierarchicalRiskParity
    from fundcloud.portfolio import Portfolio

    est = HierarchicalRiskParity()
    est.fit(returns_panel)
    p = est.predict(returns_panel)
    assert isinstance(p, Portfolio)


def test_equal_weighted_resolves_to_skfolio_adapter() -> None:
    # With skfolio installed, `fundcloud.optimize.EqualWeighted` must be the
    # adapter (not the fallback).
    import fundcloud.optimize as opt

    cls = opt.EqualWeighted
    # Adapter wraps a _skfolio_cls attribute.
    assert hasattr(cls, "_skfolio_cls"), "skfolio adapter not resolved when [pf] is installed"


def test_unknown_attribute_raises() -> None:
    import fundcloud.optimize as opt

    with pytest.raises(AttributeError):
        _ = opt.NotARealOptimizer
