"""Tests for :class:`fundcloud.portfolio.Population`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.portfolio import Population, Portfolio


def _mk_portfolio(seed: int, name: str) -> Portfolio:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    r = pd.Series(rng.normal(0.0005, 0.01, 100), index=idx)
    w = pd.DataFrame({"A": [0.5] * 100, "B": [0.5] * 100}, index=idx)
    return Portfolio(returns=r, weights=w, name=name)


def test_summary_has_one_column_per_portfolio() -> None:
    pop = Population([_mk_portfolio(1, "alpha"), _mk_portfolio(2, "beta")])
    table = pop.summary()
    assert list(table.columns) == ["alpha", "beta"]
    assert "sharpe" in table.index


def test_cumulative_returns_frame() -> None:
    pop = Population([_mk_portfolio(1, "a"), _mk_portfolio(2, "b")])
    wealth = pop.cumulative_returns()
    assert set(wealth.columns) == {"a", "b"}
    assert wealth.iloc[0].min() > 0  # first cumulated period is ≈ 1 + r > 0


def test_composition_returns_latest_weights() -> None:
    pop = Population([_mk_portfolio(1, "p")])
    comp = pop.composition()
    assert set(comp.columns) == {"A", "B"}
    assert comp.iloc[0].sum() == 1.0


def test_names_collision_is_disambiguated() -> None:
    pop = Population([_mk_portfolio(1, "same"), _mk_portfolio(2, "same")])
    assert pop.names == ["same_1", "same_2"]


def test_indexing_by_int_and_name() -> None:
    a = _mk_portfolio(1, "a")
    b = _mk_portfolio(2, "b")
    pop = Population([a, b])
    assert pop[0] is a
    assert pop["b"] is b


def test_empty_population() -> None:
    pop = Population([])
    assert len(pop) == 0
    assert pop.summary().empty
    assert pop.cumulative_returns().empty
    assert pop.composition().empty
