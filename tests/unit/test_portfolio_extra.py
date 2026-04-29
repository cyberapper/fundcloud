"""Coverage for delegating-analytics paths and from-NAV / Population.

The base :mod:`test_portfolio` covers the core invariants — apply / mark /
snapshot, summary shape, multi-column rejection. This file fills in the
delegating analytics methods, the ``from_nav`` classmethod, the empty /
edge-case branches in ``worst_drawdowns`` / ``worst_runups`` /
``attribution``, and the :class:`Population` container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio, Position
from fundcloud.portfolio.population import Population


@dataclass
class _Trade:
    ts: pd.Timestamp
    asset: str
    qty: float
    price: float
    fee: float = 0.0


@pytest.fixture
def returns_long() -> pd.Series:
    """Long enough to populate yearly bins + drawdown / runup episodes."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=800, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, 800), index=idx, name="long")


@pytest.fixture
def benchmark_long() -> pd.Series:
    rng = np.random.default_rng(12)
    idx = pd.date_range("2020-01-01", periods=800, freq="B")
    return pd.Series(rng.normal(0.0003, 0.008, 800), index=idx, name="bench")


# --------------------------------------------------------------------- analytics


def test_all_delegating_metrics(returns_long: pd.Series, benchmark_long: pd.Series) -> None:
    p = Portfolio(returns=returns_long, benchmark=benchmark_long, name="x")
    assert isinstance(p.calmar(), float)
    assert isinstance(p.omega(), float)
    assert isinstance(p.ulcer_index(), float)
    assert isinstance(p.cvar(), float)
    assert isinstance(p.value_at_risk(), float)
    assert isinstance(p.drawdown_series(), pd.Series)
    assert isinstance(p.drawdown_details(), pd.DataFrame)
    assert isinstance(p.runup_details(), pd.DataFrame)


def test_metrics_uses_constructor_benchmark(
    returns_long: pd.Series, benchmark_long: pd.Series
) -> None:
    """`metrics()` defaults to the constructor benchmark when none is passed."""
    p = Portfolio(returns=returns_long, benchmark=benchmark_long, name="x")
    bundle = p.metrics()
    # Benchmark-relative metrics are present.
    assert "alpha" in bundle.index or "beta" in bundle.index


def test_metrics_explicit_benchmark_wins(
    returns_long: pd.Series, benchmark_long: pd.Series
) -> None:
    p = Portfolio(returns=returns_long, benchmark=benchmark_long, name="x")
    rng = np.random.default_rng(13)
    other = pd.Series(
        rng.normal(0.0, 0.005, len(returns_long)),
        index=returns_long.index,
        name="other_bench",
    )
    bundle = p.metrics(benchmark=other)
    assert isinstance(bundle, pd.Series)


def test_period_returns_uses_constructor_benchmark(
    returns_long: pd.Series, benchmark_long: pd.Series
) -> None:
    p = Portfolio(returns=returns_long, benchmark=benchmark_long)
    out = p.period_returns()
    assert isinstance(out, (pd.Series, pd.DataFrame))


def test_yearly_returns_no_benchmark(returns_long: pd.Series) -> None:
    p = Portfolio(returns=returns_long, name="x")
    yr = p.yearly_returns()
    assert isinstance(yr, pd.Series)


def test_yearly_returns_with_constructor_benchmark(
    returns_long: pd.Series, benchmark_long: pd.Series
) -> None:
    p = Portfolio(returns=returns_long, benchmark=benchmark_long, name="x")
    yr = p.yearly_returns()
    assert isinstance(yr, pd.DataFrame)
    assert yr.shape[1] == 2


def test_yearly_returns_with_unnamed_benchmark(returns_long: pd.Series) -> None:
    """An unnamed benchmark falls back to the literal "benchmark" column name."""
    bench = pd.Series(np.zeros(len(returns_long)), index=returns_long.index)
    p = Portfolio(returns=returns_long, benchmark=bench, name="x")
    yr = p.yearly_returns()
    assert "benchmark" in yr.columns


# --------------------------------------------------------------------- worst_*


def test_worst_drawdowns_returns_top_n(returns_long: pd.Series) -> None:
    p = Portfolio(returns=returns_long, name="x")
    top = p.worst_drawdowns(top=3)
    assert isinstance(top, pd.DataFrame)
    assert set(top.columns) == {"Started", "Recovered", "Drawdown", "Days"}
    assert len(top) <= 3


def test_worst_drawdowns_empty_returns_empty_frame() -> None:
    """Empty returns yields the canonical empty-schema frame."""
    p = Portfolio(returns=pd.Series([], dtype=float, name="x"), name="x")
    top = p.worst_drawdowns()
    assert top.empty
    assert list(top.columns) == ["Started", "Recovered", "Drawdown", "Days"]


def test_worst_runups_returns_top_n(returns_long: pd.Series) -> None:
    p = Portfolio(returns=returns_long, name="x")
    top = p.worst_runups(top=3)
    assert isinstance(top, pd.DataFrame)
    assert set(top.columns) == {"Started", "Peaked", "Runup", "Days"}
    assert len(top) <= 3


def test_worst_runups_empty_returns_empty_frame() -> None:
    p = Portfolio(returns=pd.Series([], dtype=float, name="x"), name="x")
    top = p.worst_runups()
    assert top.empty
    assert list(top.columns) == ["Started", "Peaked", "Runup", "Days"]


# --------------------------------------------------------------------- attribution


def test_attribution_returns_per_asset_contribution() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    returns = pd.Series([0.01, 0.0, -0.005], index=idx, name="x")
    p = Portfolio(returns=returns, weights=weights, name="x")
    attr = p.attribution()
    assert list(attr.columns) == ["A", "B"]
    assert len(attr) == 3


def test_attribution_empty_when_no_weights() -> None:
    p = Portfolio(returns=pd.Series([0.01, 0.02], name="x"), name="x")
    assert p.attribution().empty


def test_attribution_empty_returns_returns_empty_columns() -> None:
    """Weights present but returns empty → a frame with the right columns and no rows."""
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=pd.bdate_range("2024-01-01", periods=1))
    p = Portfolio(returns=pd.Series([], dtype=float, name="x"), weights=weights, name="x")
    attr = p.attribution()
    assert list(attr.columns) == ["A", "B"]
    assert len(attr) == 0


def test_contribution_with_data() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    returns = pd.Series([0.01, 0.0, -0.005], index=idx, name="x")
    p = Portfolio(returns=returns, weights=weights, name="x")
    contrib = p.contribution()
    assert isinstance(contrib, pd.Series)
    assert set(contrib.index) == {"A", "B"}


def test_contribution_empty_when_no_weights() -> None:
    p = Portfolio(returns=pd.Series([0.01, 0.02], name="x"), name="x")
    assert p.contribution().empty


# --------------------------------------------------------------------- live state


def test_positions_property_reports_qty() -> None:
    p = Portfolio(cash=10_000, positions={"AAPL": 5.0, "MSFT": 2.0})
    pos = p.positions
    assert isinstance(pos, pd.Series)
    assert pos["AAPL"] == 5.0
    assert pos["MSFT"] == 2.0


def test_rename_propagates_to_returns_series() -> None:
    p = Portfolio(returns=pd.Series([0.01, 0.02], name="orig"), name="orig")
    p.rename("renamed")
    assert p.name == "renamed"
    assert p.returns.name == "renamed"


def test_default_name_when_unset() -> None:
    """Constructor with no `name=` and no Series name returns "strategy"."""
    p = Portfolio()
    assert p.name == "strategy"


def test_mark_to_market_with_nan_price_uses_last_known() -> None:
    """A NaN price on a non-trading bar falls back to the cached last price."""
    p = Portfolio(cash=10_000)
    ts0 = pd.Timestamp("2024-01-02")
    p.apply(_Trade(ts=ts0, asset="A", qty=10, price=100.0))
    p.mark_to_market(pd.Series({"A": 100.0}), ts0)

    ts1 = pd.Timestamp("2024-01-03")
    equity = p.mark_to_market(pd.Series({"A": np.nan}), ts1)
    # Last known price was 100, so equity = 9000 cash + 10*100 = 10_000.
    assert equity == pytest.approx(10_000)


def test_mark_to_market_with_zero_qty_position_skipped() -> None:
    """Positions with qty=0 are skipped in the mark-to-market loop."""
    p = Portfolio(cash=10_000, positions={"A": 0.0})
    equity = p.mark_to_market(pd.Series({"A": 50.0}), pd.Timestamp("2024-01-02"))
    assert equity == pytest.approx(10_000)


def test_apply_position_close_keeps_avg_cost() -> None:
    """Closing a long with a sell trade leaves avg_cost unchanged."""
    p = Portfolio(cash=10_000)
    ts = pd.Timestamp("2024-01-02")
    p.apply(_Trade(ts=ts, asset="A", qty=10, price=100.0))  # long 10 @ 100
    p.apply(_Trade(ts=ts, asset="A", qty=-5, price=110.0))  # sell 5 @ 110
    pos = p.position("A")
    # avg_cost is unchanged on a partial close (the spec).
    assert pos.avg_cost == pytest.approx(100.0)
    assert pos.qty == 5.0


def test_equity_curve_from_live_state() -> None:
    """When live equity history exists, equity_curve uses it instead of returns."""
    p = Portfolio(cash=10_000)
    ts0 = pd.Timestamp("2024-01-02")
    p.apply(_Trade(ts=ts0, asset="A", qty=10, price=100.0))
    p.mark_to_market(pd.Series({"A": 100.0}), ts0)
    eq = p.equity_curve
    assert len(eq) == 1
    assert eq.iloc[0] == pytest.approx(10_000)


def test_equity_curve_empty_when_no_returns_no_history() -> None:
    p = Portfolio()
    assert p.equity_curve.empty


def test_snapshot_with_no_history_yields_empty_returns() -> None:
    """Snapshot with no equity history produces a Portfolio with an empty
    returns Series — not a missing-returns error."""
    p = Portfolio(cash=10_000)
    snap = p.snapshot()
    assert snap.returns.empty


# --------------------------------------------------------------------- from_nav


def test_from_nav_with_series() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    nav = pd.Series(np.linspace(100, 110, 10), index=idx)
    p = Portfolio.from_nav(nav, name="fund")
    assert p.name == "fund"
    assert isinstance(p.returns, pd.Series)
    assert len(p.returns) == 9  # first NaN dropped


def test_from_nav_with_dataframe_nav_column() -> None:
    """A DataFrame with a 'nav' column is coerced to that column."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    df = pd.DataFrame({"nav": np.linspace(100, 110, 10), "shares": 1.0}, index=idx)
    p = Portfolio.from_nav(df)
    assert isinstance(p.returns, pd.Series)


def test_from_nav_with_single_column_dataframe() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    df = pd.DataFrame({"value": np.linspace(100, 110, 10)}, index=idx)
    p = Portfolio.from_nav(df)
    assert isinstance(p.returns, pd.Series)


def test_from_nav_rejects_multi_column_dataframe_without_nav_col() -> None:
    """A multi-column frame without a 'nav' column raises."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    df = pd.DataFrame({"a": 1.0, "b": 2.0}, index=idx)
    with pytest.raises(ValueError, match="'nav' column"):
        Portfolio.from_nav(df)


def test_from_nav_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="expected Series or DataFrame"):
        Portfolio.from_nav([1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_from_nav_with_distributions_and_method_total_return() -> None:
    """`distributions=` is forwarded to returns_from_nav for total-return mode."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    nav = pd.Series(np.linspace(100, 110, 10), index=idx)
    dist = pd.Series(0.1, index=idx)
    p = Portfolio.from_nav(nav, distributions=dist, method="total_return")
    assert isinstance(p.returns, pd.Series)


def test_from_nav_stashes_source_trades_and_positions() -> None:
    """`trades` / `positions` kwargs land on the Portfolio for inspection."""
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    nav = pd.Series(np.linspace(100, 110, 5), index=idx)
    trades = pd.DataFrame({"ts": idx[:1], "asset": ["A"], "qty": [1.0], "price": [100.0]})
    positions = pd.DataFrame({"qty": [1.0]}, index=["A"])
    p = Portfolio.from_nav(nav, trades=trades, positions=positions)
    assert p._source_trades is trades
    assert p._source_positions is positions


# --------------------------------------------------------------------- skfolio


def test_to_skfolio_requires_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """`to_skfolio()` raises a clear ImportError when skfolio is missing."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "skfolio" or name.startswith("skfolio."):
            raise ImportError("skfolio not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = Portfolio(returns=pd.Series([0.01, 0.02], name="x"), name="x")
    with pytest.raises(ImportError, match="skfolio"):
        p.to_skfolio()


def test_from_skfolio_with_numpy_returns_and_weights() -> None:
    """`from_skfolio` lifts a duck-typed object with numpy arrays."""

    class _FakeSkPortfolio:
        def __init__(self) -> None:
            idx = pd.date_range("2024-01-02", periods=5, freq="B")
            self.returns = np.array([0.01, -0.005, 0.008, 0.0, -0.002])
            self.weights = np.array([0.5, 0.5])
            self.assets = ["A", "B"]
            self.observations = idx
            self.name = "skfake"

    fc_pf = Portfolio.from_skfolio(_FakeSkPortfolio())
    assert fc_pf.name == "skfake"
    assert len(fc_pf.returns) == 5
    assert isinstance(fc_pf.weights, pd.DataFrame)


def test_from_skfolio_2d_weights_per_period() -> None:
    """A 2-D weights array becomes a per-period weights frame."""

    class _FakeSkPortfolio:
        def __init__(self) -> None:
            idx = pd.date_range("2024-01-02", periods=3, freq="B")
            self.returns = pd.Series([0.01, 0.0, -0.005], index=idx)
            self.weights = np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
            self.assets = ["A", "B"]
            self.observations = idx

    fc_pf = Portfolio.from_skfolio(_FakeSkPortfolio())
    assert fc_pf.weights is not None
    assert fc_pf.weights.shape == (3, 2)


def test_from_skfolio_without_weights() -> None:
    """`weights=None` on the source produces a weight-less Portfolio."""

    class _FakeSkPortfolio:
        def __init__(self) -> None:
            self.returns = pd.Series([0.01, 0.02])
            self.weights = None
            self.assets = ["A"]

    fc_pf = Portfolio.from_skfolio(_FakeSkPortfolio())
    assert fc_pf.weights is None


def test_from_skfolio_object_without_returns_attr_raises() -> None:
    """Object without `.returns` raises AttributeError, not a silent NaN frame."""

    class _Broken:
        pass

    with pytest.raises(AttributeError, match="returns"):
        Portfolio.from_skfolio(_Broken())


# --------------------------------------------------------------------- helpers


def test_position_dataclass_fields() -> None:
    p = Position(qty=5.0, avg_cost=99.0)
    assert p.qty == 5.0
    assert p.avg_cost == 99.0


def test_coerce_returns_one_column_dataframe_passes() -> None:
    """A 1-column DataFrame is silently coerced to its column."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"a": [0.01, 0.02, 0.0]}, index=idx)
    p = Portfolio(returns=df)
    assert isinstance(p.returns, pd.Series)


def test_coerce_weights_series_becomes_one_row_frame() -> None:
    """A Series of weights is broadcast to a 1-row frame."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    p = Portfolio(returns=pd.Series([0.01, 0.02], name="x"), weights=weights, name="x")
    assert isinstance(p.weights, pd.DataFrame)
    assert p.weights.shape[0] == 1


# --------------------------------------------------------------------- Population


def test_population_basic_iteration() -> None:
    a = Portfolio(returns=pd.Series([0.01, 0.02], name="A"), name="A")
    b = Portfolio(returns=pd.Series([0.0, 0.01], name="B"), name="B")
    pop = Population([a, b])
    assert len(pop) == 2
    assert list(pop.names) == ["A", "B"]
    assert pop[0] is a
    assert pop["B"] is b
    assert list(iter(pop)) == [a, b]


def test_population_keyerror_on_unknown_name() -> None:
    pop = Population([Portfolio(returns=pd.Series([0.01], name="A"), name="A")])
    with pytest.raises(KeyError):
        _ = pop["missing"]


def test_population_disambiguates_duplicate_names() -> None:
    a = Portfolio(returns=pd.Series([0.01, 0.02], name="dup"), name="dup")
    b = Portfolio(returns=pd.Series([0.0, 0.01], name="dup"), name="dup")
    pop = Population([a, b])
    # Both got renamed to dup_1 / dup_2.
    assert sorted(pop.names) == ["dup_1", "dup_2"]


def test_population_summary_concats_per_portfolio() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    a = Portfolio(returns=pd.Series(rng.normal(0, 0.01, 60), index=idx, name="A"), name="A")
    b = Portfolio(returns=pd.Series(rng.normal(0, 0.01, 60), index=idx, name="B"), name="B")
    pop = Population([a, b])
    summary = pop.summary()
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {"A", "B"}


def test_population_summary_empty() -> None:
    pop = Population([])
    assert pop.summary().empty


def test_population_cumulative_returns() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    a = Portfolio(returns=pd.Series([0.01] * 5, index=idx, name="A"), name="A")
    b = Portfolio(returns=pd.Series([0.02] * 5, index=idx, name="B"), name="B")
    pop = Population([a, b])
    cum = pop.cumulative_returns()
    assert set(cum.columns) == {"A", "B"}
    # Each column ends at (1 + r)^5 - cumprod is the cumulative product.
    assert cum["A"].iloc[-1] == pytest.approx(1.01**5)


def test_population_cumulative_returns_skips_returnsless_portfolio() -> None:
    """Live-only portfolios (no returns) are silently skipped, not raised on."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    a = Portfolio(returns=pd.Series([0.01, 0.02, 0.0], index=idx, name="A"), name="A")
    live = Portfolio(cash=10_000, name="live")  # no returns
    pop = Population([a, live])
    cum = pop.cumulative_returns()
    assert "A" in cum.columns
    assert "live" not in cum.columns


def test_population_cumulative_returns_empty() -> None:
    assert Population([]).cumulative_returns().empty


def test_population_composition_uses_latest_weights() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    weights = pd.DataFrame({"A": [0.4, 0.6], "B": [0.6, 0.4]}, index=idx)
    p = Portfolio(returns=pd.Series([0.0, 0.0], index=idx, name="x"), weights=weights, name="x")
    pop = Population([p])
    comp = pop.composition()
    assert comp.loc["x", "A"] == pytest.approx(0.6)
    assert comp.loc["x", "B"] == pytest.approx(0.4)


def test_population_composition_skips_weightless_portfolios() -> None:
    a = Portfolio(returns=pd.Series([0.01, 0.02], name="A"), name="A")  # no weights
    pop = Population([a])
    assert pop.composition().empty
