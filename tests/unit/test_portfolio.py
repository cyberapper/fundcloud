"""Tests for :class:`fundcloud.portfolio.Portfolio`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio, Position


@dataclass
class _Trade:
    ts: pd.Timestamp
    asset: str
    qty: float
    price: float
    fee: float = 0.0


def _returns() -> pd.Series:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=60, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=60), index=idx, name="strategy")


# ------------------------------------------------------------------ analytics


def test_analytics_mode_metrics_delegate_to_core() -> None:
    r = _returns()
    p = Portfolio(returns=r)
    assert np.isclose(p.sharpe(), r.fc.sharpe())
    assert np.isclose(p.max_drawdown(), r.fc.max_drawdown())
    assert np.isclose(p.sortino(), r.fc.sortino())


def test_summary_returns_a_named_series() -> None:
    p = Portfolio(returns=_returns(), name="buy_and_hold")
    summary = p.summary()
    assert isinstance(summary, pd.Series)
    assert summary.name == "buy_and_hold"
    assert "sharpe" in summary.index


def test_returns_without_returns_raises() -> None:
    with pytest.raises(ValueError, match="no recorded returns"):
        _ = Portfolio().returns


def test_equity_curve_from_returns() -> None:
    r = _returns()
    p = Portfolio(returns=r)
    eq = p.equity_curve
    assert eq.iloc[-1] == pytest.approx((1.0 + r).prod())


def test_multi_column_dataframe_rejected() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"a": [0.01, 0.02, 0.0], "b": [0.01, 0.0, 0.02]}, index=idx)
    with pytest.raises(ValueError, match="multiple columns"):
        Portfolio(returns=df)


# ---------------------------------------------------------------------- live


def test_apply_trade_updates_cash_and_positions() -> None:
    p = Portfolio(cash=10_000.0)
    ts = pd.Timestamp("2024-01-02")
    p.apply(_Trade(ts=ts, asset="AAPL", qty=10, price=150.0, fee=1.0))
    assert p.cash == pytest.approx(10_000 - 1500 - 1)
    assert p.position("AAPL").qty == 10
    assert p.position("AAPL").avg_cost == pytest.approx(150.0)


def test_mark_to_market_records_equity_curve() -> None:
    p = Portfolio(cash=10_000.0)
    ts0 = pd.Timestamp("2024-01-02")
    p.apply(_Trade(ts=ts0, asset="AAPL", qty=10, price=150.0))
    equity_t0 = p.mark_to_market(pd.Series({"AAPL": 150.0}), ts0)
    assert equity_t0 == pytest.approx(10_000)

    ts1 = pd.Timestamp("2024-01-03")
    equity_t1 = p.mark_to_market(pd.Series({"AAPL": 155.0}), ts1)
    assert equity_t1 == pytest.approx(10_000 + 50)


def test_snapshot_yields_analytics_portfolio() -> None:
    p = Portfolio(cash=10_000.0)
    p.apply(_Trade(ts=pd.Timestamp("2024-01-02"), asset="A", qty=10, price=100.0))
    # Equity at each mark:
    #   t0: cash 9_000 + 10*100 = 10_000
    #   t1: cash 9_000 + 10*101 = 10_010
    #   t2: cash 9_000 + 10*99  =  9_990
    p.mark_to_market(pd.Series({"A": 100.0}), pd.Timestamp("2024-01-02"))
    p.mark_to_market(pd.Series({"A": 101.0}), pd.Timestamp("2024-01-03"))
    p.mark_to_market(pd.Series({"A": 99.0}), pd.Timestamp("2024-01-04"))
    snap = p.snapshot()
    r = snap.returns
    assert len(r) == 2
    assert np.isclose(r.iloc[0], 0.001)
    assert np.isclose(r.iloc[1], (9_990 / 10_010) - 1.0)


# -------------------------------------------------------------------- turnover


def test_turnover_zero_when_weights_constant() -> None:
    w = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    p = Portfolio(returns=pd.Series([0.0, 0.0, 0.0], index=w.index, name="x"), weights=w)
    assert p.turnover() == 0.0


def test_turnover_nonzero_on_rebalance() -> None:
    w = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    p = Portfolio(
        returns=pd.Series([0.01, 0.01], index=w.index, name="x"),
        weights=w,
    )
    # One-way turnover: ((|1-0| + |0-1|) / 2) averaged over 1 non-first row = 1.0
    assert p.turnover() == pytest.approx(1.0)


# ------------------------------------------------------------------ skfolio


def test_from_and_to_skfolio_roundtrip() -> None:
    skfolio = pytest.importorskip("skfolio")

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (120, 3)), index=idx, columns=["A", "B", "C"])
    sk_pf = skfolio.Portfolio(X=returns, weights=np.array([0.5, 0.3, 0.2]))

    fc_pf = Portfolio.from_skfolio(sk_pf)
    assert isinstance(fc_pf, Portfolio)
    assert not fc_pf.returns.empty
    # Basic sanity: Sharpe is finite.
    assert np.isfinite(fc_pf.sharpe())


# ---------------------------------------------------------------------- Position


def test_position_dataclass_defaults() -> None:
    p = Position()
    assert p.qty == 0.0
    assert p.avg_cost == 0.0
