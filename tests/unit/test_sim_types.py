"""Tests for ``fundcloud.sim`` primitives (``Order``, ``Trade``, ``Portfolio.apply``)."""

from __future__ import annotations

import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.sim import Order, Trade


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2024-01-02")


def test_order_requires_qty_or_notional() -> None:
    with pytest.raises(ValueError, match="qty or notional"):
        Order(ts=_ts(), asset="A", side="buy")


def test_order_rejects_zero_qty() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        Order(ts=_ts(), asset="A", side="buy", qty=0.0)


def test_order_limit_requires_limit_price() -> None:
    with pytest.raises(ValueError, match="limit"):
        Order(ts=_ts(), asset="A", side="buy", qty=1, kind="limit")


def test_order_with_qty_preserves_metadata() -> None:
    o = Order(ts=_ts(), asset="A", side="sell", notional=1000.0)
    resized = o.with_qty(5.0)
    assert resized.qty == 5.0
    assert resized.notional is None
    assert resized.side == "sell"
    assert resized.ts == o.ts


def test_order_signed_qty() -> None:
    buy = Order(ts=_ts(), asset="A", side="buy", qty=10.0)
    sell = Order(ts=_ts(), asset="A", side="sell", qty=10.0)
    assert buy.signed_qty() == 10.0
    assert sell.signed_qty() == -10.0


def test_trade_notional_matches_price_times_qty() -> None:
    o = Order(ts=_ts(), asset="A", side="buy", qty=1.0)
    t = Trade(order=o, ts=_ts(), asset="A", qty=3.0, price=25.0, fee=1.0)
    assert t.notional == pytest.approx(75.0)


def test_portfolio_apply_consumes_a_trade() -> None:
    p = Portfolio(cash=1_000.0)
    o = Order(ts=_ts(), asset="A", side="buy", qty=1.0)
    t = Trade(order=o, ts=_ts(), asset="A", qty=2.0, price=100.0, fee=1.0)
    p.apply(t)
    assert p.cash == pytest.approx(1000 - 200 - 1)
    assert p.position("A").qty == 2.0
