"""Tests for ``fundcloud.sim`` primitives (``Order``, ``Trade``, ``Portfolio.apply``)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.sim import Order, Trade


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2024-01-02")


def test_order_requires_qty_or_notional() -> None:
    with pytest.raises(ValueError, match="qty or notional"):
        Order(ts=_ts(), asset="A", side="buy")


def test_order_rejects_both_qty_and_notional() -> None:
    """Mutually exclusive — downstream ``_execute`` uses ``qty`` and
    silently ignores ``notional`` if both are passed, which gives bad
    callers the wrong sizing semantics with no error."""
    with pytest.raises(ValueError, match="exactly one"):
        Order(ts=_ts(), asset="A", side="buy", qty=10.0, notional=1000.0)


def test_order_rejects_zero_qty() -> None:
    with pytest.raises(ValueError, match="qty must be"):
        Order(ts=_ts(), asset="A", side="buy", qty=0.0)


def test_order_rejects_negative_qty() -> None:
    """Direction comes from `side` — a negative qty would silently flip
    the trade's sign at fill time, so the constructor must reject it."""
    with pytest.raises(ValueError, match="qty must be"):
        Order(ts=_ts(), asset="A", side="buy", qty=-10.0)


def test_order_rejects_negative_notional() -> None:
    with pytest.raises(ValueError, match="notional must be"):
        Order(ts=_ts(), asset="A", side="buy", notional=-1000.0)


def test_order_rejects_zero_notional() -> None:
    with pytest.raises(ValueError, match="notional must be"):
        Order(ts=_ts(), asset="A", side="buy", notional=0.0)


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


# ----------------------------------------------------------------- bracket-order validation


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_order_rejects_non_positive_tp_stop(bad: float) -> None:
    with pytest.raises(ValueError, match="tp_stop"):
        Order(ts=_ts(), asset="A", side="buy", qty=1.0, tp_stop=bad)


def test_order_accepts_tp_stop_above_one() -> None:
    """No upper bound — values >= 1 are valid (long TP at +200% = price triples)."""
    o = Order(ts=_ts(), asset="A", side="buy", qty=1.0, tp_stop=2.0)
    assert o.tp_stop == 2.0


def test_order_accepts_bracket_with_both_stops() -> None:
    o = Order(ts=_ts(), asset="A", side="buy", qty=1.0, sl_stop=0.10, tp_stop=0.20)
    assert o.sl_stop == 0.10
    assert o.tp_stop == 0.20


def test_trade_reason_defaults_to_signal() -> None:
    o = Order(ts=_ts(), asset="A", side="buy", qty=1.0)
    t = Trade(order=o, ts=_ts(), asset="A", qty=1.0, price=100.0)
    assert t.reason == "signal"


def test_trade_accepts_take_profit_reason() -> None:
    o = Order(ts=_ts(), asset="A", side="buy", qty=1.0)
    t = Trade(order=o, ts=_ts(), asset="A", qty=-1.0, price=110.0, reason="take_profit")
    assert t.reason == "take_profit"


# ----------------------------------------------------------------- runtime side validation


@pytest.mark.parametrize("bad_side", ["BUY", "long", "", "short", "Sell"])
def test_order_rejects_invalid_side(bad_side: str) -> None:
    """``Literal["buy", "sell"]`` is static-only; runtime values otherwise
    silently default to sell-semantics in :meth:`Order.signed_qty`."""
    with pytest.raises(ValueError, match="side"):
        Order(ts=_ts(), asset="A", side=bad_side, qty=1.0)  # type: ignore[arg-type]


# -------------------------------------------------------- non-finite numeric validation


@pytest.mark.parametrize(
    "field",
    ["qty", "notional", "sl_stop", "tp_stop", "tsl_stop"],
)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_order_rejects_non_finite_numeric_inputs(field: str, bad: float) -> None:
    """NaN/Inf slip past simple ``<= 0`` and ``< 1`` comparisons and would
    poison downstream cash math; reject them at construction."""
    kwargs: dict[str, object] = {"ts": _ts(), "asset": "A", "side": "buy", "qty": 1.0}
    kwargs[field] = bad
    with pytest.raises(ValueError):
        Order(**kwargs)  # type: ignore[arg-type]
