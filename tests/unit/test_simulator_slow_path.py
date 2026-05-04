"""Slow-path Simulator coverage.

The Simulator has two execution paths:

* **Fast path** — ``_model_tags()`` recognises every model as a built-in
  concrete (``NoCost`` / ``FixedBps`` / ``PerShare`` / ``NoSlippage`` /
  ``HalfSpread`` / ``NextBarOpen`` / ``NextBarClose``) and dispatches to
  the Rust / NumPy panel kernel.
* **Slow path** — any custom subclass of ``CostModel`` /
  ``SlippageModel`` / ``ExecutionModel`` forces ``_model_tags()`` to
  return ``None``, falling back to the per-bar Python ``_drive()`` loop.

The fast path is well-covered by :mod:`test_simulator`. This file targets
the Python fallback so the per-bar bookkeeping (pending-order queue,
limit handling, last-bar fill rejection) gets exercised.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
from fundcloud.sim import Simulator
from fundcloud.sim.execution import NextBarOpen
from fundcloud.sim.orders import Order

# --------------------------------------------------------------------- helpers


class _CustomCost:
    """Non-recognised cost model → forces slow path."""

    def fee(self, *, price: float, qty: float) -> float:
        return abs(qty) * 0.01


class _CustomSlippage:
    """Non-recognised slippage model → forces slow path."""

    def apply(self, *, price: float, side: Literal["buy", "sell"]) -> tuple[float, float]:
        # 10 bps slip on either side.
        bump = price * 0.001
        return (price + bump, 10.0) if side == "buy" else (price - bump, 10.0)


class _CustomExecution:
    """Non-recognised execution model — forces the slow path.

    Returns ``signal_index + 1`` (next-bar fill at close), which honours
    the no-look-ahead invariant the simulator now enforces. Used by the
    slow-path tests to defeat the fast-path dispatcher's ``isinstance``
    check on the built-in :class:`NextBarOpen` / :class:`NextBarClose`.
    """

    def fill_at(self, *, signal_index: int, bars_index_size: int) -> int | None:
        nxt = signal_index + 1
        return nxt if nxt < bars_index_size else None

    def reference_price(self, *, bars: pd.DataFrame, fill_index: int, asset: str) -> float:
        col = ("close", asset)
        if col in bars.columns:
            return float(bars[col].iloc[fill_index])
        return float("nan")


class _LookAheadExecution:
    """Misimplemented model that fills on the same bar as the signal.

    The simulator must reject this with :class:`ValueError` because it
    re-introduces look-ahead bias.
    """

    def fill_at(self, *, signal_index: int, bars_index_size: int) -> int | None:
        return signal_index

    def reference_price(self, *, bars: pd.DataFrame, fill_index: int, asset: str) -> float:
        return float(bars[("close", asset)].iloc[fill_index])


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=20, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.5, 20))
    close_b = 50 + np.cumsum(rng.normal(0, 0.3, 20))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
        ("open", "B"): close_b,
        ("high", "B"): close_b + 1,
        ("low", "B"): close_b - 1,
        ("close", "B"): close_b,
        ("volume", "B"): 1_000_000.0,
    }
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _slow_sim(bars: pd.DataFrame, **kw: object) -> Simulator:
    """Build a simulator forced onto the Python slow path via a custom cost model."""
    return Simulator(bars, costs=_CustomCost(), cash=100_000, **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------- run_weights


def test_slow_path_run_weights_executes_all_rows(panel: pd.DataFrame) -> None:
    """Custom cost model forces the per-bar Python `_drive` loop for run_weights."""
    targets = pd.DataFrame({"A": 0.6, "B": 0.4}, index=panel.index[:5])
    result = _slow_sim(panel).run_weights(targets)
    assert len(result.equity_curve) == len(panel)
    # At least the first rebalance row generates orders for both assets.
    assert len(result.orders) > 0


def test_slow_path_run_weights_drops_empty_rows(panel: pd.DataFrame) -> None:
    """Rows full of NaN produce no orders (the `if not weights` early-out path)."""
    targets = pd.DataFrame(
        {"A": [np.nan, 0.5], "B": [np.nan, 0.5]},
        index=[panel.index[2], panel.index[5]],
    )
    result = _slow_sim(panel).run_weights(targets)
    # At least one rebalance row produced orders.
    assert len(result.orders) >= 1


# --------------------------------------------------------------------- run_signals


def test_slow_path_run_signals_buys_on_entry_sells_on_exit(panel: pd.DataFrame) -> None:
    entries = pd.DataFrame(False, index=panel.index, columns=["A", "B"])
    entries.iloc[3, 0] = True  # buy A on bar 3
    exits = pd.DataFrame(False, index=panel.index, columns=["A", "B"])
    exits.iloc[10, 0] = True  # sell A on bar 10
    result = _slow_sim(panel).run_signals(entries, exits, size=0.5)
    sides = set(result.trades["qty"].apply(lambda q: "buy" if q > 0 else "sell"))
    assert "buy" in sides
    assert "sell" in sides


def test_slow_path_run_signals_skips_zero_price() -> None:
    """If the current price is NaN/zero, no order is emitted."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=8, freq="B").values)
    cols = {
        ("open", "A"): [np.nan] * 8,
        ("high", "A"): [np.nan] * 8,
        ("low", "A"): [np.nan] * 8,
        ("close", "A"): [np.nan] * 8,
        ("volume", "A"): [0.0] * 8,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    entries = pd.DataFrame(True, index=idx, columns=["A"])
    exits = pd.DataFrame(False, index=idx, columns=["A"])
    result = _slow_sim(bars).run_signals(entries, exits, size=1.0)
    # No fills because there's no valid price to buy at.
    assert result.trades.empty


# --------------------------------------------------------------------- run_orders


def test_slow_path_run_orders_executes_market_orders(panel: pd.DataFrame) -> None:
    orders = pd.DataFrame({
        "ts": [panel.index[3]],
        "asset": ["A"],
        "side": ["buy"],
        "qty": [10.0],
    })
    result = _slow_sim(panel).run_orders(orders)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["asset"] == "A"


def test_slow_path_run_orders_drops_off_index_timestamps(panel: pd.DataFrame) -> None:
    """Orders whose ``ts`` doesn't match a bar are silently dropped."""
    orders = pd.DataFrame({
        "ts": [pd.Timestamp("2099-01-01")],
        "asset": ["A"],
        "side": ["buy"],
        "qty": [5.0],
    })
    result = _slow_sim(panel).run_orders(orders)
    assert result.trades.empty


def test_run_orders_rejects_missing_columns(panel: pd.DataFrame) -> None:
    bad = pd.DataFrame({"ts": [panel.index[0]], "asset": ["A"]})
    with pytest.raises(KeyError, match="missing columns"):
        _slow_sim(panel).run_orders(bad)


# --------------------------------------------------------------------- limit orders


def test_slow_path_limit_buy_skipped_when_above_limit(panel: pd.DataFrame) -> None:
    """A buy limit at $1 is far below market → never fills (line 471-472)."""
    sim = Simulator(panel, costs=_CustomCost(), execution=_CustomExecution(), cash=100_000)
    order = Order(ts=panel.index[3], asset="A", side="buy", qty=1.0, kind="limit", limit_price=1.0)
    fill = sim._execute(order, fill_idx=3)
    assert fill is None


def test_slow_path_limit_sell_skipped_when_below_limit(panel: pd.DataFrame) -> None:
    """A sell limit at $1000 is far above market → never fills (line 473-474)."""
    sim = Simulator(panel, costs=_CustomCost(), execution=_CustomExecution(), cash=100_000)
    order = Order(
        ts=panel.index[3], asset="A", side="sell", qty=1.0, kind="limit", limit_price=1000.0
    )
    fill = sim._execute(order, fill_idx=3)
    assert fill is None


def test_slow_path_limit_buy_fills_when_below_limit(panel: pd.DataFrame) -> None:
    """A buy limit far above market price fills at the market price."""
    sim = Simulator(panel, costs=_CustomCost(), execution=_CustomExecution(), cash=100_000)
    order = Order(
        ts=panel.index[3], asset="A", side="buy", qty=1.0, kind="limit", limit_price=10_000.0
    )
    fill = sim._execute(order, fill_idx=3)
    assert fill is not None
    assert fill.qty == 1.0


# --------------------------------------------------------------------- _execute edges


def test_execute_skips_when_reference_price_invalid() -> None:
    """If ref_price is NaN/zero, `_execute` returns None (line 456-457)."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=4, freq="B").values)
    cols = {
        ("open", "A"): [np.nan] * 4,
        ("high", "A"): [np.nan] * 4,
        ("low", "A"): [np.nan] * 4,
        ("close", "A"): [np.nan] * 4,
        ("volume", "A"): [0.0] * 4,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    sim = Simulator(bars, costs=_CustomCost(), execution=_CustomExecution(), cash=100_000)
    order = Order(ts=idx[1], asset="A", side="buy", qty=1.0)
    assert sim._execute(order, fill_idx=1) is None


def test_execute_resolves_qty_from_notional(panel: pd.DataFrame) -> None:
    """An order with `notional=` (no `qty=`) gets `qty = |notional| / ref_price`."""
    sim = Simulator(panel, costs=_CustomCost(), execution=_CustomExecution(), cash=100_000)
    order = Order(ts=panel.index[3], asset="A", side="buy", qty=None, notional=1000.0)
    fill = sim._execute(order, fill_idx=3)
    assert fill is not None
    assert fill.qty > 0


def test_order_rejects_negative_qty_at_construction(panel: pd.DataFrame) -> None:
    """``Order.qty`` is unsigned — direction comes from ``side``. Negative
    values are rejected at construction time so they can't silently flow
    into ``_execute`` and flip the trade's sign."""
    del panel  # only here for the parametrised fixture symmetry
    with pytest.raises(ValueError, match="qty must be"):
        Order(ts=pd.Timestamp("2024-01-04"), asset="A", side="buy", qty=-1.0)


# --------------------------------------------------------------------- pending queue


def test_drive_records_unfilled_order_on_last_bar(panel: pd.DataFrame) -> None:
    """NextBarOpen on the last bar can't fill — orders frame still has the row."""
    # Custom cost forces slow path; default execution = NextBarOpen → can't
    # schedule a fill from the last bar.
    last_ts = panel.index[-1]
    orders = pd.DataFrame({
        "ts": [last_ts],
        "asset": ["A"],
        "side": ["buy"],
        "qty": [1.0],
    })
    result = _slow_sim(panel, execution=NextBarOpen()).run_orders(orders)
    # Order recorded as unfilled.
    assert (result.orders["filled"] == False).any()  # noqa: E712
    # No trade emitted from that order.
    assert result.trades.empty


# --------------------------------------------------------------------- _resolve_bars


def test_simulator_accepts_backend_object(panel: pd.DataFrame) -> None:
    """Backend with a `.read()` method is accepted as the data argument."""

    class _MiniBackend:
        def read(self) -> pd.DataFrame:
            return panel

    sim = Simulator(_MiniBackend())  # type: ignore[arg-type]
    assert len(sim.bars) == len(panel)


def test_simulator_rejects_invalid_backend() -> None:
    """A backend whose `.read()` returns a non-DataFrame raises TypeError."""

    class _BrokenBackend:
        def read(self) -> object:
            return "not-a-frame"

    with pytest.raises(TypeError, match="must return a DataFrame"):
        Simulator(_BrokenBackend())  # type: ignore[arg-type]


def test_simulator_rejects_unknown_data_type() -> None:
    """An int isn't a DataFrame and has no `.read()` → TypeError."""
    with pytest.raises(TypeError, match="DataFrame or Backend"):
        Simulator(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------- _current_prices_map


def test_current_prices_map_handles_flat_columns(panel: pd.DataFrame) -> None:
    """`_current_prices_map` falls back to flat-column iteration for non-MultiIndex bars."""
    from fundcloud.sim.simulator import _current_prices_map
    from fundcloud.strategies.base import Context

    flat = pd.DataFrame(
        {"A": [100.0, 101.0], "B": [50.0, 50.5]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    ctx = Context(
        ts=flat.index[0],
        bar=flat.iloc[0],
        history=flat,
        portfolio=None,  # type: ignore[arg-type]
        assets=("A", "B"),
    )
    out = _current_prices_map(ctx)
    assert out == {"A": 100.0, "B": 50.0}


# --------------------------------------------------------------------- _model_tags fallback


def test_model_tags_returns_none_for_custom_slippage(panel: pd.DataFrame) -> None:
    """A custom slippage model also forces the slow path."""
    sim = Simulator(panel, slippage=_CustomSlippage(), cash=100_000)  # type: ignore[arg-type]
    targets = pd.DataFrame({"A": 1.0}, index=panel.index[:3])
    result = sim.run_weights(targets)
    assert result is not None


def test_model_tags_returns_none_for_custom_execution(panel: pd.DataFrame) -> None:
    """A custom execution model also forces the slow path."""
    sim = Simulator(panel, execution=_CustomExecution(), cash=100_000)
    targets = pd.DataFrame({"A": 1.0}, index=panel.index[:3])
    result = sim.run_weights(targets)
    assert result is not None


def test_drive_rejects_same_bar_custom_execution(panel: pd.DataFrame) -> None:
    """``ExecutionModel.fill_at`` must return a bar strictly later than the
    signal bar — same-bar fills introduce look-ahead bias, so the simulator
    raises ``ValueError`` rather than honouring them.
    """
    from fundcloud.strategies import Hold

    sim = Simulator(panel, execution=_LookAheadExecution(), cash=100_000)
    with pytest.raises(ValueError, match="look-ahead bias"):
        sim.run_strategy(Hold(weights={"A": 1.0}))
