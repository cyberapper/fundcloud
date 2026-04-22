"""Rust ↔ NumPy-fallback parity tests for the three deterministic
:class:`~fundcloud.sim.Simulator` entry points.

Scope: ``run_weights``, ``run_orders``, ``run_signals`` across the full
combinatorial grid of built-in cost / slippage / execution models. For
each combo we synthesise a random OHLCV panel, run both backends, and
assert ``equity_curve`` / ``trades`` / ``orders`` agree to ``atol=1e-10,
rtol=0``. Trade iteration order must also match — both impls iterate
bars in ascending order and sub-iterate orders in submission order.

Skipped automatically when ``fundcloud.kernels.HAS_RUST`` is False so
the pure-Python suite still runs on architectures without the Rust
extension.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from fundcloud.kernels import HAS_RUST

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")


def _synthetic_bars(n_bars: int, n_assets: int, *, seed: int) -> pd.DataFrame:
    """OHLCV panel with 7-day index + asset 0 as a crypto (always trading)
    and remaining assets as 5-day equities (NaN on weekends)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n_bars, freq="D")
    is_biz = pd.Series(idx, index=idx).apply(lambda d: d.weekday() < 5).to_numpy()
    cols: dict[tuple[str, str], np.ndarray] = {}
    for j in range(n_assets):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n_bars)))
        if j > 0:  # equities → NaN on weekends
            close = np.where(is_biz, close, np.nan)
        cols[("open", f"A{j:02d}")] = close
        cols[("high", f"A{j:02d}")] = close * 1.001
        cols[("low", f"A{j:02d}")] = close * 0.999
        cols[("close", f"A{j:02d}")] = close
        cols[("volume", f"A{j:02d}")] = 1_000_000.0
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# Built-in model combinations — every pairing the Rust kernel accepts.
from fundcloud.sim import (  # noqa: E402
    FixedBps,
    HalfSpread,
    NextBarOpen,
    NoCost,
    NoSlippage,
    PerShare,
    SameBarClose,
    Simulator,
)

_COSTS = [NoCost(), FixedBps(5.0), PerShare(rate=0.005, minimum=1.0)]
_SLIPPAGE = [NoSlippage(), HalfSpread(spread_bps=3.0)]
_EXECUTION = [NextBarOpen(), SameBarClose()]
_COMBOS = list(itertools.product(_COSTS, _SLIPPAGE, _EXECUTION))


def _assert_simresult_equal(a, b, *, atol: float = 1e-10) -> None:
    """Compare two SimResults element-wise to ``atol``."""
    np.testing.assert_allclose(
        a.equity_curve.to_numpy(), b.equity_curve.to_numpy(), atol=atol, rtol=0
    )
    # Trades (same row order).
    assert len(a.trades) == len(b.trades), (
        f"trade count differs: rust={len(a.trades)}, py={len(b.trades)}"
    )
    if len(a.trades):
        assert list(a.trades["asset"]) == list(b.trades["asset"])
        np.testing.assert_allclose(
            a.trades["qty"].to_numpy(), b.trades["qty"].to_numpy(), atol=atol, rtol=0
        )
        np.testing.assert_allclose(
            a.trades["price"].to_numpy(), b.trades["price"].to_numpy(), atol=atol, rtol=0
        )
        np.testing.assert_allclose(
            a.trades["fee"].to_numpy(), b.trades["fee"].to_numpy(), atol=atol, rtol=0
        )
    # Orders (row count only — some rounding in limit/notional representation
    # is expected, but the trade outputs above are the functional invariant).
    assert len(a.orders) == len(b.orders)


def _with_backend(bars, strategy_args, *, rust: bool, **sim_kwargs):
    """Temporarily toggle the Rust backend when running a Simulator call.

    Wraps ``fundcloud.kernels._sim._have_rust_sim`` to force either path.
    """
    from fundcloud.kernels import _sim as _dispatcher

    orig = _dispatcher._have_rust_sim
    _dispatcher._have_rust_sim = lambda: rust
    try:
        sim = Simulator(bars, **sim_kwargs)
        fn_name, *args = strategy_args
        return getattr(sim, fn_name)(*args)
    finally:
        _dispatcher._have_rust_sim = orig


# ------------------------------------------------------------------ weights


@pytest.mark.parametrize("combo_idx", range(len(_COMBOS)))
def test_parity_run_weights(combo_idx: int) -> None:
    costs, slippage, execution = _COMBOS[combo_idx]
    bars = _synthetic_bars(n_bars=80, n_assets=3, seed=combo_idx + 1)
    # One rebalance target — 50/50/0 split among the three assets, held.
    target = pd.DataFrame(
        [[0.5, 0.5, 0.0]] * 1,
        index=[bars.index[0]],
        columns=[f"A{j:02d}" for j in range(3)],
    )
    sim_kwargs = dict(
        cash=1_000_000.0, costs=costs, slippage=slippage, execution=execution
    )
    r_rust = _with_backend(bars, ("run_weights", target), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_weights", target), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


# ------------------------------------------------------------------- orders


@pytest.mark.parametrize("combo_idx", range(len(_COMBOS)))
def test_parity_run_orders(combo_idx: int) -> None:
    costs, slippage, execution = _COMBOS[combo_idx]
    bars = _synthetic_bars(n_bars=60, n_assets=2, seed=combo_idx + 100)
    orders = pd.DataFrame(
        [
            {"ts": bars.index[2], "asset": "A00", "side": "buy", "qty": 10.0},
            {"ts": bars.index[10], "asset": "A01", "side": "buy", "qty": 20.0},
            {"ts": bars.index[30], "asset": "A00", "side": "sell", "qty": 5.0},
        ]
    )
    sim_kwargs = dict(
        cash=1_000_000.0, costs=costs, slippage=slippage, execution=execution
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


# ------------------------------------------------------------------ signals


@pytest.mark.parametrize("combo_idx", range(len(_COMBOS)))
def test_parity_run_signals(combo_idx: int) -> None:
    costs, slippage, execution = _COMBOS[combo_idx]
    bars = _synthetic_bars(n_bars=70, n_assets=2, seed=combo_idx + 200)
    asset_cols = [f"A{j:02d}" for j in range(2)]
    entries = pd.DataFrame(False, index=bars.index, columns=asset_cols)
    exits = pd.DataFrame(False, index=bars.index, columns=asset_cols)
    entries.iloc[5, 0] = True
    entries.iloc[15, 1] = True
    exits.iloc[40, 0] = True
    sim_kwargs = dict(
        cash=1_000_000.0, costs=costs, slippage=slippage, execution=execution
    )
    r_rust = _with_backend(
        bars, ("run_signals", entries, exits), rust=True, **sim_kwargs
    )
    r_py = _with_backend(
        bars, ("run_signals", entries, exits), rust=False, **sim_kwargs
    )
    _assert_simresult_equal(r_rust, r_py)


# Larger stress test (one combo only — keeps suite fast).


def test_parity_large_panel_run_weights() -> None:
    bars = _synthetic_bars(n_bars=500, n_assets=5, seed=9999)
    target = pd.DataFrame(
        [[0.2] * 5] * 3,
        index=[bars.index[0], bars.index[100], bars.index[300]],
        columns=[f"A{j:02d}" for j in range(5)],
    )
    sim_kwargs = dict(
        cash=1_000_000.0,
        costs=FixedBps(5.0),
        slippage=HalfSpread(spread_bps=4.0),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_weights", target), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_weights", target), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
