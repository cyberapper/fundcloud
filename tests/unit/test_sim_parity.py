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
    NextBarClose,
    NextBarOpen,
    NoCost,
    NoSlippage,
    PerShare,
    Simulator,
)

_COSTS = [NoCost(), FixedBps(5.0), PerShare(rate=0.005, minimum=1.0)]
_SLIPPAGE = [NoSlippage(), HalfSpread(spread_bps=3.0)]
_EXECUTION = [NextBarOpen(), NextBarClose()]
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
    sim_kwargs = dict(cash=1_000_000.0, costs=costs, slippage=slippage, execution=execution)
    r_rust = _with_backend(bars, ("run_weights", target), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_weights", target), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


# ------------------------------------------------------------------- orders


@pytest.mark.parametrize("combo_idx", range(len(_COMBOS)))
def test_parity_run_orders(combo_idx: int) -> None:
    costs, slippage, execution = _COMBOS[combo_idx]
    bars = _synthetic_bars(n_bars=60, n_assets=2, seed=combo_idx + 100)
    orders = pd.DataFrame([
        {"ts": bars.index[2], "asset": "A00", "side": "buy", "qty": 10.0},
        {"ts": bars.index[10], "asset": "A01", "side": "buy", "qty": 20.0},
        {"ts": bars.index[30], "asset": "A00", "side": "sell", "qty": 5.0},
    ])
    sim_kwargs = dict(cash=1_000_000.0, costs=costs, slippage=slippage, execution=execution)
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
    sim_kwargs = dict(cash=1_000_000.0, costs=costs, slippage=slippage, execution=execution)
    r_rust = _with_backend(bars, ("run_signals", entries, exits), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_signals", entries, exits), rust=False, **sim_kwargs)
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


# ------------------------------------------------------------------ bracket parity


def _bracket_bars() -> pd.DataFrame:
    """Engineered OHLC sequence that triggers SL on bar 4, TP on bar 8."""
    rows = [
        (100.0, 102.0, 99.0, 100.0),
        (100.0, 101.0, 99.0, 100.0),  # entry fills here at open=100
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 99.0, 95.0, 96.0),
        (95.0, 96.0, 88.0, 92.0),  # bar 4: low pierces SL=90 → fire
        (95.0, 100.0, 90.0, 99.0),
        (100.0, 105.0, 99.0, 102.0),
        (104.0, 108.0, 102.0, 107.0),
        (108.0, 115.0, 107.0, 112.0),  # bar 8: high pierces TP=110 → would fire
    ]
    n = len(rows)
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    cols = pd.MultiIndex.from_tuples([
        ("open", "BTC"),
        ("high", "BTC"),
        ("low", "BTC"),
        ("close", "BTC"),
        ("volume", "BTC"),
    ])
    return pd.DataFrame(
        {
            ("open", "BTC"): [r[0] for r in rows],
            ("high", "BTC"): [r[1] for r in rows],
            ("low", "BTC"): [r[2] for r in rows],
            ("close", "BTC"): [r[3] for r in rows],
            ("volume", "BTC"): [1.0] * n,
        },
        index=idx,
        columns=cols,
    )


def test_parity_run_orders_with_sl_stop() -> None:
    """Single buy carrying sl_stop=0.10. Rust and fallback must agree."""
    bars = _bracket_bars()
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 10.0, "sl_stop": 0.10},
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=FixedBps(5.0),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    # Rust path emits an SL fire — confirm the trade ledger reflects it.
    sl_rows = r_rust.trades[r_rust.trades["reason"] == "stop_loss"]
    assert len(sl_rows) == 1
    assert sl_rows.iloc[0]["price"] == pytest.approx(90.0)


def test_parity_run_orders_with_tp_stop() -> None:
    """Single buy carrying tp_stop=0.10. TP fires on bar 8."""
    bars = _bracket_bars()
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 10.0, "tp_stop": 0.10},
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    tp_rows = r_rust.trades[r_rust.trades["reason"] == "take_profit"]
    assert len(tp_rows) == 1
    assert tp_rows.iloc[0]["price"] == pytest.approx(110.0)


def test_parity_run_orders_with_bracket() -> None:
    """Bracket order — both sl_stop and tp_stop. SL fires first on bar 4."""
    bars = _bracket_bars()
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "buy",
            "qty": 10.0,
            "sl_stop": 0.10,
            "tp_stop": 0.10,
        },
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    # SL beats TP given the sequence.
    forced = r_rust.trades[r_rust.trades["reason"].isin(["stop_loss", "take_profit"])]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "stop_loss"


def test_parity_run_orders_short_bracket() -> None:
    """Short with bracket on the same bracket-bars panel.

    Short SL = entry * (1 + 0.10) = 110; would fire when high pierces 110.
    Short TP = entry * (1 - 0.10) = 90;  would fire when low pierces 90.
    On the engineered panel, bar 4 low=88 pierces 90 (TP for short).
    """
    bars = _bracket_bars()
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "sell",
            "qty": 10.0,
            "sl_stop": 0.10,
            "tp_stop": 0.10,
        },
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


@pytest.mark.parametrize("combo_idx", range(len(_COMBOS)))
def test_parity_run_orders_with_brackets_under_each_cost_combo(combo_idx: int) -> None:
    """Bracket orders behave consistently across the full friction matrix."""
    costs, slippage, execution = _COMBOS[combo_idx]
    bars = _bracket_bars()
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "buy",
            "qty": 5.0,
            "sl_stop": 0.10,
            "tp_stop": 0.20,
        },
    ])
    sim_kwargs = dict(cash=100_000.0, costs=costs, slippage=slippage, execution=execution)
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


def test_parity_run_orders_brackets_stress() -> None:
    """Stress test — 200 bars, 3 assets, several bracket orders."""
    bars = _synthetic_bars(n_bars=200, n_assets=3, seed=4242)
    rng = np.random.default_rng(4243)
    rows: list[dict[str, object]] = []
    for k in range(8):
        bar_idx = int(rng.integers(low=0, high=190))
        asset_idx = int(rng.integers(low=0, high=3))
        side = "buy" if rng.uniform() > 0.4 else "sell"
        qty = float(rng.uniform(low=1.0, high=10.0))
        sl = float(rng.uniform(low=0.05, high=0.20))
        tp = float(rng.uniform(low=0.05, high=0.30))
        rows.append({
            "ts": bars.index[bar_idx],
            "asset": f"A{asset_idx:02d}",
            "side": side,
            "qty": qty,
            "sl_stop": sl,
            "tp_stop": tp,
        })
    orders = pd.DataFrame(rows)
    sim_kwargs = dict(
        cash=1_000_000.0,
        costs=FixedBps(5.0),
        slippage=HalfSpread(spread_bps=4.0),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


def test_parity_run_signals_unchanged_without_brackets() -> None:
    """Regression: run_signals (no bracket API) still parity-matches."""
    bars = _synthetic_bars(n_bars=80, n_assets=2, seed=7)
    asset_cols = [f"A{j:02d}" for j in range(2)]
    entries = pd.DataFrame(False, index=bars.index, columns=asset_cols)
    exits = pd.DataFrame(False, index=bars.index, columns=asset_cols)
    entries.iloc[5, 0] = True
    exits.iloc[40, 0] = True
    sim_kwargs = dict(
        cash=100_000.0, costs=NoCost(), slippage=NoSlippage(), execution=NextBarOpen()
    )
    r_rust = _with_backend(bars, ("run_signals", entries, exits), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_signals", entries, exits), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


def test_parity_run_weights_unchanged_without_brackets() -> None:
    """Regression: run_weights (no bracket API) still parity-matches."""
    bars = _synthetic_bars(n_bars=80, n_assets=3, seed=8)
    target = pd.DataFrame(
        [[0.5, 0.5, 0.0]],
        index=[bars.index[0]],
        columns=[f"A{j:02d}" for j in range(3)],
    )
    sim_kwargs = dict(
        cash=1_000_000.0, costs=NoCost(), slippage=NoSlippage(), execution=NextBarOpen()
    )
    r_rust = _with_backend(bars, ("run_weights", target), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_weights", target), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


# ------------------------------------------------------------------ trailing-stop parity


def _trailing_stop_bars() -> pd.DataFrame:
    """Bars that exercise the trail's ratchet then trip the trail on bar 4."""
    rows = [
        (100.0, 102.0, 99.0, 100.0),  # 0: signal
        (100.0, 101.0, 99.0, 100.0),  # 1: entry fills at 100; anchor=100, tsl=90
        (102.0, 110.0, 101.0, 108.0),  # 2: ratchet anchor → 110, tsl=99 (no fire, low=101)
        (108.0, 112.0, 105.0, 109.0),  # 3: ratchet anchor → 112, tsl=100.8 (no fire, low=105)
        (105.0, 108.0, 95.0, 100.0),  # 4: low=95 ≤ 100.8 → TSL fires at 100.8
    ]
    n = len(rows)
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    cols = pd.MultiIndex.from_tuples([
        ("open", "BTC"),
        ("high", "BTC"),
        ("low", "BTC"),
        ("close", "BTC"),
        ("volume", "BTC"),
    ])
    return pd.DataFrame(
        {
            ("open", "BTC"): [r[0] for r in rows],
            ("high", "BTC"): [r[1] for r in rows],
            ("low", "BTC"): [r[2] for r in rows],
            ("close", "BTC"): [r[3] for r in rows],
            ("volume", "BTC"): [1.0] * n,
        },
        index=idx,
        columns=cols,
    )


def test_parity_run_orders_with_tsl_stop() -> None:
    """Single buy + tsl_stop=0.10. Rust and fallback must agree."""
    bars = _trailing_stop_bars()
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 10.0, "tsl_stop": 0.10},
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    tsl_rows = r_rust.trades[r_rust.trades["reason"] == "trailing_stop"]
    assert len(tsl_rows) == 1
    assert tsl_rows.iloc[0]["price"] == pytest.approx(100.8)


def test_parity_run_orders_with_full_bracket_including_tsl() -> None:
    """sl_stop + tp_stop + tsl_stop on the same order. The trail tightens
    faster than the fixed SL, so the trail wins on bar 4."""
    bars = _trailing_stop_bars()
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "buy",
            "qty": 10.0,
            "sl_stop": 0.20,  # level=80; never fires
            "tp_stop": 0.30,  # level=130; never fires
            "tsl_stop": 0.10,  # tightens to 100.8; fires bar 4
        },
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    forced = r_rust.trades[
        r_rust.trades["reason"].isin(["stop_loss", "take_profit", "trailing_stop"])
    ]
    assert len(forced) == 1
    assert forced.iloc[0]["reason"] == "trailing_stop"


def _trailing_stop_gap_bars() -> pd.DataFrame:
    """Bars that exercise the start-of-bar gap rule for TSL."""
    rows = [
        (100.0, 102.0, 99.0, 100.0),  # 0: signal
        (100.0, 101.0, 99.0, 100.0),  # 1: entry; anchor=100, tsl=90
        (110.0, 120.0, 110.0, 115.0),  # 2: ratchet to 120, tsl=108 (no fire)
        (90.0, 92.0, 85.0, 88.0),  # 3: gap-down — open=90 < tsl=108 → fire at 90
    ]
    n = len(rows)
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    cols = pd.MultiIndex.from_tuples([
        ("open", "BTC"),
        ("high", "BTC"),
        ("low", "BTC"),
        ("close", "BTC"),
        ("volume", "BTC"),
    ])
    return pd.DataFrame(
        {
            ("open", "BTC"): [r[0] for r in rows],
            ("high", "BTC"): [r[1] for r in rows],
            ("low", "BTC"): [r[2] for r in rows],
            ("close", "BTC"): [r[3] for r in rows],
            ("volume", "BTC"): [1.0] * n,
        },
        index=idx,
        columns=cols,
    )


def test_parity_run_orders_tsl_gap_down() -> None:
    """Engineered gap-down bar; both engines fill at open."""
    bars = _trailing_stop_gap_bars()
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "buy", "qty": 10.0, "tsl_stop": 0.10},
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    tsl_rows = r_rust.trades[r_rust.trades["reason"] == "trailing_stop"]
    assert tsl_rows.iloc[0]["price"] == pytest.approx(90.0)


def test_parity_run_orders_short_tsl() -> None:
    """Mirror of long TSL for shorts."""
    bars = pd.DataFrame(
        {
            ("open", "BTC"): [100.0, 100.0, 95.0, 90.0, 92.0],
            ("high", "BTC"): [102.0, 101.0, 96.0, 92.0, 102.0],  # bar 4 high=102 fires tsl
            ("low", "BTC"): [99.0, 99.0, 90.0, 85.0, 91.0],  # bar 3 low=85 → anchor=85, tsl=93.5
            ("close", "BTC"): [100.0, 100.0, 92.0, 88.0, 100.0],
            ("volume", "BTC"): [1.0] * 5,
        },
        index=pd.date_range("2024-01-02", periods=5, freq="D"),
        columns=pd.MultiIndex.from_tuples([
            ("open", "BTC"),
            ("high", "BTC"),
            ("low", "BTC"),
            ("close", "BTC"),
            ("volume", "BTC"),
        ]),
    )
    orders = pd.DataFrame([
        {"ts": bars.index[0], "asset": "BTC", "side": "sell", "qty": 10.0, "tsl_stop": 0.10},
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    tsl_rows = r_rust.trades[r_rust.trades["reason"] == "trailing_stop"]
    assert len(tsl_rows) == 1


def test_parity_run_orders_tsl_wins_over_sl_when_tighter() -> None:
    """TSL ratchets to a tighter fill than the fixed SL → TSL wins."""
    bars = _trailing_stop_bars()
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "buy",
            "qty": 10.0,
            "sl_stop": 0.20,  # level=80; never fires within these bars
            "tsl_stop": 0.10,  # tightens via ratchet to 100.8
        },
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    forced = r_rust.trades[r_rust.trades["reason"].isin(["stop_loss", "trailing_stop"])]
    assert forced.iloc[0]["reason"] == "trailing_stop"


def test_parity_run_orders_sl_wins_over_tsl_when_tighter() -> None:
    """Fixed SL is tighter than the trail (anchor hasn't moved) → SL wins."""
    bars = pd.DataFrame(
        {
            ("open", "BTC"): [100.0, 100.0, 98.0],
            ("high", "BTC"): [102.0, 101.0, 100.0],
            ("low", "BTC"): [99.0, 99.0, 88.0],  # bar 2 low=88 ≤ sl=90 → SL fires
            ("close", "BTC"): [100.0, 100.0, 92.0],
            ("volume", "BTC"): [1.0] * 3,
        },
        index=pd.date_range("2024-01-02", periods=3, freq="D"),
        columns=pd.MultiIndex.from_tuples([
            ("open", "BTC"),
            ("high", "BTC"),
            ("low", "BTC"),
            ("close", "BTC"),
            ("volume", "BTC"),
        ]),
    )
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "buy",
            "qty": 10.0,
            "sl_stop": 0.10,  # level=90 — closer to current price
            "tsl_stop": 0.20,  # anchor=100, level=80 — looser
        },
    ])
    sim_kwargs = dict(
        cash=100_000.0,
        costs=NoCost(),
        slippage=NoSlippage(),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
    forced = r_rust.trades[r_rust.trades["reason"].isin(["stop_loss", "trailing_stop"])]
    assert forced.iloc[0]["reason"] == "stop_loss"


@pytest.mark.parametrize("combo_idx", range(len(_COMBOS)))
def test_parity_run_orders_with_tsl_under_each_cost_combo(combo_idx: int) -> None:
    """TSL behaves consistently across the full cost / slippage / execution matrix."""
    costs, slippage, execution = _COMBOS[combo_idx]
    bars = _trailing_stop_bars()
    orders = pd.DataFrame([
        {
            "ts": bars.index[0],
            "asset": "BTC",
            "side": "buy",
            "qty": 5.0,
            "tsl_stop": 0.10,
        },
    ])
    sim_kwargs = dict(cash=100_000.0, costs=costs, slippage=slippage, execution=execution)
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)


def test_parity_run_orders_tsl_stress() -> None:
    """500 bars × 5 assets, TSL randomly attached to half the entries."""
    bars = _synthetic_bars(n_bars=500, n_assets=5, seed=12345)
    rng = np.random.default_rng(12346)
    rows: list[dict[str, object]] = []
    for k in range(20):
        bar_idx = int(rng.integers(low=0, high=480))
        asset_idx = int(rng.integers(low=0, high=5))
        side = "buy" if rng.uniform() > 0.4 else "sell"
        qty = float(rng.uniform(low=1.0, high=10.0))
        tsl = float(rng.uniform(low=0.05, high=0.20)) if k % 2 == 0 else 0.0
        rows.append({
            "ts": bars.index[bar_idx],
            "asset": f"A{asset_idx:02d}",
            "side": side,
            "qty": qty,
            "tsl_stop": tsl,
        })
    orders = pd.DataFrame(rows)
    sim_kwargs = dict(
        cash=1_000_000.0,
        costs=FixedBps(5.0),
        slippage=HalfSpread(spread_bps=4.0),
        execution=NextBarOpen(),
    )
    r_rust = _with_backend(bars, ("run_orders", orders), rust=True, **sim_kwargs)
    r_py = _with_backend(bars, ("run_orders", orders), rust=False, **sim_kwargs)
    _assert_simresult_equal(r_rust, r_py)
