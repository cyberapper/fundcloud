"""Dispatcher for the deterministic simulator kernels.

Mirrors the pattern used by :mod:`fundcloud.kernels.__init__`: prefers
the Rust-backed ``fundcloud._core`` extension when available, falls back
to :mod:`fundcloud.kernels._sim_pyfallback` when it isn't. The two paths
are bit-compatible to ``atol=1e-10`` (verified in
``tests/unit/test_sim_parity.py``).

Bracket-order extension
-----------------------
The fallback supports intra-bar stop-loss / take-profit / trailing-stop
checks driven by the high/low panels. The Rust kernel ships gradually
— two probes (:func:`_rust_supports_brackets` for SL/TP and
:func:`_rust_supports_tsl_brackets` for the trailing stop) inspect the
binding so calls fall through to the fallback only if the Rust side
hasn't caught up to that bracket type yet. Non-bracket calls always
take the fast Rust path when available.

The output ``trade_reason`` field is emitted by both paths: the fallback
emits ``int`` codes that the simulator's
:func:`fundcloud.sim.simulator._rehydrate_sim_result` translates to
``"signal"`` / ``"stop_loss"`` / ``"take_profit"`` / ``"trailing_stop"``
for the trades DataFrame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fundcloud.kernels import HAS_RUST, _core
from fundcloud.kernels import _sim_pyfallback as _pyfb


def _have_rust_sim() -> bool:
    return bool(HAS_RUST and _core is not None and hasattr(_core, "sim_run_weights"))


def _rust_supports_brackets() -> bool:
    """Probe whether the loaded Rust binding accepts the bracket-order
    panels (high/low + sl/tp arrays).

    The Rust kernel ships gradually; this probe checks the binding's
    parameter list so we can route bracket-order calls through the
    fallback until the Rust side catches up. After parity lands the
    probe will return ``True`` and the dispatcher will use Rust for
    every call.
    """
    if not _have_rust_sim():
        return False
    import inspect

    try:
        sig = inspect.signature(_core.sim_run_orders)
    except (TypeError, ValueError):
        return False
    return "high_panel" in sig.parameters or "high" in sig.parameters


def _rust_supports_tsl_brackets() -> bool:
    """Probe whether the Rust binding's ``sim_run_orders`` accepts an
    ``order_tsl_stop`` array. Routes TSL-bearing calls through the
    fallback until the Rust binding is rebuilt with the new signature.
    """
    if not _have_rust_sim():
        return False
    import inspect

    try:
        sig = inspect.signature(_core.sim_run_orders)
    except (TypeError, ValueError):
        return False
    return "order_tsl_stop" in sig.parameters


def run_weights(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    high_panel: np.ndarray,
    low_panel: np.ndarray,
    target_weights: np.ndarray,
    target_bar_indices: np.ndarray,
    cfg: _pyfb.SimCfg,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Weights-path dispatch. Rust when available; fallback otherwise.

    ``high`` / ``low`` are forwarded to the fallback for parity with the
    bracket pipeline but are unused by the weights path (the
    ``run_weights`` API has no surface to attach brackets — they only
    enter via ``Order(sl_stop=..., tp_stop=...)``).
    """
    if _rust_supports_brackets():
        return _core.sim_run_weights(
            open_panel,
            close_panel,
            high_panel,
            low_panel,
            target_weights,
            target_bar_indices,
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
            float(tolerance),
        )
    if _have_rust_sim():
        return _core.sim_run_weights(
            open_panel,
            close_panel,
            target_weights,
            target_bar_indices,
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
            float(tolerance),
        )
    return _pyfb.run_weights_loop(
        open_panel,
        close_panel,
        high_panel,
        low_panel,
        target_weights,
        target_bar_indices,
        cfg,
        tolerance=tolerance,
    )


def run_orders(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    high_panel: np.ndarray,
    low_panel: np.ndarray,
    order_bar: np.ndarray,
    order_asset: np.ndarray,
    order_side: np.ndarray,
    order_qty: np.ndarray,
    order_notional: np.ndarray,
    order_kind: np.ndarray,
    order_limit_price: np.ndarray,
    order_sl_stop: np.ndarray,
    order_tp_stop: np.ndarray,
    order_tsl_stop: np.ndarray,
    cfg: _pyfb.SimCfg,
) -> dict[str, Any]:
    """Orders-path dispatch with bracket-order support (SL / TP / TSL).

    Routing rules — pick the highest-capability binding that supports
    every bracket type the call actually carries:

    * TSL-aware Rust + any call → Rust
    * Bracket-aware Rust + non-TSL call → Rust
    * Bracket-aware Rust + TSL-bearing call → fallback (Rust binding
      not yet rebuilt for TSL)
    * Bracket-naive Rust + non-bracket call → Rust (drop new params)
    * Bracket-naive Rust + bracket-bearing call → fallback
    * No Rust → fallback handles everything
    """
    has_sl_or_tp = (order_sl_stop.size > 0 and bool(np.any(order_sl_stop > 0))) or (
        order_tp_stop.size > 0 and bool(np.any(order_tp_stop > 0))
    )
    has_tsl = order_tsl_stop.size > 0 and bool(np.any(order_tsl_stop > 0))

    if _rust_supports_tsl_brackets():
        return _core.sim_run_orders(
            open_panel,
            close_panel,
            high_panel,
            low_panel,
            order_bar,
            order_asset,
            order_side,
            order_qty,
            order_notional,
            order_kind,
            order_limit_price,
            order_sl_stop,
            order_tp_stop,
            order_tsl_stop,
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
        )
    if not has_tsl and _rust_supports_brackets():
        return _core.sim_run_orders(
            open_panel,
            close_panel,
            high_panel,
            low_panel,
            order_bar,
            order_asset,
            order_side,
            order_qty,
            order_notional,
            order_kind,
            order_limit_price,
            order_sl_stop,
            order_tp_stop,
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
        )
    if not has_sl_or_tp and not has_tsl and _have_rust_sim():
        return _core.sim_run_orders(
            open_panel,
            close_panel,
            order_bar,
            order_asset,
            order_side,
            order_qty,
            order_notional,
            order_kind,
            order_limit_price,
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
        )
    return _pyfb.run_orders_loop(
        open_panel,
        close_panel,
        high_panel,
        low_panel,
        order_bar,
        order_asset,
        order_side,
        order_qty,
        order_notional,
        order_kind,
        order_limit_price,
        order_sl_stop,
        order_tp_stop,
        order_tsl_stop,
        cfg,
    )


def run_signals(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    high_panel: np.ndarray,
    low_panel: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    size: float,
    cfg: _pyfb.SimCfg,
) -> dict[str, Any]:
    """Signals-path dispatch. ``run_signals`` doesn't take per-order
    bracket parameters (entries/exits are boolean panels), so the
    routing reduces to "Rust when available". ``high`` / ``low`` are
    forwarded to the fallback for parity."""
    if _rust_supports_brackets():
        en8 = np.ascontiguousarray(entries, dtype=np.uint8)
        ex8 = np.ascontiguousarray(exits, dtype=np.uint8)
        return _core.sim_run_signals(
            open_panel,
            close_panel,
            high_panel,
            low_panel,
            en8,
            ex8,
            float(size),
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
        )
    if _have_rust_sim():
        # Rust binding expects uint8 arrays; np.bool_ isn't auto-converted.
        en8 = np.ascontiguousarray(entries, dtype=np.uint8)
        ex8 = np.ascontiguousarray(exits, dtype=np.uint8)
        return _core.sim_run_signals(
            open_panel,
            close_panel,
            en8,
            ex8,
            float(size),
            cfg.cash,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            cfg.exec_kind,
        )
    return _pyfb.run_signals_loop(
        open_panel,
        close_panel,
        high_panel,
        low_panel,
        entries,
        exits,
        size,
        cfg,
    )
