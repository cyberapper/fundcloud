"""Dispatcher for the deterministic simulator kernels.

Mirrors the pattern used by :mod:`fundcloud.kernels.__init__`: prefers
the Rust-backed ``fundcloud._core`` extension when available, falls back
to :mod:`fundcloud.kernels._sim_pyfallback` when it isn't. The two paths
are bit-compatible to ``atol=1e-10`` (verified in
``tests/unit/test_sim_parity.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fundcloud.kernels import HAS_RUST, _core
from fundcloud.kernels import _sim_pyfallback as _pyfb


def _have_rust_sim() -> bool:
    return bool(HAS_RUST and _core is not None and hasattr(_core, "sim_run_weights"))


def run_weights(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    target_weights: np.ndarray,
    target_bar_indices: np.ndarray,
    cfg: _pyfb.SimCfg,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    if _have_rust_sim():
        return _core.sim_run_weights(
            open_panel, close_panel, target_weights, target_bar_indices,
            cfg.cash, cfg.cost_kind, cfg.cost_param1, cfg.cost_param2,
            cfg.slip_kind, cfg.slip_param1, cfg.exec_kind, float(tolerance),
        )
    return _pyfb.run_weights_loop(
        open_panel, close_panel, target_weights, target_bar_indices, cfg, tolerance=tolerance
    )


def run_orders(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    order_bar: np.ndarray,
    order_asset: np.ndarray,
    order_side: np.ndarray,
    order_qty: np.ndarray,
    order_notional: np.ndarray,
    order_kind: np.ndarray,
    order_limit_price: np.ndarray,
    cfg: _pyfb.SimCfg,
) -> dict[str, Any]:
    if _have_rust_sim():
        return _core.sim_run_orders(
            open_panel, close_panel,
            order_bar, order_asset, order_side, order_qty, order_notional,
            order_kind, order_limit_price,
            cfg.cash, cfg.cost_kind, cfg.cost_param1, cfg.cost_param2,
            cfg.slip_kind, cfg.slip_param1, cfg.exec_kind,
        )
    return _pyfb.run_orders_loop(
        open_panel, close_panel,
        order_bar, order_asset, order_side, order_qty, order_notional,
        order_kind, order_limit_price, cfg,
    )


def run_signals(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    size: float,
    cfg: _pyfb.SimCfg,
) -> dict[str, Any]:
    if _have_rust_sim():
        # Rust binding expects uint8 arrays; np.bool_ isn't auto-converted.
        en8 = np.ascontiguousarray(entries, dtype=np.uint8)
        ex8 = np.ascontiguousarray(exits, dtype=np.uint8)
        return _core.sim_run_signals(
            open_panel, close_panel, en8, ex8, float(size),
            cfg.cash, cfg.cost_kind, cfg.cost_param1, cfg.cost_param2,
            cfg.slip_kind, cfg.slip_param1, cfg.exec_kind,
        )
    return _pyfb.run_signals_loop(
        open_panel, close_panel, entries, exits, size, cfg,
    )
