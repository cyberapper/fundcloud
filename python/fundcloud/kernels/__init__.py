"""Accelerated numerical kernels.

Prefers the Rust-backed ``fundcloud._core`` extension module when available
(built by maturin from ``crates/fundcloud-py``); falls back to pure-Python
implementations in :mod:`fundcloud.kernels._pyfallback` when the extension
isn't present.

Call sites are deliberately keep argument lists identical between the two
backends so the parity tests in ``tests/unit/test_kernels_parity.py`` can
swap implementations without adapters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fundcloud.kernels import _pyfallback

__all__ = [
    "HAS_RUST",
    "cvar_batch",
    "drawdown_series",
    "kernel_version",
    "max_drawdown_batch",
    "returns_from_prices",
    "rolling_mean",
    "rolling_mean_batch",
    "rolling_std",
    "rolling_std_batch",
    "sharpe_batch",
    "sortino_batch",
    "var_batch",
]


_core: Any
try:
    from fundcloud import _core as _core_module  # type: ignore[attr-defined]

    _core = _core_module
    HAS_RUST = True
except ImportError:  # pragma: no cover — exercised only without the Rust wheel
    _core = None
    HAS_RUST = False


def kernel_version() -> str:
    """Version string of the active kernel backend."""
    if _core is not None:
        return str(_core.kernel_version())
    return _pyfallback.kernel_version()


# ---------------------------------------------------------------------- helpers


def _as_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"_as_1d: expected 1-D array, got shape {arr.shape}")
    return arr


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        msg = f"expected a 2-D array, got shape {arr.shape}"
        raise ValueError(msg)
    # Ensure C-contiguous so rust-numpy reads it zero-copy.
    return np.ascontiguousarray(arr)


# ------------------------------------------------------------------ returns


def returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Simple period-over-period returns. First element is NaN."""
    arr = _as_1d(prices)
    if _core is not None:
        return np.asarray(_core.returns_from_prices(arr))
    return _pyfallback.returns_from_prices(arr)


# ------------------------------------------------------------------ rolling


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    arr = _as_1d(x)
    if _core is not None:
        return np.asarray(_core.rolling_mean(arr, int(window)))
    return _pyfallback.rolling_mean(arr, int(window))


def rolling_std(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    arr = _as_1d(x)
    if _core is not None:
        return np.asarray(_core.rolling_std(arr, int(window), int(ddof)))
    return _pyfallback.rolling_std(arr, int(window), int(ddof))


def rolling_mean_batch(x: np.ndarray, window: int) -> np.ndarray:
    arr = _as_2d(x)
    if _core is not None:
        return np.asarray(_core.rolling_mean_batch(arr, int(window)))
    return _pyfallback.rolling_mean_batch(arr, int(window))


def rolling_std_batch(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    arr = _as_2d(x)
    if _core is not None:
        return np.asarray(_core.rolling_std_batch(arr, int(window), int(ddof)))
    return _pyfallback.rolling_std_batch(arr, int(window), int(ddof))


# ------------------------------------------------------------------ drawdown


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    arr = _as_1d(returns)
    if _core is not None:
        return np.asarray(_core.drawdown_series(arr))
    return _pyfallback.drawdown_series(arr)


def max_drawdown_batch(returns: np.ndarray) -> np.ndarray:
    arr = _as_2d(returns)
    if _core is not None:
        return np.asarray(_core.max_drawdown_batch(arr))
    return _pyfallback.max_drawdown_batch(arr)


# ------------------------------------------------------------------ moments


def sharpe_batch(
    returns: np.ndarray, rf_per_period: float = 0.0, periods_per_year: float = 252.0
) -> np.ndarray:
    arr = _as_2d(returns)
    if _core is not None:
        return np.asarray(_core.sharpe_batch(arr, float(rf_per_period), float(periods_per_year)))
    return _pyfallback.sharpe_batch(arr, float(rf_per_period), float(periods_per_year))


def sortino_batch(
    returns: np.ndarray, target: float = 0.0, periods_per_year: float = 252.0
) -> np.ndarray:
    arr = _as_2d(returns)
    if _core is not None:
        return np.asarray(_core.sortino_batch(arr, float(target), float(periods_per_year)))
    return _pyfallback.sortino_batch(arr, float(target), float(periods_per_year))


# --------------------------------------------------------------- tail risk


def var_batch(returns: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    arr = _as_2d(returns)
    if _core is not None:
        return np.asarray(_core.var_batch(arr, float(alpha)))
    return _pyfallback.var_batch(arr, float(alpha))


def cvar_batch(returns: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    arr = _as_2d(returns)
    if _core is not None:
        return np.asarray(_core.cvar_batch(arr, float(alpha)))
    return _pyfallback.cvar_batch(arr, float(alpha))
