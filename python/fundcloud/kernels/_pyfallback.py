"""Pure-Python reference implementations for every kernel.

These mirror the Rust kernels in :mod:`fundcloud._core` one-for-one so the
parity tests in ``tests/unit/test_kernels_parity.py`` can detect any drift
between the two. The fallback also runs transparently when the Rust
extension isn't available (e.g. on an arch without a prebuilt wheel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
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


def kernel_version() -> str:
    return "python-fallback"


def returns_from_prices(prices: np.ndarray) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    out = np.empty_like(arr)
    out[:] = np.nan
    if arr.size >= 2:
        out[1:] = arr[1:] / arr[:-1] - 1.0
    return out


# ---------------------------------------------------------------------- rolling


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rolling(window).mean().to_numpy()


def rolling_std(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rolling(window).std(ddof=ddof).to_numpy()


def rolling_mean_batch(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return pd.DataFrame(arr).rolling(window).mean().to_numpy()


def rolling_std_batch(x: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return pd.DataFrame(arr).rolling(window).std(ddof=ddof).to_numpy()


# --------------------------------------------------------------------- drawdown


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return arr
    filled = np.where(np.isnan(arr), 0.0, arr)
    wealth = np.cumprod(1.0 + filled)
    peak = np.maximum.accumulate(wealth)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = wealth / peak - 1.0
    out = np.where(np.isnan(out), 0.0, out)
    return out


def max_drawdown_batch(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_cols = arr.shape[1]
    out = np.zeros(n_cols, dtype=float)
    for c in range(n_cols):
        dd = drawdown_series(arr[:, c])
        out[c] = float(dd.min()) if dd.size else 0.0
    return out


# ---------------------------------------------------------------------- moments


def _safe_mean(col: np.ndarray) -> float:
    col = col[~np.isnan(col)]
    return float(col.mean()) if col.size else float("nan")


def _safe_std(col: np.ndarray, ddof: int) -> float:
    col = col[~np.isnan(col)]
    if col.size <= ddof:
        return float("nan")
    return float(col.std(ddof=ddof))


def sharpe_batch(returns: np.ndarray, rf_per_period: float, periods_per_year: float) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.empty(arr.shape[1], dtype=float)
    sqrt_pp = float(np.sqrt(periods_per_year))
    for c in range(arr.shape[1]):
        col = arr[:, c]
        mu = _safe_mean(col) - rf_per_period
        sigma = _safe_std(col, ddof=1)
        if not np.isfinite(sigma) or sigma == 0.0:
            out[c] = np.nan
        else:
            out[c] = (mu / sigma) * sqrt_pp
    return out


def sortino_batch(returns: np.ndarray, target: float, periods_per_year: float) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.empty(arr.shape[1], dtype=float)
    sqrt_pp = float(np.sqrt(periods_per_year))
    for c in range(arr.shape[1]):
        col = arr[:, c]
        clean = col[~np.isnan(col)]
        if clean.size == 0:
            out[c] = np.nan
            continue
        mu = float(clean.mean()) - target
        downside = np.clip(clean - target, a_min=None, a_max=0.0)
        dd = float(np.sqrt((downside**2).mean()))
        out[c] = np.nan if dd == 0.0 else (mu / dd) * sqrt_pp
    return out


# -------------------------------------------------------------------- tail risk


def var_batch(returns: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.empty(arr.shape[1], dtype=float)
    for c in range(arr.shape[1]):
        col = arr[~np.isnan(arr[:, c]), c]
        out[c] = float(np.quantile(col, 1.0 - alpha)) if col.size else float("nan")
    return out


def cvar_batch(returns: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.empty(arr.shape[1], dtype=float)
    for c in range(arr.shape[1]):
        col = arr[~np.isnan(arr[:, c]), c]
        if col.size == 0:
            out[c] = np.nan
            continue
        q = float(np.quantile(col, 1.0 - alpha))
        tail = col[col <= q]
        out[c] = float(tail.mean()) if tail.size else np.nan
    return out
