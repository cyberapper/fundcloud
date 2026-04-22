"""Parity tests — Rust kernels ↔ pure-Python fallback.

Every public kernel in :mod:`fundcloud.kernels` is exercised on the same
input twice: once through the active backend (Rust when present, Python
otherwise) and once through the :mod:`fundcloud.kernels._pyfallback`
reference. Results must agree to 1e-12 on well-behaved inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from fundcloud import kernels
from fundcloud.kernels import _pyfallback


@pytest.fixture
def seed() -> int:
    return 17


@pytest.fixture
def prices(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0.0, 0.5, 500))


@pytest.fixture
def returns_panel(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=(500, 10))


@pytest.fixture
def returns_1d(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=500)


def _close(a: np.ndarray, b: np.ndarray, atol: float = 1e-12) -> None:
    """NaN-aware close check."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    assert np.array_equal(nan_a, nan_b), "NaN positions differ"
    mask = ~nan_a
    if mask.any():
        np.testing.assert_allclose(a[mask], b[mask], atol=atol)


# ---------------------------------------------------------------------- returns


def test_returns_from_prices_parity(prices: np.ndarray) -> None:
    _close(kernels.returns_from_prices(prices), _pyfallback.returns_from_prices(prices))


# ---------------------------------------------------------------------- rolling


@pytest.mark.parametrize("window", [2, 5, 30])
def test_rolling_mean_parity(returns_1d: np.ndarray, window: int) -> None:
    _close(
        kernels.rolling_mean(returns_1d, window),
        _pyfallback.rolling_mean(returns_1d, window),
        atol=1e-12,
    )


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("window", [3, 10, 60])
def test_rolling_std_parity(returns_1d: np.ndarray, window: int, ddof: int) -> None:
    _close(
        kernels.rolling_std(returns_1d, window, ddof=ddof),
        _pyfallback.rolling_std(returns_1d, window, ddof=ddof),
        atol=1e-10,
    )


def test_rolling_mean_batch_parity(returns_panel: np.ndarray) -> None:
    _close(
        kernels.rolling_mean_batch(returns_panel, 20),
        _pyfallback.rolling_mean_batch(returns_panel, 20),
        atol=1e-12,
    )


def test_rolling_std_batch_parity(returns_panel: np.ndarray) -> None:
    _close(
        kernels.rolling_std_batch(returns_panel, 20, ddof=1),
        _pyfallback.rolling_std_batch(returns_panel, 20, ddof=1),
        atol=1e-10,
    )


# -------------------------------------------------------------------- drawdown


def test_drawdown_series_parity(returns_1d: np.ndarray) -> None:
    _close(kernels.drawdown_series(returns_1d), _pyfallback.drawdown_series(returns_1d))


def test_max_drawdown_batch_parity(returns_panel: np.ndarray) -> None:
    _close(
        kernels.max_drawdown_batch(returns_panel),
        _pyfallback.max_drawdown_batch(returns_panel),
    )


# -------------------------------------------------------------------- moments


def test_sharpe_batch_parity(returns_panel: np.ndarray) -> None:
    _close(
        kernels.sharpe_batch(returns_panel, 0.0001, 252.0),
        _pyfallback.sharpe_batch(returns_panel, 0.0001, 252.0),
        atol=1e-10,
    )


def test_sortino_batch_parity(returns_panel: np.ndarray) -> None:
    _close(
        kernels.sortino_batch(returns_panel, 0.0, 252.0),
        _pyfallback.sortino_batch(returns_panel, 0.0, 252.0),
        atol=1e-10,
    )


# -------------------------------------------------------------------- tail risk


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_var_batch_parity(returns_panel: np.ndarray, alpha: float) -> None:
    _close(
        kernels.var_batch(returns_panel, alpha),
        _pyfallback.var_batch(returns_panel, alpha),
        atol=1e-10,
    )


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_cvar_batch_parity(returns_panel: np.ndarray, alpha: float) -> None:
    _close(
        kernels.cvar_batch(returns_panel, alpha),
        _pyfallback.cvar_batch(returns_panel, alpha),
        atol=1e-10,
    )


# ---------------------------------------------------------------------- meta


def test_backend_version_reports_rust_when_extension_loaded() -> None:
    v = kernels.kernel_version()
    assert v
    if kernels.HAS_RUST:
        assert v != "python-fallback"
    else:
        assert v == "python-fallback"
