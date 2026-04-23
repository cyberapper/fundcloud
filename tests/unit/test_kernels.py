"""Tests for the kernels shim (Rust extension or pure-Python fallback)."""

from __future__ import annotations

import numpy as np
from fundcloud import kernels


def test_kernel_version_is_a_string() -> None:
    v = kernels.kernel_version()
    assert isinstance(v, str) and v


def test_returns_from_prices_empty() -> None:
    out = kernels.returns_from_prices(np.array([], dtype=float))
    assert out.shape == (0,)


def test_returns_from_prices_matches_pandas() -> None:
    import pandas as pd

    prices = np.array([100.0, 110.0, 99.0, 99.0, 101.0])
    ours = kernels.returns_from_prices(prices)
    ref = pd.Series(prices).pct_change().to_numpy()
    # NaN-safe elementwise compare.
    assert np.isnan(ours[0]) and np.isnan(ref[0])
    assert np.allclose(ours[1:], ref[1:], atol=1e-12)


def test_returns_from_prices_single_value() -> None:
    out = kernels.returns_from_prices(np.array([42.0]))
    assert np.isnan(out[0])
