"""Synthetic OHLCV generator shared by every example.

The library ships with network providers (``YF``, ``FMP``, …) but we use
synthesised prices for examples so they're:

* reproducible (no rate-limits / outages),
* fast (no network latency), and
* license-clean (no third-party historical data bundled).

Swap ``generate_ohlcv(...)`` for ``YF(...).read()`` in your own notebooks once
you've added the ``fundcloud[data]`` extra.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

__all__ = ["AssetProfile", "generate_ohlcv"]


class AssetProfile:
    """Annualised mu / sigma plus a starting price."""

    __slots__ = ("mu", "price0", "sigma")

    def __init__(self, mu: float, sigma: float, price0: float = 100.0) -> None:
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.price0 = float(price0)


def generate_ohlcv(
    profiles: dict[str, AssetProfile],
    *,
    start: str = "2022-01-03",
    periods: int = 504,
    correlations: np.ndarray | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a MultiIndex ``Bars`` frame with columns ``(field, symbol)``.

    Each asset's closes follow a daily GBM parametrised by its
    :class:`AssetProfile`. Cross-correlations default to identity (assets
    independent); supply a correlation matrix to model co-movement.
    """
    rng = np.random.default_rng(seed)
    symbols = list(profiles)
    n_assets = len(symbols)

    corr = np.eye(n_assets) if correlations is None else np.asarray(correlations, dtype=float)
    if corr.shape != (n_assets, n_assets):
        msg = f"correlations must be {n_assets}x{n_assets}, got {corr.shape}"
        raise ValueError(msg)
    chol = np.linalg.cholesky(corr)

    dt = 1 / 252.0
    mus = np.array([profiles[s].mu for s in symbols]) * dt
    sigmas = np.array([profiles[s].sigma for s in symbols]) * np.sqrt(dt)
    raw = rng.normal(size=(periods, n_assets))
    shocks = raw @ chol.T
    log_returns = mus + sigmas * shocks

    price0 = np.array([profiles[s].price0 for s in symbols])
    prices = price0 * np.exp(np.cumsum(log_returns, axis=0))

    idx = pd.DatetimeIndex(pd.bdate_range(start, periods=periods).values)
    data: dict[tuple[str, str], np.ndarray] = {}
    for i, sym in enumerate(symbols):
        close = prices[:, i]
        open_ = close * (1.0 + rng.normal(0.0, 0.0005, periods))
        noise = np.abs(rng.normal(0.0, 0.002, periods))
        high = np.maximum(open_, close) * (1.0 + noise)
        low = np.minimum(open_, close) * (1.0 - noise)
        volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
        data[("open", sym)] = open_
        data[("high", sym)] = high
        data[("low", sym)] = low
        data[("close", sym)] = close
        data[("volume", sym)] = volume

    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return cast(pd.DataFrame, df.sort_index(axis=1))


def closes(bars: pd.DataFrame) -> pd.DataFrame:
    """Shorthand: pull the ``close`` field out of a Bars frame."""
    return cast(pd.DataFrame, bars.xs("close", axis=1, level=0))


def close_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Shorthand: simple pct-change of closes, first row dropped."""
    return cast(pd.DataFrame, closes(bars).pct_change().dropna())
