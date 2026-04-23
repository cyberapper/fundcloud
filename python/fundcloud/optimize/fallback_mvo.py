"""Pure-Python fallback optimisers.

Always available (no skfolio dep). Each class is a
:class:`sklearn.base.BaseEstimator` so it slots into ``Pipeline`` /
``GridSearchCV`` without adapter. ``predict(X)`` returns a
:class:`fundcloud.portfolio.Portfolio`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from fundcloud.portfolio import Portfolio

__all__ = ["MVO", "BaseFallbackOptimizer", "EqualWeighted", "InverseVolatility"]


class BaseFallbackOptimizer(BaseEstimator):  # type: ignore[misc]
    """Common machinery: sklearn ``fit`` / ``predict`` / ``score`` contract."""

    weights_: np.ndarray | None = None
    assets_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: object | None = None) -> BaseFallbackOptimizer:
        df = _require_frame(X)
        self.assets_ = list(df.columns)
        self.weights_ = self._solve(df)
        return self

    def predict(self, X: pd.DataFrame) -> Portfolio:
        if self.weights_ is None or self.assets_ is None:
            raise RuntimeError("call fit() before predict()")
        df = _require_frame(X)
        if list(df.columns) != self.assets_:
            # Reorder test-set columns to training order; extra or missing
            # columns are an error we surface early.
            try:
                df = df[self.assets_]
            except KeyError as e:
                msg = f"predict() asset columns do not match fit(): missing {e}"
                raise ValueError(msg) from e
        returns = df @ self.weights_
        weights_row = pd.Series(self.weights_, index=self.assets_)
        return Portfolio(
            returns=returns.rename(type(self).__name__),
            weights=weights_row.to_frame().T.set_axis([returns.index[-1]], axis=0),
            name=type(self).__name__,
        )

    def score(self, X: pd.DataFrame, y: object | None = None) -> float:
        """Higher is better — defaults to annualised Sharpe on ``predict(X)``."""
        return float(self.predict(X).sharpe())

    # Subclasses implement the actual math.
    def _solve(self, returns: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------- naive


class EqualWeighted(BaseFallbackOptimizer):
    """Assign ``1/n`` to every asset."""

    def _solve(self, returns: pd.DataFrame) -> np.ndarray:
        n = returns.shape[1]
        return np.full(n, 1.0 / n)


class InverseVolatility(BaseFallbackOptimizer):
    """Weights inversely proportional to per-asset volatility (ddof=1)."""

    def _solve(self, returns: pd.DataFrame) -> np.ndarray:
        vol = returns.std(ddof=1).to_numpy()
        inv = np.where(vol > 0, 1.0 / vol, 0.0)
        total = inv.sum()
        if total == 0:
            n = len(vol)
            return np.full(n, 1.0 / n)
        return inv / total


# ---------------------------------------------------------------------- MVO


class MVO(BaseFallbackOptimizer):
    """Mean-Variance Optimiser — max-Sharpe under long-only, fully-invested.

    Parameters
    ----------
    risk_free
        Per-period risk-free rate (same units as returns). Defaults to ``0``.
    l2
        Optional L2 regularisation on the weight vector (helps when covariance
        is ill-conditioned).
    """

    def __init__(self, risk_free: float = 0.0, l2: float = 0.0) -> None:
        self.risk_free = float(risk_free)
        self.l2 = float(l2)

    def _solve(self, returns: pd.DataFrame) -> np.ndarray:
        mu = returns.mean().to_numpy()
        cov = returns.cov().to_numpy()
        n = len(mu)

        def neg_sharpe(w: np.ndarray) -> float:
            excess = float(w @ mu - self.risk_free)
            var = float(w @ cov @ w + self.l2 * (w @ w))
            if var <= 0:
                return 0.0
            return -excess / np.sqrt(var)

        x0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        constraints = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not result.success:
            # Fall back to equal weights rather than blowing up the pipeline.
            return x0
        w = np.asarray(result.x, dtype=float)
        # Clamp tiny negative numerical artefacts to zero and renormalise.
        w = np.clip(w, 0.0, None)
        total = w.sum()
        return (w / total) if total > 0 else x0


# -------------------------------------------------------------------- helpers


def _require_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    msg = f"Optimiser expected DataFrame, got {type(X).__name__}"
    raise TypeError(msg)
