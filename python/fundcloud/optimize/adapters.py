"""skfolio optimiser wrappers.

Each wrapper is a thin shell around a skfolio optimiser that:

1. accepts Fundcloud conventions on input (``fit(returns)``),
2. returns a Fundcloud ``Portfolio`` from ``predict``,
3. stays compatible with sklearn pipelines / GridSearchCV.

If the ``[pf]`` extra isn't installed the import-time ``__getattr__`` hook
in :mod:`fundcloud.optimize` raises a helpful ``ImportError`` instead.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# These references are resolved at import time — this module is only imported
# via the ``fundcloud.optimize.__getattr__`` hook when the ``[pf]`` extra is
# present, so the skfolio import is safe here.
from skfolio.optimization import (
    EqualWeighted as _SkEqualWeighted,
)
from skfolio.optimization import (
    HierarchicalEqualRiskContribution as _SkHERC,
)
from skfolio.optimization import (
    HierarchicalRiskParity as _SkHRP,
)
from skfolio.optimization import (
    InverseVolatility as _SkInverseVolatility,
)
from skfolio.optimization import (
    MaximumDiversification as _SkMaxDiv,
)
from skfolio.optimization import (
    MeanRisk as _SkMeanRisk,
)
from skfolio.optimization import (
    NestedClustersOptimization as _SkNested,
)
from skfolio.optimization import (
    RiskBudgeting as _SkRiskBudgeting,
)
from sklearn.base import BaseEstimator

from fundcloud.portfolio import Portfolio

__all__ = [
    "EqualWeighted",
    "HierarchicalEqualRiskContribution",
    "HierarchicalRiskParity",
    "InverseVolatility",
    "MaximumDiversification",
    "MeanRisk",
    "NestedClustersOptimization",
    "RiskBudgeting",
]


class _SkfolioAdapter(BaseEstimator):  # type: ignore[misc]
    """Shared machinery: build the underlying skfolio estimator and delegate."""

    _skfolio_cls: type[BaseEstimator]

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._inner: BaseEstimator | None = None
        self.weights_: Any = None
        self.assets_: list[str] | None = None

    # ---- plumbing -----------------------------------------------------
    def _build(self) -> BaseEstimator:
        return self._skfolio_cls(**self._kwargs)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self._kwargs)

    def set_params(self, **params: Any) -> _SkfolioAdapter:
        self._kwargs.update(params)
        self._inner = None
        return self

    # ---- sklearn API --------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any = None) -> _SkfolioAdapter:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.assets_ = list(X.columns)
        self._inner = self._build()
        self._inner.fit(X, y)
        self.weights_ = getattr(self._inner, "weights_", None)
        return self

    def predict(self, X: pd.DataFrame) -> Portfolio:
        if self._inner is None:
            raise RuntimeError("call fit() before predict()")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        sk_portfolio = self._inner.predict(X)
        return Portfolio.from_skfolio(sk_portfolio)

    def score(self, X: pd.DataFrame, y: Any = None) -> float:
        return float(self.predict(X).sharpe())


# --------------------------------------------------------------- concrete


def _make_adapter(name: str, skfolio_cls: type) -> type:
    cls = type(
        name, (_SkfolioAdapter,), {"_skfolio_cls": skfolio_cls, "__doc__": skfolio_cls.__doc__}
    )
    return cls


MeanRisk = _make_adapter("MeanRisk", _SkMeanRisk)
RiskBudgeting = _make_adapter("RiskBudgeting", _SkRiskBudgeting)
HierarchicalRiskParity = _make_adapter("HierarchicalRiskParity", _SkHRP)
HierarchicalEqualRiskContribution = _make_adapter("HierarchicalEqualRiskContribution", _SkHERC)
MaximumDiversification = _make_adapter("MaximumDiversification", _SkMaxDiv)
NestedClustersOptimization = _make_adapter("NestedClustersOptimization", _SkNested)
InverseVolatility = _make_adapter("InverseVolatility", _SkInverseVolatility)
EqualWeighted = _make_adapter("EqualWeighted", _SkEqualWeighted)
