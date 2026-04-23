"""Portfolio optimisation.

Always available: the pure-Python fallback optimisers
(:class:`EqualWeighted`, :class:`InverseVolatility`, :class:`MVO`) from
:mod:`fundcloud.optimize.fallback_mvo`.

Behind the ``[pf]`` extra: the skfolio adapters
(:class:`MeanRisk`, :class:`RiskBudgeting`, :class:`HierarchicalRiskParity`,
…) are imported lazily so the core install stays cheap.

When both sets name the same class (e.g. ``EqualWeighted``) the skfolio
version wins **when the extra is installed**; otherwise the fallback is
returned. This keeps user code identical whether or not skfolio is present;
it just loses the richer constraint language when skfolio is missing.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from fundcloud.optimize.fallback_mvo import (
    MVO,
    BaseFallbackOptimizer,
)
from fundcloud.optimize.fallback_mvo import (
    EqualWeighted as _FallbackEqualWeighted,
)
from fundcloud.optimize.fallback_mvo import (
    InverseVolatility as _FallbackInverseVolatility,
)

__all__ = [
    "MVO",
    "BaseFallbackOptimizer",
    "EqualWeighted",
    "HierarchicalEqualRiskContribution",
    "HierarchicalRiskParity",
    "InverseVolatility",
    "MaximumDiversification",
    "MeanRisk",
    "NestedClustersOptimization",
    "RiskBudgeting",
    "RiskMeasure",
]


class _LocalRiskMeasure(str, Enum):
    """Minimal stand-in for ``skfolio.RiskMeasure`` when skfolio is absent."""

    VARIANCE = "VARIANCE"
    SEMI_VARIANCE = "SEMI_VARIANCE"
    STANDARD_DEVIATION = "STANDARD_DEVIATION"
    CVAR = "CVAR"
    WORST_REALIZATION = "WORST_REALIZATION"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    MEAN_ABSOLUTE_DEVIATION = "MEAN_ABSOLUTE_DEVIATION"


# ``RiskMeasure`` resolves eagerly to whichever concrete enum is available.
try:
    from skfolio import RiskMeasure as _SkRiskMeasure  # type: ignore[import-not-found]

    RiskMeasure: type[Enum] = _SkRiskMeasure  # type: ignore[assignment]
    _SKFOLIO_AVAILABLE = True
except ImportError:
    RiskMeasure = _LocalRiskMeasure  # type: ignore[assignment]
    _SKFOLIO_AVAILABLE = False


_SKFOLIO_NAMES = {
    "MeanRisk",
    "RiskBudgeting",
    "HierarchicalRiskParity",
    "HierarchicalEqualRiskContribution",
    "MaximumDiversification",
    "NestedClustersOptimization",
    # Overloadable — skfolio's richer version wins when installed.
    "EqualWeighted",
    "InverseVolatility",
}


def __getattr__(name: str) -> Any:
    if name in _SKFOLIO_NAMES:
        if not _SKFOLIO_AVAILABLE:
            if name in {"EqualWeighted", "InverseVolatility"}:
                return {
                    "EqualWeighted": _FallbackEqualWeighted,
                    "InverseVolatility": _FallbackInverseVolatility,
                }[name]
            msg = (
                f"{name!r} requires skfolio. Install with: "
                "uv add 'fundcloud[pf]' (or pass the pf extra to your existing install)."
            )
            raise ImportError(msg)
        from fundcloud.optimize import adapters

        return getattr(adapters, name)
    raise AttributeError(f"module 'fundcloud.optimize' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise lazy-loaded skfolio adapters for tab-completion."""
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # pragma: no cover
    from fundcloud.optimize.adapters import (
        EqualWeighted,
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        InverseVolatility,
        MaximumDiversification,
        MeanRisk,
        NestedClustersOptimization,
        RiskBudgeting,
    )
