"""Cross-validation splitters.

Ships Fundcloud's own :class:`PurgedKFold` and :class:`EmbargoedKFold`
(sklearn-compatible, always available) plus re-exports of skfolio's
:class:`CombinatorialPurgedCV` and :class:`WalkForward` when the ``[pf]``
extra is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fundcloud.validate.splitters import EmbargoedKFold, PurgedKFold

__all__ = ["CombinatorialPurgedCV", "EmbargoedKFold", "PurgedKFold", "WalkForward"]


def __getattr__(name: str) -> Any:
    """Lazily expose skfolio re-exports without forcing the import."""
    if name in {"CombinatorialPurgedCV", "WalkForward"}:
        try:
            from skfolio.model_selection import (
                CombinatorialPurgedCV,
                WalkForward,
            )
        except ImportError as e:
            msg = (
                f"'{name}' requires the optional dependency skfolio. "
                "Install with: uv add 'fundcloud[pf]'"
            )
            raise AttributeError(msg) from e
        return {"CombinatorialPurgedCV": CombinatorialPurgedCV, "WalkForward": WalkForward}[name]
    raise AttributeError(f"module 'fundcloud.validate' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise lazy skfolio re-exports for tab-completion."""
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # pragma: no cover
    # Exposed for type-checkers when skfolio is installed.
    from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
