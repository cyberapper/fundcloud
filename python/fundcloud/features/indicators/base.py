"""``IndicatorSpec`` — sklearn-compatible base for technical indicators.

A concrete indicator either subclasses :class:`IndicatorSpec` directly and
overrides :meth:`_compute`, or is auto-generated from a TA-Lib function by
:mod:`fundcloud.features.indicators._talib_autogen`. Every indicator exposes
the usual ``fit``/``transform`` pair and plays nicely with
:class:`fundcloud.features.pipeline.FeaturePipeline`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["IndicatorSpec", "register_indicator", "registered_indicators"]


class IndicatorSpec(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Base class for every technical indicator.

    Subclasses declare their TA-Lib function name (or ``None`` for custom
    indicators) and a mapping of default parameters. The public interface
    (``fit`` / ``transform`` / ``get_params`` / ``set_params``) matches sklearn
    exactly.
    """

    talib_name: ClassVar[str | None] = None
    #: Ordered list of input price fields the indicator consumes. TA-Lib's
    #: abstract API uses exactly this naming.
    inputs: ClassVar[tuple[str, ...]] = ("close",)
    #: Name(s) of the output column(s) produced per asset.
    outputs: ClassVar[tuple[str, ...]] = ("value",)
    #: Default parameter values. Subclasses override.
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        resolved = {**self.default_params, **params}
        for key, val in resolved.items():
            setattr(self, key, val)
        # Record the keys so sklearn's get_params works without a hand-written
        # __init__ on every subclass.
        self._param_names: tuple[str, ...] = tuple(resolved.keys())

    # ------------------------------------------------------------------ sklearn

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._param_names}

    def set_params(self, **params: Any) -> IndicatorSpec:
        for k, v in params.items():
            if k not in self._param_names:
                self._param_names = (*self._param_names, k)
            setattr(self, k, v)
        return self

    def fit(self, X: pd.DataFrame, y: object | None = None) -> IndicatorSpec:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the indicator to each asset and concatenate.

        Accepts either a wide single-field frame (``columns = assets``) or a
        MultiIndex ``Bars`` frame with ``(field, asset)`` columns. Returns a
        wide frame whose columns are ``f"{output}_{asset}"``.
        """
        if isinstance(X.columns, pd.MultiIndex):
            assets = sorted(set(X.columns.get_level_values(-1)))
            per_asset: dict[str, pd.DataFrame] = {}
            for asset in assets:
                missing = [f for f in self.inputs if (f, asset) not in X.columns]
                if missing:
                    raise KeyError(
                        f"{type(self).__name__} requires fields {self.inputs!r}; "
                        f"asset {asset!r} is missing: {missing}"
                    )
                sub = {f: X[(f, asset)] for f in self.inputs}
                per_asset[asset] = self._compute(sub, X.index)
            return _stack(per_asset, outputs=self.outputs)
        # Wide single-field frame: one column per asset, close-field assumed.
        per_asset = {}
        for asset in X.columns:
            per_asset[asset] = self._compute({"close": X[asset]}, X.index)
        return _stack(per_asset, outputs=self.outputs)

    # ----------------------------------------------------------------- concrete

    def _compute(
        self,
        series_by_field: dict[str, pd.Series],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Compute the indicator for a single asset.

        Override this in subclasses that don't rely on TA-Lib. For TA-Lib-backed
        indicators the default implementation dispatches to the abstract API.
        """
        if self.talib_name is None:
            raise NotImplementedError(
                f"{type(self).__name__} must either set `talib_name` or override `_compute`."
            )
        import talib  # lazy so the module is importable without TA-Lib installed

        params = {k: getattr(self, k) for k in self._param_names if k in self.default_params}
        abstract = talib.abstract.Function(self.talib_name)
        prepared = {k: np.asarray(v.values, dtype=float) for k, v in series_by_field.items()}
        result = abstract(prepared, **params)
        # TA-Lib's abstract API returns a list for multi-output indicators
        # (BBANDS, MACD, STOCH…) and a bare ndarray for single-output ones.
        arrays = list(result) if isinstance(result, (list, tuple)) else [np.asarray(result)]
        if len(arrays) != len(self.outputs):
            msg = f"Expected {len(self.outputs)} outputs from {self.talib_name}, got {len(arrays)}"
            raise RuntimeError(msg)
        return pd.DataFrame({
            name: pd.Series(arr, index=index)
            for name, arr in zip(self.outputs, arrays, strict=True)
        })


def _stack(
    per_asset: dict[str, pd.DataFrame],
    *,
    outputs: tuple[str, ...],
) -> pd.DataFrame:
    """Concatenate per-asset frames into a wide frame.

    Column naming: single-output indicators use ``asset`` as the column name;
    multi-output indicators use ``{output}__{asset}``.
    """
    if len(outputs) == 1:
        cols = {asset: df[outputs[0]] for asset, df in per_asset.items()}
        return pd.DataFrame(cols)
    frames: dict[str, pd.Series] = {}
    for asset, df in per_asset.items():
        for out in outputs:
            frames[f"{out}__{asset}"] = df[out]
    return pd.DataFrame(frames)


# -------------------------------------------------------------------- registry


_REGISTRY: dict[str, type[IndicatorSpec]] = {}


def register_indicator(name: str) -> Any:
    """Decorator that makes a custom :class:`IndicatorSpec` discoverable by name."""

    def deco(cls: type[IndicatorSpec]) -> type[IndicatorSpec]:
        _REGISTRY[name] = cls
        return cls

    return deco


def registered_indicators() -> dict[str, type[IndicatorSpec]]:
    """Return the registry as a (defensive) copy."""
    return dict(_REGISTRY)
