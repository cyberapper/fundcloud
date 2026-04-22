"""Feature pipelines — sklearn-compatible panels of feature transformers.

A :class:`FeaturePipeline` behaves like
:class:`sklearn.pipeline.FeatureUnion`: it fits every component on the same
input and concatenates their outputs column-wise into a single wide frame.
Column names are prefixed with the transformer's step name so output columns
never collide.

The pipeline is deterministic — equal sequences of (name, transformer)
produce equal outputs and share a stable ``pipeline_hash`` that the
:class:`fundcloud.features.store.FeatureStore` uses as part of its cache key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

__all__ = ["FeaturePipeline"]


class FeaturePipeline(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Apply a list of feature transformers and stack the columns.

    Parameters
    ----------
    transformers
        Sequence of ``(name, transformer)`` pairs. Each transformer must
        implement ``fit`` and ``transform``; each ``transform`` must return a
        :class:`pandas.DataFrame` aligned on the input's index.

    Notes
    -----
    Following sklearn convention, ``transformers`` is the single public param
    — this is what ``get_params`` / ``set_params`` operate on, which lets the
    pipeline round-trip through ``GridSearchCV`` cleanly.
    """

    def __init__(self, transformers: list[tuple[str, Any]] | None = None) -> None:
        self.transformers = transformers or []

    # ------------------------------------------------------------------ sklearn

    def fit(self, X: pd.DataFrame, y: object | None = None) -> FeaturePipeline:
        for _name, tr in self.transformers:
            tr.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.transformers:
            return pd.DataFrame(index=X.index)
        frames: list[pd.DataFrame] = []
        for name, tr in self.transformers:
            out = tr.transform(X)
            if not isinstance(out, pd.DataFrame):
                # Lift Series → one-column frame so downstream can rely on shape.
                if isinstance(out, pd.Series):
                    out = out.to_frame(out.name or "value")
                else:
                    msg = f"Transformer {name!r} must return DataFrame or Series, got {type(out).__name__}"
                    raise TypeError(msg)
            prefixed = out.copy()
            prefixed.columns = [f"{name}__{c}" for c in prefixed.columns]
            frames.append(prefixed)
        if len(frames) > 1:
            for frame in frames[1:]:
                if not frame.index.equals(frames[0].index):
                    raise ValueError(
                        "FeaturePipeline: transformer outputs have misaligned indices — "
                        "all transformers must return a frame with the same index as X."
                    )
        return pd.concat(frames, axis=1, join="inner")

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: object | None = None,
        **_fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ----------------------------------------------------------------- sugar

    def __len__(self) -> int:
        return len(self.transformers)

    def __getitem__(self, name_or_index: str | int) -> Any:
        if isinstance(name_or_index, int):
            return self.transformers[name_or_index][1]
        for name, tr in self.transformers:
            if name == name_or_index:
                return tr
        raise KeyError(name_or_index)

    def named_steps(self) -> dict[str, Any]:
        return dict(self.transformers)

    def clone(self) -> FeaturePipeline:
        """Return a freshly-cloned pipeline with reset estimator state."""
        return FeaturePipeline(transformers=[(name, clone(tr)) for name, tr in self.transformers])

    # ------------------------------------------------------------------ hashing

    @property
    def pipeline_hash(self) -> str:
        """Deterministic hash of the pipeline spec, for use as a cache key."""
        payload = [(name, _transformer_fingerprint(tr)) for name, tr in self.transformers]
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # --------------------------------------------------------------- iteration

    def steps(self) -> Iterable[tuple[str, Any]]:
        return iter(self.transformers)


def _transformer_fingerprint(tr: Any) -> dict[str, Any]:
    """Stable description of a transformer: class path + estimator params."""
    cls = type(tr)
    qualname = f"{cls.__module__}.{cls.__qualname__}"
    params = tr.get_params(deep=False) if hasattr(tr, "get_params") else {}
    return {"cls": qualname, "params": params}
