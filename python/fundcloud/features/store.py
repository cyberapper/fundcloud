"""Feature cache keyed by ``(dataset, pipeline_hash)``.

The :class:`FeatureStore` lets a user compute features once and re-use them
across fits, walk-forward splits, and report renders without paying the CPU
cost twice. It leans on whichever :class:`~fundcloud.data.Backend` is
passed in, so disk vs. in-memory vs. DuckDB is a swap decision, not a
rewrite.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fundcloud.data import Backend
from fundcloud.features.pipeline import FeaturePipeline

__all__ = ["FeatureStore"]


class FeatureStore:
    """Persistent cache for features derived from a pipeline.

    Parameters
    ----------
    store
        Any object implementing the :class:`~fundcloud.data.Backend` protocol.
        Typically a :class:`~fundcloud.data.Parquet` pointed at ``./.features/``.
    prefix
        Store-key prefix used for every feature dataset; defaults to
        ``"features"`` so feature frames don't collide with raw bars living in
        the same underlying store.
    """

    def __init__(self, store: Backend, *, prefix: str = "features") -> None:
        self._store = store
        self._prefix = prefix.rstrip("/")

    # ------------------------------------------------------------------ keys

    def _key(self, dataset: str, pipeline: FeaturePipeline) -> str:
        return f"{self._prefix}/{dataset}/{pipeline.pipeline_hash}"

    # ------------------------------------------------------------------ API

    def has(self, dataset: str, pipeline: FeaturePipeline) -> bool:
        return self._store.exists(self._key(dataset, pipeline))

    def get_or_compute(
        self,
        dataset: str,
        pipeline: FeaturePipeline,
        bars: pd.DataFrame,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return cached features if available, otherwise compute + persist.

        Parameters
        ----------
        dataset
            Logical dataset name (e.g. ``"equity-us-daily"``).
        pipeline
            Feature pipeline. Its ``pipeline_hash`` is part of the cache key.
        bars
            Input ``Bars`` frame to run through the pipeline.
        force
            If ``True``, ignore any cached result and recompute.
        """
        key = self._key(dataset, pipeline)
        if not force and self._store.exists(key):
            return self._store.read(key)
        features = pipeline.fit_transform(bars)
        self._store.write(key, features, mode="overwrite")
        return features

    def save(
        self,
        dataset: str,
        pipeline: FeaturePipeline,
        features: pd.DataFrame,
    ) -> None:
        """Persist a pre-computed feature frame. Useful when the user ran the
        pipeline manually and wants to share the result with later runs."""
        self._store.write(self._key(dataset, pipeline), features, mode="overwrite")

    def load(self, dataset: str, pipeline: FeaturePipeline) -> pd.DataFrame:
        return self._store.read(self._key(dataset, pipeline))

    def invalidate(self, dataset: str, pipeline: FeaturePipeline) -> None:
        self._store.delete(self._key(dataset, pipeline))

    # ----------------------------------------------------------------- inspect

    def list(self, dataset: str | None = None) -> list[str]:
        """List keys owned by this store (optionally filtered by dataset)."""
        prefix = f"{self._prefix}/" if dataset is None else f"{self._prefix}/{dataset}/"
        return [k for k in self._store.keys() if k.startswith(prefix)]  # noqa: SIM118 — Backend.keys() is a method, not a dict view

    # ------------------------------------------------------------------ dunder

    def __contains__(self, key: tuple[str, FeaturePipeline]) -> bool:
        dataset, pipeline = key
        return self.has(dataset, pipeline)

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        store_cls = type(self._store).__name__
        return f"FeatureStore(store={store_cls}, prefix={self._prefix!r})"


# Silence "unused" from linters when this symbol is only used as a type hint.
_ = Any
