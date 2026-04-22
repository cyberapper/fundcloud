"""Time-series cross-validation splitters.

Both implementations follow sklearn's ``BaseCrossValidator`` contract so they
drop into ``GridSearchCV``, ``cross_val_score``, and
``skfolio.model_selection.cross_val_predict`` without adapters.

``PurgedKFold`` removes the ``purge`` timesteps immediately before each test
fold from the train set, preventing leakage through overlapping label
windows. ``EmbargoedKFold`` additionally drops ``embargo`` timesteps
immediately *after* each test fold so serially-correlated residuals cannot
leak the other way.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import BaseCrossValidator

__all__ = ["EmbargoedKFold", "PurgedKFold"]


class PurgedKFold(BaseCrossValidator):  # type: ignore[misc]
    """K-fold splitter with a purge buffer between train and test.

    Parameters
    ----------
    n_splits
        Number of folds. Must be at least 2.
    purge
        Number of samples immediately before each test fold to remove from the
        train set.
    """

    def __init__(self, n_splits: int = 5, *, purge: int = 0) -> None:
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)
        if purge < 0:
            raise ValueError("purge must be non-negative")
        self.n_splits = n_splits
        self.purge = purge

    def get_n_splits(
        self,
        X: object | None = None,
        y: object | None = None,
        groups: object | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: object,
        y: object | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = _n_samples(X)
        if n_samples < self.n_splits:
            msg = f"Cannot have n_splits={self.n_splits} > n_samples={n_samples}"
            raise ValueError(msg)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test = indices[start:stop]
            purge_start = max(0, start - self.purge)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:stop] = False
            yield indices[train_mask], test
            current = stop


class EmbargoedKFold(BaseCrossValidator):  # type: ignore[misc]
    """Purged K-fold with an additional *embargo* period after each test fold.

    Parameters
    ----------
    n_splits
        Number of folds. Must be at least 2.
    purge
        Samples to remove *before* the test fold.
    embargo
        Samples to remove *after* the test fold.
    """

    def __init__(self, n_splits: int = 5, *, purge: int = 0, embargo: int = 0) -> None:
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)
        if purge < 0 or embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def get_n_splits(
        self,
        X: object | None = None,
        y: object | None = None,
        groups: object | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: object,
        y: object | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = _n_samples(X)
        if n_samples < self.n_splits:
            msg = f"Cannot have n_splits={self.n_splits} > n_samples={n_samples}"
            raise ValueError(msg)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test = indices[start:stop]
            purge_start = max(0, start - self.purge)
            embargo_stop = min(n_samples, stop + self.embargo)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_stop] = False
            yield indices[train_mask], test
            current = stop


def _n_samples(X: object) -> int:
    """Best-effort sample count that works for DataFrames, arrays, and sequences."""
    if hasattr(X, "shape"):
        shape = X.shape
        if isinstance(shape, tuple) and len(shape) >= 1:
            return int(shape[0])
    try:
        return len(X)  # type: ignore[arg-type]
    except TypeError as e:
        msg = f"Cannot infer n_samples from {type(X).__name__}"
        raise TypeError(msg) from e
