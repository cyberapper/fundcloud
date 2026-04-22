"""Tests for ``fundcloud.validate.splitters``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.validate import EmbargoedKFold, PurgedKFold
from sklearn.model_selection import BaseCrossValidator


def test_purged_kfold_is_basecrossvalidator() -> None:
    assert isinstance(PurgedKFold(n_splits=3), BaseCrossValidator)
    assert isinstance(EmbargoedKFold(n_splits=3), BaseCrossValidator)


def test_purged_kfold_splits_dataframe() -> None:
    X = pd.DataFrame(np.arange(100).reshape(-1, 1), columns=["x"])
    cv = PurgedKFold(n_splits=5, purge=2)
    folds = list(cv.split(X))
    assert len(folds) == 5
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        # Every test-adjacent index within the purge window must be excluded.
        for ti in test:
            for p in range(1, 3):
                assert (ti - p) not in set(train) or (ti - p) < 0


def test_purged_kfold_split_covers_all_test_indices() -> None:
    n = 50
    X = np.arange(n)
    cv = PurgedKFold(n_splits=5)
    seen: set[int] = set()
    for _train, test in cv.split(X):
        seen |= set(test)
    assert seen == set(range(n))


def test_embargoed_kfold_purges_both_sides() -> None:
    X = np.arange(100)
    cv = EmbargoedKFold(n_splits=5, purge=3, embargo=2)
    for train, test in cv.split(X):
        train_set = set(train)
        # Right after the test fold, ``embargo`` samples must be absent.
        end = test.max()
        for e in range(1, 3):
            if end + e < 100:
                assert (end + e) not in train_set


def test_invalid_n_splits_raises() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        EmbargoedKFold(n_splits=0)


def test_get_n_splits_returns_constant() -> None:
    assert PurgedKFold(n_splits=7).get_n_splits() == 7
    assert EmbargoedKFold(n_splits=4).get_n_splits() == 4


def test_skfolio_reexport_when_missing_raises() -> None:
    """When skfolio is not installed, accessing its re-exports raises."""
    import importlib

    import fundcloud.validate as v

    # If skfolio is installed in this env, attribute access just succeeds.
    try:
        importlib.import_module("skfolio")
        importlib.reload(v)
        assert hasattr(v, "CombinatorialPurgedCV")
    except ImportError:
        with pytest.raises(AttributeError, match="skfolio"):
            _ = v.CombinatorialPurgedCV
