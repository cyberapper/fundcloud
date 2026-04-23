"""Tests for :class:`fundcloud.features.FeaturePipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.features import FeaturePipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone


class _DoubleCloser(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # For MultiIndex Bars: take close for each asset.
        if isinstance(X.columns, pd.MultiIndex):
            close = X.xs("close", axis=1, level=0)
        else:
            close = X
        return close * 2


class _ShiftCloser(TransformerMixin, BaseEstimator):
    def __init__(self, periods: int = 1):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X.columns, pd.MultiIndex):
            close = X.xs("close", axis=1, level=0)
        else:
            close = X
        return close.shift(self.periods)


def _make_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    data = {
        ("close", "AAA"): 100 + np.cumsum(rng.normal(0, 1, 20)),
        ("close", "BBB"): 200 + np.cumsum(rng.normal(0, 1, 20)),
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_empty_pipeline_returns_empty_frame() -> None:
    pipe = FeaturePipeline()
    out = pipe.fit_transform(_make_panel())
    assert out.shape == (20, 0)


def test_single_transformer_columns_prefixed() -> None:
    pipe = FeaturePipeline([("dbl", _DoubleCloser())])
    out = pipe.fit_transform(_make_panel())
    assert list(out.columns) == ["dbl__AAA", "dbl__BBB"]


def test_multiple_transformers_concatenate_columnwise() -> None:
    pipe = FeaturePipeline([("dbl", _DoubleCloser()), ("shift", _ShiftCloser(periods=2))])
    out = pipe.fit_transform(_make_panel())
    assert list(out.columns) == ["dbl__AAA", "dbl__BBB", "shift__AAA", "shift__BBB"]
    assert len(out) == 20


def test_pipeline_hash_is_deterministic() -> None:
    a = FeaturePipeline([("s", _ShiftCloser(periods=3))])
    b = FeaturePipeline([("s", _ShiftCloser(periods=3))])
    c = FeaturePipeline([("s", _ShiftCloser(periods=4))])
    assert a.pipeline_hash == b.pipeline_hash
    assert a.pipeline_hash != c.pipeline_hash


def test_pipeline_clone_resets_state() -> None:
    pipe = FeaturePipeline([("s", _ShiftCloser(periods=2))])
    pipe.fit(_make_panel())
    fresh = pipe.clone()
    assert isinstance(fresh, FeaturePipeline)
    # clone is a new instance; mutating fresh doesn't affect the original.
    fresh["s"].set_params(periods=99)
    assert pipe["s"].periods == 2


def test_pipeline_is_sklearn_compatible() -> None:
    pipe = FeaturePipeline([("s", _ShiftCloser())])
    assert isinstance(pipe, BaseEstimator)
    cloned = clone(pipe)  # sklearn's clone()
    assert isinstance(cloned, FeaturePipeline)


def test_series_output_lifted_to_dataframe() -> None:
    class ReturnsSeries(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            close = X.xs("close", axis=1, level=0)
            return close.iloc[:, 0].rename("latest")  # a Series

    pipe = FeaturePipeline([("one", ReturnsSeries())])
    out = pipe.fit_transform(_make_panel())
    assert list(out.columns) == ["one__latest"]
