"""Tests for :class:`fundcloud.features.FeatureStore`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import Memory, Parquet
from fundcloud.features import FeaturePipeline, FeatureStore
from sklearn.base import BaseEstimator, TransformerMixin


class _Pct(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Simple pct-change, fill first row with 0.
        if isinstance(X.columns, pd.MultiIndex):
            close = X.xs("close", axis=1, level=0)
        else:
            close = X
        return close.pct_change().fillna(0.0)


def _bars() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # Strip inferred freq so round-tripping through parquet does not trigger
    # spurious ``assert_frame_equal`` differences (`None` vs `<Day>`).
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=30, freq="D").values)
    df = pd.DataFrame(
        {
            ("close", "A"): 100 + np.cumsum(rng.normal(0, 1, 30)),
            ("close", "B"): 50 + np.cumsum(rng.normal(0, 1, 30)),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_get_or_compute_persists_on_first_call() -> None:
    store = Memory()
    fs = FeatureStore(store)
    pipe = FeaturePipeline([("p", _Pct())])
    assert not fs.has("equity", pipe)
    out = fs.get_or_compute("equity", pipe, _bars())
    assert fs.has("equity", pipe)
    assert out.shape == (30, 2)


def test_get_or_compute_reads_on_second_call(tmp_path: Path) -> None:
    fs = FeatureStore(Parquet(tmp_path))
    tr = _Pct()
    pipe = FeaturePipeline([("p", tr)])
    bars = _bars()
    first = fs.get_or_compute("equity", pipe, bars)

    # Spy on the transformer's transform method. A cache hit means transform is
    # never called on the second go.
    calls: list[object] = []
    original = tr.transform

    def counted(X):
        calls.append(X)
        return original(X)

    tr.transform = counted  # type: ignore[method-assign]
    second = fs.get_or_compute("equity", pipe, bars)
    assert not calls, "Expected cache hit; transform was invoked again"
    pd.testing.assert_frame_equal(second.sort_index(axis=1), first.sort_index(axis=1))


def test_force_recomputes_and_overwrites() -> None:
    fs = FeatureStore(Memory())
    pipe = FeaturePipeline([("p", _Pct())])
    bars = _bars()
    fs.get_or_compute("equity", pipe, bars)
    forced = fs.get_or_compute("equity", pipe, bars, force=True)
    assert forced.shape == (30, 2)


def test_invalidate_removes_entry() -> None:
    fs = FeatureStore(Memory())
    pipe = FeaturePipeline([("p", _Pct())])
    fs.get_or_compute("equity", pipe, _bars())
    fs.invalidate("equity", pipe)
    assert not fs.has("equity", pipe)


def test_list_filters_by_dataset() -> None:
    fs = FeatureStore(Memory())
    pipe_a = FeaturePipeline([("p1", _Pct())])
    pipe_b = FeaturePipeline([("p2", _Pct())])
    fs.get_or_compute("equity", pipe_a, _bars())
    fs.get_or_compute("crypto", pipe_b, _bars())
    assert any("equity" in k for k in fs.list("equity"))
    assert all("equity" not in k for k in fs.list("crypto"))


def test_contains_dunder() -> None:
    fs = FeatureStore(Memory())
    pipe = FeaturePipeline([("p", _Pct())])
    fs.get_or_compute("equity", pipe, _bars())
    assert ("equity", pipe) in fs


# Silence pytest "ARG002 unused" in fixtures above.
_ = pytest
