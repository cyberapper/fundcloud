"""Tests for ``fundcloud.data._base`` defaults."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from fundcloud.data._base import Backend, BaseBackend, ReadOnlyError


class _RecordingSink(BaseBackend):
    """In-memory sink used to verify ``sync_to`` plumbing."""

    name = "recording"

    def __init__(self) -> None:
        self.read_only = False
        self._data: dict[str, pd.DataFrame] = {}
        self.write_calls: list[tuple[str, str]] = []  # (key, mode)

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self._data[key or "default"].copy()

    def keys(self) -> list[str]:
        return sorted(self._data)

    def write(self, key: str, df: pd.DataFrame, *, mode: str = "overwrite") -> None:
        self._check_writable()
        self.write_calls.append((key, mode))
        self._data[key] = df.copy()

    def delete(self, key: str) -> None:
        self._check_writable()
        self._data.pop(key, None)


class _ConstantSource(BaseBackend):
    """Read-only source that always returns the same frame."""

    name = "constant"
    read_only = True

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self._df.copy()


@pytest.fixture
def small_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D").values)
    return pd.DataFrame({"x": np.arange(5, dtype=float)}, index=idx)


def test_satisfies_protocol(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    sink = _RecordingSink()
    assert isinstance(src, Backend)
    assert isinstance(sink, Backend)


def test_read_only_blocks_write(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    with pytest.raises(ReadOnlyError):
        src.write("foo", small_frame)
    with pytest.raises(ReadOnlyError):
        src.delete("foo")


def test_writable_subclass_missing_override_raises_not_implemented() -> None:
    class _Bare(BaseBackend):
        name = "bare"

        def read(self, key=None, *, start=None, end=None, columns=None):  # type: ignore[no-untyped-def]
            return pd.DataFrame()

    with pytest.raises(NotImplementedError):
        _Bare().write("k", pd.DataFrame())
    with pytest.raises(NotImplementedError):
        _Bare().delete("k")


def test_default_keys_returns_default(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    assert src.keys() == ["default"]
    assert src.exists("default")
    assert not src.exists("other")


def test_default_last_index(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    assert src.last_index() == small_frame.index[-1]


def test_default_last_index_empty_returns_none() -> None:
    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    assert _ConstantSource(empty).last_index() is None


def test_sync_to_delegates_to_read_then_write(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    sink = _RecordingSink()

    out = src.sync_to(sink, key="x", mode="overwrite")

    assert sink.write_calls == [("x", "overwrite")]
    pd.testing.assert_frame_equal(sink.read("x"), small_frame)
    pd.testing.assert_frame_equal(out, small_frame)


def test_sync_to_default_mode_is_upsert(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    sink = _RecordingSink()
    src.sync_to(sink, key="x")
    assert sink.write_calls == [("x", "upsert")]


def test_sync_to_uses_default_key_when_unspecified(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    sink = _RecordingSink()
    src.sync_to(sink)
    assert sink.write_calls == [("default", "upsert")]


def test_sync_to_into_read_only_sink_raises(small_frame: pd.DataFrame) -> None:
    src = _ConstantSource(small_frame)
    sink = _RecordingSink()
    sink.read_only = True
    with pytest.raises(ReadOnlyError):
        src.sync_to(sink, key="x")
