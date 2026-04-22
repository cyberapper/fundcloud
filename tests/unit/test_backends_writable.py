"""Protocol-conformance tests for writable backends.

Parametrised over every writable concrete backend so we get uniform
coverage of round-trip, append, upsert, error, last_index, delete, and
``read_only`` enforcement.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import Backend, ReadOnlyError
from fundcloud.data.duckdb import DuckDB
from fundcloud.data.memory import Memory
from fundcloud.data.parquet import Parquet


@pytest.fixture
def small_frame() -> pd.DataFrame:
    # Strip inferred freq so assert_frame_equal doesn't see spurious diffs
    # after parquet / duckdb round-trips (both legitimately drop it).
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D").values)
    return pd.DataFrame(
        {
            "open": np.arange(10, dtype=float),
            "close": np.arange(10, dtype=float) + 1,
        },
        index=idx,
    )


# Each factory returns a writable Backend. tmp_path is provided for
# disk-backed backends; in-memory backends ignore it.
_FACTORIES: list[pytest.ParameterSet] = [
    pytest.param(lambda _tmp: Memory(), id="memory"),
    pytest.param(lambda tmp: Parquet(tmp), id="parquet"),
    pytest.param(lambda tmp: DuckDB(tmp / "backend.duckdb"), id="duckdb"),
]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_round_trip_is_identity(
    factory: Callable[[Path], Backend], tmp_path: Path, small_frame: pd.DataFrame
) -> None:
    backend = factory(tmp_path)
    assert not backend.exists("equity/us/daily")
    backend.write("equity/us/daily", small_frame)
    assert backend.exists("equity/us/daily")
    got = backend.read("equity/us/daily")
    pd.testing.assert_frame_equal(got.sort_index(axis=1), small_frame.sort_index(axis=1))


@pytest.mark.parametrize("factory", _FACTORIES)
def test_upsert_dedupes_overlap(
    factory: Callable[[Path], Backend], tmp_path: Path, small_frame: pd.DataFrame
) -> None:
    backend = factory(tmp_path)
    head = small_frame.iloc[:6]
    overlap_tail = small_frame.iloc[4:]
    backend.write("k", head)
    backend.write("k", overlap_tail, mode="upsert")
    got = backend.read("k")
    assert len(got) == len(small_frame)
    pd.testing.assert_index_equal(got.sort_index().index, small_frame.sort_index().index)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_append_does_not_dedupe(
    factory: Callable[[Path], Backend], tmp_path: Path, small_frame: pd.DataFrame
) -> None:
    backend = factory(tmp_path)
    head = small_frame.iloc[:6]
    overlap_tail = small_frame.iloc[4:]
    backend.write("k", head)
    backend.write("k", overlap_tail, mode="append")
    got = backend.read("k")
    # 6 head + 6 tail = 12 rows; overlap rows ARE present twice.
    assert len(got) == len(head) + len(overlap_tail)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_error_mode_raises(
    factory: Callable[[Path], Backend], tmp_path: Path, small_frame: pd.DataFrame
) -> None:
    backend = factory(tmp_path)
    backend.write("k", small_frame)
    with pytest.raises(FileExistsError):
        backend.write("k", small_frame, mode="error")


@pytest.mark.parametrize("factory", _FACTORIES)
def test_last_index_and_delete(
    factory: Callable[[Path], Backend], tmp_path: Path, small_frame: pd.DataFrame
) -> None:
    backend = factory(tmp_path)
    assert backend.last_index("k") is None
    backend.write("k", small_frame)
    assert backend.last_index("k") == small_frame.index[-1]
    backend.delete("k")
    assert not backend.exists("k")
    # Deleting a missing key is a no-op.
    backend.delete("k")


@pytest.mark.parametrize("factory", _FACTORIES)
def test_read_with_start_end_and_columns(
    factory: Callable[[Path], Backend], tmp_path: Path, small_frame: pd.DataFrame
) -> None:
    backend = factory(tmp_path)
    backend.write("k", small_frame)
    sliced = backend.read(
        "k", start="2024-01-03", end="2024-01-07", columns=["close"]
    )
    assert list(sliced.columns) == ["close"]
    assert sliced.index.min() == pd.Timestamp("2024-01-03")
    assert sliced.index.max() == pd.Timestamp("2024-01-07")


@pytest.fixture
def multi_frame() -> pd.DataFrame:
    """Two-symbol OHLCV frame with a canonical (field, symbol) MultiIndex."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=8, freq="D").values)
    data = {}
    for sym in ("SPY", "AAPL"):
        data[("open", sym)] = np.arange(8, dtype=float)
        data[("high", sym)] = np.arange(8, dtype=float) + 2
        data[("low", sym)] = np.arange(8, dtype=float) - 1
        data[("close", sym)] = np.arange(8, dtype=float) + 1
        data[("volume", sym)] = np.full(8, 1000.0)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.mark.parametrize("factory", _FACTORIES)
def test_multiindex_columns_round_trip(
    factory: Callable[[Path], Backend], tmp_path: Path, multi_frame: pd.DataFrame
) -> None:
    """Multi-symbol OHLCV frames must survive write → read on every backend.

    YF / FMP / AV / Binance all emit `(field, symbol)` MultiIndex columns for
    multi-symbol pulls; every writable backend must round-trip that shape.
    """
    backend = factory(tmp_path)
    backend.write("us_eq", multi_frame)
    got = backend.read("us_eq")
    assert isinstance(got.columns, pd.MultiIndex)
    pd.testing.assert_frame_equal(
        got.sort_index(axis=1), multi_frame.sort_index(axis=1)
    )


@pytest.mark.parametrize("factory", _FACTORIES)
def test_multiindex_columns_filter(
    factory: Callable[[Path], Backend], tmp_path: Path, multi_frame: pd.DataFrame
) -> None:
    """`columns=["close"]` on a MultiIndex frame keeps all `close` entries."""
    backend = factory(tmp_path)
    backend.write("us_eq", multi_frame)
    got = backend.read("us_eq", columns=["close"])
    fields = {c[0] for c in got.columns}
    assert fields == {"close"}
    symbols = {c[1] for c in got.columns}
    assert symbols == {"SPY", "AAPL"}


def test_memory_initial_preload(small_frame: pd.DataFrame) -> None:
    mem = Memory({"foo": small_frame})
    assert mem.keys() == ["foo"]
    pd.testing.assert_frame_equal(mem.read("foo"), small_frame)


def test_memory_initial_requires_datetime_index() -> None:
    with pytest.raises(TypeError):
        Memory({"foo": pd.DataFrame({"x": [1, 2, 3]})})


def test_memory_read_no_key_when_single_key(small_frame: pd.DataFrame) -> None:
    mem = Memory({"only": small_frame})
    pd.testing.assert_frame_equal(mem.read(), small_frame)


def test_memory_read_no_key_with_multiple_keys_raises(small_frame: pd.DataFrame) -> None:
    mem = Memory({"a": small_frame, "b": small_frame})
    with pytest.raises(KeyError):
        mem.read()


def test_read_only_blocks_write(small_frame: pd.DataFrame) -> None:
    mem = Memory({"foo": small_frame}, read_only=True)
    with pytest.raises(ReadOnlyError):
        mem.write("foo", small_frame)
    with pytest.raises(ReadOnlyError):
        mem.delete("foo")
