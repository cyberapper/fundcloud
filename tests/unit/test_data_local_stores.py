"""Local-store backend tests — Memory / CSV / Parquet / DuckDB.

The four classes implement :class:`fundcloud.data._base.BaseBackend`'s
read / write / keys / exists / last_index / delete contract over
different storage substrates. Tests verify the contract holds for each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------- helpers


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=8)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "open": rng.normal(100, 1, 8),
            "high": rng.normal(101, 1, 8),
            "low": rng.normal(99, 1, 8),
            "close": rng.normal(100.5, 1, 8),
            "volume": rng.integers(100_000, 200_000, 8).astype(float),
        },
        index=idx,
    )


@pytest.fixture
def multiindex_bars() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=5)
    rng = np.random.default_rng(1)
    cols: dict[tuple[str, str], np.ndarray] = {}
    for s in ["AAPL", "MSFT"]:
        base = 100 + np.cumsum(rng.normal(0, 0.5, 5))
        cols[("open", s)] = base
        cols[("close", s)] = base + 0.5
        cols[("volume", s)] = np.full(5, 1_000_000.0)
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# --------------------------------------------------------------------- Memory


def test_memory_initial_loads(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k1": daily_frame})
    out = src.read("k1")
    assert len(out) == len(daily_frame)


def test_memory_default_key_when_unique(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"only": daily_frame})
    out = src.read()  # no key
    assert len(out) == len(daily_frame)


def test_memory_keyless_read_with_multiple_keys_raises(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"a": daily_frame, "b": daily_frame})
    with pytest.raises(KeyError, match="multiple keys"):
        src.read()


def test_memory_unknown_key_raises(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"a": daily_frame})
    with pytest.raises(KeyError):
        src.read("unknown")


def test_memory_initial_rejects_non_datetime_index() -> None:
    from fundcloud.data.memory import Memory

    bad = pd.DataFrame({"x": [1.0, 2.0]})  # default RangeIndex
    with pytest.raises(TypeError, match="DatetimeIndex"):
        Memory(initial={"k": bad})


def test_memory_window_slice(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k": daily_frame})
    out = src.read("k", start="2024-01-03", end="2024-01-05")
    assert len(out) <= len(daily_frame)
    assert out.index.min() >= pd.Timestamp("2024-01-03")


def test_memory_columns_filter(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k": daily_frame})
    out = src.read("k", columns=["close"])
    assert list(out.columns) == ["close"]


def test_memory_keys_sorted(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"b": daily_frame, "a": daily_frame})
    assert src.keys() == ["a", "b"]


def test_memory_exists(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"a": daily_frame})
    assert src.exists("a")
    assert not src.exists("missing")


def test_memory_last_index(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k": daily_frame})
    assert src.last_index("k") == daily_frame.index[-1]
    assert src.last_index() == daily_frame.index[-1]
    assert src.last_index("missing") is None


def test_memory_last_index_returns_none_for_empty_frame() -> None:
    from fundcloud.data.memory import Memory

    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    src = Memory(initial={"k": empty})
    assert src.last_index("k") is None


def test_memory_write_overwrite(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory()
    src.write("k", daily_frame)
    assert src.exists("k")


def test_memory_write_error_when_exists(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k": daily_frame})
    with pytest.raises(FileExistsError):
        src.write("k", daily_frame, mode="error")


def test_memory_write_append_concats(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory()
    src.write("k", daily_frame.iloc[:4])
    src.write("k", daily_frame.iloc[4:], mode="append")
    assert len(src.read("k")) == 8


def test_memory_write_upsert_dedupes(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory()
    src.write("k", daily_frame.iloc[:5])
    # Overlapping range — last write wins.
    src.write("k", daily_frame.iloc[3:], mode="upsert")
    assert len(src.read("k")) == 8


def test_memory_write_rejects_non_datetime_index() -> None:
    from fundcloud.data.memory import Memory

    src = Memory()
    bad = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        src.write("k", bad)


def test_memory_read_only_blocks_write(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data import ReadOnlyError
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k": daily_frame}, read_only=True)
    with pytest.raises(ReadOnlyError):
        src.write("k2", daily_frame)


def test_memory_delete(daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.memory import Memory

    src = Memory(initial={"k": daily_frame})
    src.delete("k")
    assert not src.exists("k")
    # Deleting a missing key is silent.
    src.delete("k")


# --------------------------------------------------------------------- CSV


def test_csv_read_single_file(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    path = tmp_path / "AAPL.csv"
    daily_frame.rename_axis("date").to_csv(path)
    src = CSV(path)
    out = src.read()
    assert len(out) == len(daily_frame)


def test_csv_read_directory(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    daily_frame.rename_axis("date").to_csv(tmp_path / "AAPL.csv")
    daily_frame.rename_axis("date").to_csv(tmp_path / "MSFT.csv")
    src = CSV(tmp_path)
    out = src.read()
    # Two files → MultiIndex columns (field, symbol).
    assert isinstance(out.columns, pd.MultiIndex)
    assert set(out.columns.get_level_values(1)) == {"AAPL", "MSFT"}


def test_csv_read_directory_filtered_by_symbols(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    for sym in ["AAPL", "MSFT", "GOOG"]:
        daily_frame.rename_axis("date").to_csv(tmp_path / f"{sym}.csv")
    src = CSV(tmp_path, symbols=["AAPL", "GOOG"])
    out = src.read()
    assert set(out.columns.get_level_values(1)) == {"AAPL", "GOOG"}


def test_csv_read_directory_no_matching_files_raises(tmp_path: Path) -> None:
    from fundcloud.data.csv import CSV

    with pytest.raises(FileNotFoundError, match="No matching CSVs"):
        CSV(tmp_path).read()


def test_csv_window_slice(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    path = tmp_path / "AAPL.csv"
    daily_frame.rename_axis("date").to_csv(path)
    out = CSV(path).read(start="2024-01-03", end="2024-01-05")
    assert len(out) <= len(daily_frame)


def test_csv_columns_filter_flat(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    path = tmp_path / "AAPL.csv"
    daily_frame.rename_axis("date").to_csv(path)
    out = CSV(path).read(columns=["close"])
    assert list(out.columns) == ["close"]


def test_csv_columns_filter_multiindex(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    daily_frame.rename_axis("date").to_csv(tmp_path / "AAPL.csv")
    daily_frame.rename_axis("date").to_csv(tmp_path / "MSFT.csv")
    out = CSV(tmp_path).read(columns=["close"])
    fields = [c[0] for c in out.columns]
    assert set(fields) == {"close"}


def test_csv_keys_for_single_file(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    path = tmp_path / "AAPL.csv"
    daily_frame.rename_axis("date").to_csv(path)
    assert CSV(path).keys() == ["AAPL"]


def test_csv_keys_for_directory(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    for sym in ["B", "A", "C"]:
        daily_frame.rename_axis("date").to_csv(tmp_path / f"{sym}.csv")
    assert CSV(tmp_path).keys() == ["A", "B", "C"]


def test_csv_missing_date_column_raises(tmp_path: Path) -> None:
    from fundcloud.data.csv import CSV

    path = tmp_path / "x.csv"
    pd.DataFrame({"close": [1.0, 2.0]}).to_csv(path, index=False)  # no 'date'
    with pytest.raises(KeyError, match="missing date column"):
        CSV(path).read()


def test_csv_write_to_single_file(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    path = tmp_path / "out.csv"
    src = CSV(path, read_only=False)
    src.write("ignored", daily_frame)
    assert path.exists()
    rt = CSV(path).read()
    assert len(rt) == len(daily_frame)


def test_csv_write_to_directory(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    src = CSV(tmp_path, read_only=False)
    src.write("AAPL", daily_frame)
    assert (tmp_path / "AAPL.csv").exists()


def test_csv_write_only_overwrite_supported(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.csv import CSV

    src = CSV(tmp_path / "out.csv", read_only=False)
    with pytest.raises(NotImplementedError, match="overwrite"):
        src.write("k", daily_frame, mode="append")


def test_csv_write_rejects_non_datetime_index(tmp_path: Path) -> None:
    from fundcloud.data.csv import CSV

    src = CSV(tmp_path, read_only=False)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        src.write("k", pd.DataFrame({"x": [1.0]}))


def test_csv_default_read_only_blocks_write(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data import ReadOnlyError
    from fundcloud.data.csv import CSV

    src = CSV(tmp_path / "out.csv")  # read_only=True default
    with pytest.raises(ReadOnlyError):
        src.write("k", daily_frame)


# --------------------------------------------------------------------- Parquet


def test_parquet_write_then_read(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("AAPL", daily_frame)
    assert src.exists("AAPL")
    out = src.read("AAPL")
    assert len(out) == len(daily_frame)


def test_parquet_default_key_when_unique(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("AAPL", daily_frame)
    out = src.read()  # no key
    assert len(out) == len(daily_frame)


def test_parquet_keyless_read_multiple_keys_raises(
    tmp_path: Path, daily_frame: pd.DataFrame
) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("AAPL", daily_frame)
    src.write("MSFT", daily_frame)
    with pytest.raises(KeyError, match="multiple keys"):
        src.read()


def test_parquet_unknown_key_raises(tmp_path: Path) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    with pytest.raises(KeyError):
        src.read("missing")


def test_parquet_window_slice(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame)
    out = src.read("k", start="2024-01-03", end="2024-01-05")
    assert len(out) <= len(daily_frame)


def test_parquet_columns_filter(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame)
    out = src.read("k", columns=["close"])
    assert list(out.columns) == ["close"]


def test_parquet_columns_filter_multiindex(tmp_path: Path, multiindex_bars: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("bars", multiindex_bars)
    out = src.read("bars", columns=["close"])
    fields = {c[0] for c in out.columns}
    assert fields == {"close"}


def test_parquet_keys_with_nested_paths(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("equity/us/daily", daily_frame)
    keys = src.keys()
    assert "equity/us/daily" in keys


def test_parquet_last_index(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame)
    last = src.last_index("k")
    assert last == daily_frame.index[-1]


def test_parquet_last_index_default_key(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame)
    last = src.last_index()
    assert last is not None


def test_parquet_last_index_returns_none_for_missing(tmp_path: Path) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    assert src.last_index("missing") is None


def test_parquet_last_index_returns_none_when_no_unique_key(
    tmp_path: Path, daily_frame: pd.DataFrame
) -> None:
    """With multiple keys and no key=, last_index returns None."""
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("a", daily_frame)
    src.write("b", daily_frame)
    assert src.last_index() is None


def test_parquet_write_error_when_exists(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame)
    with pytest.raises(FileExistsError):
        src.write("k", daily_frame, mode="error")


def test_parquet_write_append(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame.iloc[:4])
    src.write("k", daily_frame.iloc[4:], mode="append")
    assert len(src.read("k")) == 8


def test_parquet_write_upsert(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame.iloc[:5])
    src.write("k", daily_frame.iloc[3:], mode="upsert")
    assert len(src.read("k")) == 8


def test_parquet_write_rejects_non_datetime_index(tmp_path: Path) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        src.write("k", pd.DataFrame({"x": [1.0]}))


def test_parquet_delete(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.parquet import Parquet

    src = Parquet(tmp_path)
    src.write("k", daily_frame)
    src.delete("k")
    assert not src.exists("k")


# --------------------------------------------------------------------- DuckDB


def test_duckdb_write_then_read(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    db = tmp_path / "store.duckdb"
    with DuckDB(db) as src:
        src.write("k", daily_frame)
        out = src.read("k")
    assert len(out) == len(daily_frame)


def test_duckdb_keys_and_exists(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("a", daily_frame)
        src.write("b", daily_frame)
        assert sorted(src.keys()) == ["a", "b"]
        assert src.exists("a")
        assert not src.exists("missing")


def test_duckdb_read_default_key_when_unique(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("only", daily_frame)
        out = src.read()
        assert len(out) == len(daily_frame)


def test_duckdb_keyless_read_multiple_raises(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("a", daily_frame)
        src.write("b", daily_frame)
        with pytest.raises(KeyError, match="multiple keys"):
            src.read()


def test_duckdb_unknown_key_raises(tmp_path: Path) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src, pytest.raises(KeyError):
        src.read("missing")


def test_duckdb_window_and_columns(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("k", daily_frame)
        out = src.read("k", start="2024-01-03", end="2024-01-05", columns=["close"])
        assert list(out.columns) == ["close"]
        assert out.index.min() >= pd.Timestamp("2024-01-03")


def test_duckdb_multiindex_columns_roundtrip(tmp_path: Path, multiindex_bars: pd.DataFrame) -> None:
    """`(field, symbol)` columns are flattened on write, rebuilt on read."""
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("bars", multiindex_bars)
        out = src.read("bars")
        assert isinstance(out.columns, pd.MultiIndex)
        # Column filtering on the rebuilt MultiIndex.
        sliced = src.read("bars", columns=["close"])
        fields = {c[0] for c in sliced.columns}
        assert fields == {"close"}


def test_duckdb_last_index(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("k", daily_frame)
        assert src.last_index("k") == daily_frame.index[-1]


def test_duckdb_last_index_no_unique_key(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("a", daily_frame)
        src.write("b", daily_frame)
        assert src.last_index() is None


def test_duckdb_last_index_default_key_when_unique(
    tmp_path: Path, daily_frame: pd.DataFrame
) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("only", daily_frame)
        last = src.last_index()
        assert last == daily_frame.index[-1]


def test_duckdb_last_index_missing_key(tmp_path: Path) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        assert src.last_index("missing") is None


def test_duckdb_write_error_mode(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("k", daily_frame)
        with pytest.raises(FileExistsError):
            src.write("k", daily_frame, mode="error")


def test_duckdb_write_append(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("k", daily_frame.iloc[:4])
        src.write("k", daily_frame.iloc[4:], mode="append")
        assert len(src.read("k")) == 8


def test_duckdb_write_upsert(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("k", daily_frame.iloc[:5])
        src.write("k", daily_frame.iloc[3:], mode="upsert")
        assert len(src.read("k")) == 8


def test_duckdb_write_rejects_non_datetime_index(tmp_path: Path) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src, pytest.raises(TypeError, match="DatetimeIndex"):
        src.write("k", pd.DataFrame({"x": [1.0]}))


def test_duckdb_delete(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("k", daily_frame)
        src.delete("k")
        assert not src.exists("k")
        # Deleting a missing key is silent.
        src.delete("k")


def test_duckdb_key_with_slashes_roundtrips(tmp_path: Path, daily_frame: pd.DataFrame) -> None:
    from fundcloud.data.duckdb import DuckDB

    with DuckDB(tmp_path / "x.db") as src:
        src.write("equity/us/daily", daily_frame)
        assert "equity/us/daily" in src.keys()
        out = src.read("equity/us/daily")
        assert len(out) == len(daily_frame)
