"""Tests for local ``fundcloud.data`` backends: CSV.

The keyed read-write format backends (Memory, Parquet, DuckDB) have
their protocol-conformance coverage in ``test_backends_writable.py``.
CSV is read-only and has specific multi-file / MultiIndex semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import ReadOnlyError
from fundcloud.data.csv import CSV


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=5, freq="D").values,
        name="date",
    )
    return pd.DataFrame(
        {"open": np.arange(5, dtype=float), "close": np.arange(5, dtype=float) + 1},
        index=idx,
    )


def test_csv_single_file(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    path = tmp_path / "AAPL.csv"
    tiny_frame.to_csv(path, index_label="date")
    src = CSV(path, date_col="date")
    out = src.read()
    assert out.shape == tiny_frame.shape
    assert src.symbols == ["AAPL"]


def test_csv_directory(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    for sym in ("AAA", "BBB"):
        tiny_frame.to_csv(tmp_path / f"{sym}.csv", index_label="date")
    src = CSV(tmp_path, date_col="date")
    out = src.read()
    assert isinstance(out.columns, pd.MultiIndex)
    assert set(src.symbols) == {"AAA", "BBB"}
    assert ("close", "AAA") in out.columns


def test_csv_directory_symbols_filter(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    for sym in ("AAA", "BBB", "CCC"):
        tiny_frame.to_csv(tmp_path / f"{sym}.csv", index_label="date")
    src = CSV(tmp_path, date_col="date", symbols=["AAA", "BBB"])
    out = src.read()
    assert set(out.columns.get_level_values(-1)) == {"AAA", "BBB"}


def test_csv_is_read_only_by_default(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    path = tmp_path / "AAPL.csv"
    tiny_frame.to_csv(path, index_label="date")
    src = CSV(path, date_col="date")
    assert src.read_only is True
    with pytest.raises(ReadOnlyError):
        src.write("AAPL", tiny_frame)


def test_csv_overwrite_when_not_read_only(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    path = tmp_path / "out.csv"
    src = CSV(path, date_col="date", read_only=False)
    src.write("out", tiny_frame)
    round_trip = CSV(path, date_col="date").read()
    expected = tiny_frame.copy()
    expected.index.name = None  # CSV.read normalises index.name to None
    pd.testing.assert_frame_equal(round_trip.sort_index(axis=1), expected.sort_index(axis=1))


def test_csv_rejects_non_overwrite_modes(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    path = tmp_path / "out.csv"
    src = CSV(path, date_col="date", read_only=False)
    src.write("out", tiny_frame)
    with pytest.raises(NotImplementedError):
        src.write("out", tiny_frame, mode="append")
    with pytest.raises(NotImplementedError):
        src.write("out", tiny_frame, mode="upsert")


def test_csv_keys_single_file(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    path = tmp_path / "AAPL.csv"
    tiny_frame.to_csv(path, index_label="date")
    src = CSV(path, date_col="date")
    assert src.keys() == ["AAPL"]


def test_csv_keys_directory(tmp_path: Path, tiny_frame: pd.DataFrame) -> None:
    for sym in ("AAA", "BBB"):
        tiny_frame.to_csv(tmp_path / f"{sym}.csv", index_label="date")
    src = CSV(tmp_path, date_col="date")
    assert src.keys() == ["AAA", "BBB"]
