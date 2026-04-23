"""Tests for ``Backend.sync_to`` end-to-end semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import Memory, ReadOnlyError


@pytest.fixture
def small_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D").values)
    return pd.DataFrame(
        {
            "open": np.arange(10, dtype=float),
            "close": np.arange(10, dtype=float) + 1,
        },
        index=idx,
    )


def test_sync_round_trip_via_upsert(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory()
    src.sync_to(sink, key="dest")
    pd.testing.assert_frame_equal(
        sink.read("dest").sort_index(axis=1),
        small_frame.sort_index(axis=1),
    )


def test_sync_upsert_dedupes_overlap_on_repeat(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory()
    src.sync_to(sink, key="dest")
    src.sync_to(sink, key="dest")  # repeat with full overlap
    assert len(sink.read("dest")) == len(small_frame)
    pd.testing.assert_index_equal(sink.read("dest").sort_index().index, small_frame.index)


def test_sync_append_mode_does_not_dedupe(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory()
    src.sync_to(sink, key="dest", mode="overwrite")
    src.sync_to(sink, key="dest", mode="append")
    # Append duplicates the entire frame.
    assert len(sink.read("dest")) == 2 * len(small_frame)


def test_sync_overwrite_replaces(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory({"dest": small_frame.head(3)})
    src.sync_to(sink, key="dest", mode="overwrite")
    assert len(sink.read("dest")) == len(small_frame)


def test_sync_error_mode_raises_when_key_exists(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory({"dest": small_frame.head(3)})
    with pytest.raises(FileExistsError):
        src.sync_to(sink, key="dest", mode="error")


def test_sync_to_read_only_sink_raises(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory({"dest": small_frame.head(3)}, read_only=True)
    with pytest.raises(ReadOnlyError):
        src.sync_to(sink, key="dest")


def test_sync_with_window(small_frame: pd.DataFrame) -> None:
    src = Memory({"only": small_frame})
    sink = Memory()
    src.sync_to(sink, key="dest", start="2024-01-03", end="2024-01-07")
    out = sink.read("dest")
    assert out.index.min() == pd.Timestamp("2024-01-03")
    assert out.index.max() == pd.Timestamp("2024-01-07")


def test_sync_with_explicit_source_key(small_frame: pd.DataFrame) -> None:
    src = Memory({"alpha": small_frame, "beta": small_frame.iloc[5:]})
    sink = Memory()
    src.sync_to(sink, key="cached_alpha", source_key="alpha")
    pd.testing.assert_frame_equal(
        sink.read("cached_alpha").sort_index(axis=1),
        small_frame.sort_index(axis=1),
    )
    src.sync_to(sink, key="cached_beta", source_key="beta")
    assert len(sink.read("cached_beta")) == 5
