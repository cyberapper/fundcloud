"""Tests for ``fundcloud.datasets``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud import datasets
from fundcloud.datasets import available_datasets, load_example_panel


def test_available_datasets_returns_list() -> None:
    out = available_datasets()
    assert isinstance(out, list)


def test_load_example_panel_raises_when_missing() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_example_panel("definitely_does_not_exist")


def test_load_example_panel_reads_bundled_parquet(tmp_path: Path, monkeypatch) -> None:
    # Point the module at a tmp dir with a dummy parquet we control.
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D").values)
    frame = pd.DataFrame({"x": np.arange(5, dtype=float)}, index=idx)
    fake_data_dir = tmp_path / "_data"
    fake_data_dir.mkdir()
    frame.to_parquet(fake_data_dir / "sample.parquet")

    # Swap the module-level DATASET_DIR constant without reimporting the library.
    monkeypatch.setattr(datasets.loaders, "DATASET_DIR", fake_data_dir)

    assert "sample" in available_datasets()  # type: ignore[operator]
    # available_datasets() reads from the module-level dir, which we've just patched.
    names = [p.stem for p in fake_data_dir.glob("*.parquet")]
    assert "sample" in names

    loaded = load_example_panel("sample")
    pd.testing.assert_frame_equal(loaded, frame)
