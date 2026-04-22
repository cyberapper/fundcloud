"""Tests for :class:`fundcloud.data.Catalog`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.data import Catalog, DatasetSpec, Memory, Parquet
from fundcloud.data._base import BaseBackend


class _RecordingSource(BaseBackend):
    """A read-only source that records the ``start``/``end`` it sees."""

    name = "_recording"
    read_only = True

    def __init__(self, df: pd.DataFrame, *, symbols: Sequence[str] | None = None) -> None:
        self._df = df
        self.symbols = list(symbols) if symbols is not None else []
        self.interval = "1d"
        self.calls: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = []

    def read(
        self,
        key: str | None = None,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        norm_start = pd.Timestamp(start) if start is not None else None
        norm_end = pd.Timestamp(end) if end is not None else None
        self.calls.append((norm_start, norm_end))
        df = self._df
        if start is not None or end is not None:
            df = df.loc[start:end]  # type: ignore[misc]
        return df.copy()


@pytest.fixture
def small_panel() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10, freq="D").values)
    return pd.DataFrame(
        {"close": np.arange(10, dtype=float)},
        index=idx,
    )


def test_register_and_load(small_panel: pd.DataFrame) -> None:
    cat = Catalog(store=Memory())
    src = Memory({"x": small_panel})
    spec = cat.register("equity", src)
    assert isinstance(spec, DatasetSpec)
    assert "equity" in cat

    out = cat.load("equity")
    pd.testing.assert_frame_equal(out.sort_index(axis=1), small_panel.sort_index(axis=1))


def test_prefer_store_short_circuits_the_source(small_panel: pd.DataFrame) -> None:
    store = Memory()
    cat = Catalog(store=store)
    src = Memory({"_": small_panel})
    cat.register("equity", src)
    cat.load("equity")  # warms cache

    # Mutate the underlying source; the cached read should *not* see the change.
    src._data["_"] = src._data["_"] + 999.0

    got = cat.load("equity", prefer_store=True)
    assert (got["close"] == small_panel["close"].values).all()


def test_refresh_appends_new_rows_only(small_panel: pd.DataFrame) -> None:
    cat = Catalog(store=Memory())
    src = Memory({"_": small_panel})
    cat.register("equity", src)
    cat.load("equity")  # establish watermark

    new_rows = small_panel.head(5).copy()
    new_rows.index = pd.DatetimeIndex(
        pd.date_range(small_panel.index[-1] + pd.Timedelta(days=1), periods=5, freq="D").values
    )
    extended = pd.concat([small_panel, new_rows])
    src._data["_"] = extended

    fresh = cat.refresh("equity")

    # The refresh pulls from watermark forwards; upsert dedups the watermark row.
    # The total cache must equal the extended source.
    assert len(cat.load("equity")) == len(extended)
    # `fresh` is whatever sync_to wrote (start=watermark forward).
    assert fresh.index.min() >= small_panel.index[-1]


def test_register_duplicate_name_raises(small_panel: pd.DataFrame) -> None:
    cat = Catalog(store=Memory())
    cat.register("equity", Memory({"_": small_panel}))
    with pytest.raises(ValueError, match="already registered"):
        cat.register("equity", Memory({"_": small_panel}))


def test_describe_has_expected_columns(small_panel: pd.DataFrame) -> None:
    cat = Catalog(store=Memory())
    cat.register("equity", Memory({"_": small_panel}), tags=("equity", "us"))
    cat.load("equity")

    table = cat.describe()
    expected = {"name", "source", "symbols", "interval", "store_key", "last_index", "tags"}
    assert expected.issubset(set(table.columns))
    assert table.iloc[0]["name"] == "equity"


def test_to_and_from_spec_roundtrip(tmp_path: Path, small_panel: pd.DataFrame) -> None:
    parquet_root = tmp_path / "panels"
    parquet_root.mkdir()
    backend = Parquet(parquet_root)
    backend.write("equity", small_panel)

    cat = Catalog(store=Memory())
    cat.register("equity", backend)

    spec = cat.to_spec()
    assert "equity" in spec
    assert spec["equity"]["source"].endswith("Parquet")

    restored = Catalog.from_spec(Memory(), spec)
    assert "equity" in restored
    assert restored.spec("equity").store_key == "equity"


def test_refresh_all_tag_filter(small_panel: pd.DataFrame) -> None:
    cat = Catalog(store=Memory())
    cat.register("a", Memory({"_": small_panel}), tags=("x",))
    cat.register("b", Memory({"_": small_panel}), tags=("y",))
    cat.load("a")
    cat.load("b")

    only_x = cat.refresh_all(tags=("x",))
    assert set(only_x.keys()) == {"a"}


# ----------------------------------------------------------------- load period


def test_load_forwards_window_to_store(small_panel: pd.DataFrame) -> None:
    store = Memory()
    cat = Catalog(store=store)
    cat.register("equity", _RecordingSource(small_panel))
    cat.load("equity")  # warm cache; recording source called once with no start/end

    sliced = cat.load("equity", start="2024-01-03", end="2024-01-07")
    assert sliced.index.min() == pd.Timestamp("2024-01-03")
    assert sliced.index.max() == pd.Timestamp("2024-01-07")


def test_load_forwards_window_to_source_when_store_empty(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    cat = Catalog(store=Memory())
    cat.register("equity", src)
    cat.load("equity", start="2024-01-03", end="2024-01-07")
    assert src.calls[-1] == (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-07"))


def test_load_uses_refresh_kwargs_defaults(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    cat = Catalog(store=Memory())
    cat.register(
        "equity",
        src,
        refresh_kwargs={"start": "2024-01-04", "end": "2024-01-06"},
    )
    cat.load("equity")  # no call-site overrides → uses spec defaults
    assert src.calls[-1] == (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-06"))


def test_refresh_uses_refresh_kwargs_start_when_store_empty(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    cat = Catalog(store=Memory())
    cat.register("equity", src, refresh_kwargs={"start": "2024-01-05"})
    cat.refresh("equity")
    assert src.calls[-1][0] == pd.Timestamp("2024-01-05")


def test_refresh_uses_watermark_minus_lookback(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    store = Memory()
    cat = Catalog(store=store)
    cat.register("equity", src, refresh_kwargs={"lookback": "3D"})

    cat.load("equity")  # establish watermark
    pre_watermark_calls = len(src.calls)

    cat.refresh("equity")
    last_call_start = src.calls[-1][0]
    assert last_call_start == small_panel.index[-1] - pd.Timedelta("3D")
    assert len(src.calls) == pre_watermark_calls + 1


def test_refresh_default_lookback_is_zero(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    cat = Catalog(store=Memory())
    cat.register("equity", src)
    cat.load("equity")
    cat.refresh("equity")
    last_call_start = src.calls[-1][0]
    assert last_call_start == small_panel.index[-1]  # exactly the watermark, no lookback


def test_refresh_call_site_end_overrides_refresh_kwargs(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    cat = Catalog(store=Memory())
    cat.register("equity", src, refresh_kwargs={"end": "2024-01-08"})
    cat.refresh("equity", end="2024-01-05")
    assert src.calls[-1][1] == pd.Timestamp("2024-01-05")


def test_refresh_is_idempotent_under_upsert(small_panel: pd.DataFrame) -> None:
    src = _RecordingSource(small_panel)
    store = Memory()
    cat = Catalog(store=store)
    cat.register("equity", src, refresh_kwargs={"lookback": "3D"})

    cat.load("equity")  # initial pull
    first_len = len(store.read("equity"))
    cat.refresh("equity")
    assert len(store.read("equity")) == first_len  # upsert dedups; no row growth
