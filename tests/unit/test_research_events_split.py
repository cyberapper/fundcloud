"""Tests for the frozen train / validation / holdout split.

The split must be a clean half-open partition (every row in exactly one bucket),
expose train + validation as the only selectable surface, and refuse degenerate
or malformed inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fundcloud.research.events.schema import build_observations
from fundcloud.research.events.split import FrozenSplit, frozen_split


def _obs(dates: list[str]) -> pd.DataFrame:
    """Observation frame with one row per ``confirmed_ts`` date."""
    rows = [
        {
            "event_id": "ev_disp_up",
            "asset": "AAA",
            "timeframe": "1D",
            "formation_end_ts": pd.Timestamp(d, tz="UTC"),
            "confirmed_ts": pd.Timestamp(d, tz="UTC"),
            "execution_ts": pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1),
            "params": {},
            "logic_version": 1,
            "params_hash": "h",
            "entry_ref_price": 1.0,
            "stop_ref_price": float("nan"),
            "zone_lo": float("nan"),
            "zone_hi": float("nan"),
            "quality": float("nan"),
            "atr_at_confirm": 1.0,
        }
        for d in dates
    ]
    return build_observations(rows)


def test_half_open_partition_is_disjoint_and_complete() -> None:
    obs = _obs(["2017-06-01", "2018-12-31", "2019-01-01", "2021-06-01", "2022-01-01", "2023-01-01"])

    split = frozen_split(obs, train_end="2019-01-01", val_end="2022-01-01")

    # train: ts < 2019-01-01 ; val: [2019-01-01, 2022-01-01) ; holdout: >= 2022-01-01
    assert split.sizes == {"train": 2, "val": 2, "holdout": 2}
    assert len(split.train) + len(split.val) + len(split.holdout) == len(obs)
    # The 2019-01-01 boundary row lands in val (left-closed), not train.
    assert (split.val["confirmed_ts"].min()) == pd.Timestamp("2019-01-01", tz="UTC")
    # The 2022-01-01 boundary row lands in holdout (left-closed), not val.
    assert (split.holdout["confirmed_ts"].min()) == pd.Timestamp("2022-01-01", tz="UTC")


def test_discovery_is_train_plus_val_only() -> None:
    obs = _obs(["2018-01-01", "2020-01-01", "2023-01-01"])

    split = frozen_split(obs, train_end="2019-01-01", val_end="2022-01-01")

    assert len(split.discovery) == len(split.train) + len(split.val)
    # Holdout rows never appear in the discovery surface.
    holdout_ts = set(split.holdout["confirmed_ts"])
    assert holdout_ts.isdisjoint(set(split.discovery["confirmed_ts"]))


def test_empty_input_yields_three_empty_frames() -> None:
    obs = build_observations([])

    split = frozen_split(obs, train_end="2019-01-01", val_end="2022-01-01")

    assert split.sizes == {"train": 0, "val": 0, "holdout": 0}
    assert isinstance(split, FrozenSplit)


def test_split_on_breakout_ts_column() -> None:
    obs = _obs(["2018-01-01", "2023-01-01"])
    events = obs.rename(columns={"confirmed_ts": "breakout_ts"})

    split = frozen_split(events, train_end="2019-01-01", val_end="2022-01-01", ts_col="breakout_ts")

    assert split.sizes == {"train": 1, "val": 0, "holdout": 1}


def test_missing_ts_col_raises() -> None:
    obs = _obs(["2018-01-01"]).drop(columns=["confirmed_ts"])

    with pytest.raises(KeyError, match="confirmed_ts"):
        frozen_split(obs, train_end="2019-01-01", val_end="2022-01-01")


def test_non_increasing_cuts_raise() -> None:
    obs = _obs(["2018-01-01"])

    with pytest.raises(ValueError, match="strictly before"):
        frozen_split(obs, train_end="2022-01-01", val_end="2019-01-01")


# --- purge + embargo (FIX 4) -----------------------------------------------


def test_purge_drops_train_event_whose_label_window_crosses_boundary() -> None:
    # Cut at 2019-01-07 (Mon). A train event confirmed on 2019-01-02 (Wed) with a
    # horizon=3 label window reaches ~3 business days forward into the validation
    # region, so it must be purged from train and appear in NO bucket. An older
    # train event (2018-06-01), far from the cut, survives.
    obs = _obs(["2018-06-01", "2019-01-02", "2020-06-01", "2023-01-01"])

    plain = frozen_split(obs, train_end="2019-01-07", val_end="2022-01-01")
    purged = frozen_split(obs, train_end="2019-01-07", val_end="2022-01-01", horizon=3)

    # Without purge the 2019-01-02 event is a normal train row.
    assert plain.sizes == {"train": 2, "val": 1, "holdout": 1}
    # With purge it is dropped from train entirely (not reassigned to val).
    assert purged.sizes == {"train": 1, "val": 1, "holdout": 1}
    train_ts = set(purged.train["confirmed_ts"])
    assert pd.Timestamp("2019-01-02", tz="UTC") not in train_ts
    assert pd.Timestamp("2018-06-01", tz="UTC") in train_ts
    # The purged row is gone from every bucket: train + val + holdout < input rows.
    assert len(purged.train) + len(purged.val) + len(purged.holdout) == len(obs) - 1
    val_ts = set(purged.val["confirmed_ts"]) | set(purged.holdout["confirmed_ts"])
    assert pd.Timestamp("2019-01-02", tz="UTC") not in val_ts


def test_embargo_widens_the_purge_band() -> None:
    # 2018-12-31 (Mon) is 4 business days before the 2019-01-07 (Mon) cut. With
    # horizon=2 it survives (band = 2 BDays), but adding embargo=3 widens the band
    # to 5 BDays and purges it.
    obs = _obs(["2018-12-31", "2020-06-01", "2023-01-01"])

    kept = frozen_split(obs, train_end="2019-01-07", val_end="2022-01-01", horizon=2)
    dropped = frozen_split(
        obs, train_end="2019-01-07", val_end="2022-01-01", horizon=2, embargo=3
    )

    assert pd.Timestamp("2018-12-31", tz="UTC") in set(kept.train["confirmed_ts"])
    assert pd.Timestamp("2018-12-31", tz="UTC") not in set(dropped.train["confirmed_ts"])
    assert len(dropped.train) == 0


def test_horizon_zero_reproduces_plain_partition() -> None:
    obs = _obs(["2017-06-01", "2018-12-31", "2019-01-01", "2021-06-01", "2022-01-01"])

    plain = frozen_split(obs, train_end="2019-01-01", val_end="2022-01-01")
    explicit = frozen_split(
        obs, train_end="2019-01-01", val_end="2022-01-01", horizon=0, embargo=0
    )

    assert plain.sizes == explicit.sizes
    assert plain.sizes == {"train": 2, "val": 2, "holdout": 1}


def test_negative_horizon_raises() -> None:
    obs = _obs(["2018-01-01"])

    with pytest.raises(ValueError, match="non-negative"):
        frozen_split(obs, train_end="2019-01-01", val_end="2022-01-01", horizon=-1)


def test_val_event_label_window_crossing_into_holdout_is_purged() -> None:
    # Symmetric purge against val_end: a validation event confirmed on 2022-01-04
    # (Tue) with horizon=3 has its label window land on val_end (2022-01-07 Fri),
    # i.e. it touches the holdout, so it must be purged from val. A far-from-cut
    # val event (2020-06-01) survives.
    obs = _obs(["2020-06-01", "2022-01-04", "2023-01-01"])

    purged = frozen_split(
        obs, train_end="2019-01-07", val_end="2022-01-07", horizon=3
    )

    val_ts = set(purged.val["confirmed_ts"])
    assert pd.Timestamp("2022-01-04", tz="UTC") not in val_ts
    assert pd.Timestamp("2020-06-01", tz="UTC") in val_ts
    # And it did not leak into holdout either — purged rows vanish.
    holdout_ts = set(purged.holdout["confirmed_ts"])
    assert pd.Timestamp("2022-01-04", tz="UTC") not in holdout_ts
