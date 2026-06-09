"""Tests for the variant grid + neutral multi-detector scan.

Grid expansion must be a deterministic Cartesian product; ``scan_variants`` must
pool every variant's detected events into one observation frame without
assigning any direction or performance, and stay empty when nothing fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.research.events._displacement import detect_displacement
from fundcloud.research.events._fvg import detect_fvg
from fundcloud.research.events.schema import OBSERVATION_COLUMNS, build_observations
from fundcloud.research.events.study import (
    DEFAULT_GRIDS,
    FULL_GRIDS,
    Variant,
    count_variants,
    decode_params,
    default_variants,
    expand_grid,
    scan_variants,
)


def test_expand_grid_is_deterministic_cartesian_product() -> None:
    variants = expand_grid("ev_disp_bar", detect_displacement, {"z_body": [1.0, 1.5], "clv_min": [0.7]})

    assert len(variants) == 2
    # Sorted-key order, so params tuples are ('clv_min', ...), ('z_body', ...).
    assert variants[0].params_dict == {"clv_min": 0.7, "z_body": 1.0}
    assert variants[1].params_dict == {"clv_min": 0.7, "z_body": 1.5}
    # Distinct params -> distinct hashes.
    assert variants[0].params_hash != variants[1].params_hash


def test_expand_grid_empty_grid_yields_one_variant() -> None:
    variants = expand_grid("ev_disp_bar", detect_displacement, {})

    assert len(variants) == 1
    assert variants[0].params_dict == {}


def test_default_variants_match_default_grids() -> None:
    # 10 detectors at 2 variants each, except ev_inside_bar (strict False/True = 2)
    # and the single-grid ones — count straight from the grids to stay in sync.
    assert len(default_variants()) == count_variants(DEFAULT_GRIDS)
    assert len(default_variants()) == 20


def _panel_with_bullish_fvg() -> pd.DataFrame:
    """A two-symbol panel with a bullish FVG after a long ATR-warmup run."""
    rows: list[tuple[float, float, float, float]] = [
        (100.0, 100.5, 99.5, 100.0) for _ in range(26)
    ]  # warmup long enough for evaluate's default ATR(14) at the gap bar
    rows.append((100.5, 110.0, 100.0, 109.5))  # t-1: wide up-candle (impulse)
    rows.append((110.0, 111.0, 101.0, 110.5))  # t: low > high[t-2] -> bullish gap
    rows.extend((112.0 + i, 113.0 + i, 111.0 + i, 112.5 + i) for i in range(15))  # forward path
    arr = np.asarray(rows, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(arr), freq="D", tz="UTC")

    data: dict[tuple[str, str], pd.Series] = {}
    for sym in ("AAA", "BBB"):
        data[("open", sym)] = pd.Series(arr[:, 0], index=idx)
        data[("high", sym)] = pd.Series(arr[:, 1], index=idx)
        data[("low", sym)] = pd.Series(arr[:, 2], index=idx)
        data[("close", sym)] = pd.Series(arr[:, 3], index=idx)
        data[("volume", sym)] = pd.Series(np.ones(len(arr)), index=idx)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)


def test_scan_variants_pools_observations_without_direction_mapping() -> None:
    panel = _panel_with_bullish_fvg()
    variant = Variant(
        event_id="ev_gap_imb_3c",
        detect=detect_fvg,
        params=(("body_min", 0.5), ("z_imp", 1.0)),
    )

    obs = scan_variants(panel, [variant])

    assert not obs.empty
    # It is a plain observation frame — no trade direction / performance columns.
    assert list(obs.columns) == list(OBSERVATION_COLUMNS)
    # Each geometric branch is its own event_id (the detection equation).
    assert set(obs["event_id"]) <= {"ev_gap_up", "ev_gap_dn"}
    # The base id used purely to key params_hash never appears as an event_id.
    assert (obs["event_id"] != "ev_gap_imb_3c").all()
    # Both branches of one detector call share ONE params_hash (keyed on the
    # detector base id) so the two _up / _dn rows never split a parameterisation.
    assert obs["params_hash"].nunique() == 1


def test_scan_variants_pools_multiple_detectors() -> None:
    panel = _panel_with_bullish_fvg()
    variants = [
        Variant(event_id="ev_gap_imb_3c", detect=detect_fvg, params=(("body_min", 0.5),)),
        Variant(event_id="ev_disp_bar", detect=detect_displacement, params=(("atr_n", 14),)),
    ]

    obs = scan_variants(panel, variants)

    assert set(obs["event_id"]) <= {"ev_gap_up", "ev_gap_dn", "ev_disp_up", "ev_disp_dn"}
    assert list(obs.columns) == list(OBSERVATION_COLUMNS)


def test_scan_variants_empty_when_nothing_fires() -> None:
    # A flat panel produces no events.
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    flat = np.column_stack([
        np.full(40, 100.0), np.full(40, 100.5), np.full(40, 99.5), np.full(40, 100.0)
    ])
    data: dict[tuple[str, str], pd.Series] = {
        ("open", "AAA"): pd.Series(flat[:, 0], index=idx),
        ("high", "AAA"): pd.Series(flat[:, 1], index=idx),
        ("low", "AAA"): pd.Series(flat[:, 2], index=idx),
        ("close", "AAA"): pd.Series(flat[:, 3], index=idx),
        ("volume", "AAA"): pd.Series(np.ones(40), index=idx),
    }
    panel = pd.DataFrame(data, index=idx)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    variant = Variant(event_id="ev_gap_imb_3c", detect=detect_fvg, params=())
    obs = scan_variants(panel, [variant])

    assert obs.empty
    assert list(obs.columns) == list(OBSERVATION_COLUMNS)


# --- params decode + grid policy -------------------------------------------


def test_decode_params_maps_hash_to_flattened_params() -> None:
    # Two FVG variants on one bullish gap -> two params_hash for ev_gap_up; decode
    # must give one readable row per hash with body_min as its own column.
    panel = _panel_with_bullish_fvg()
    variants = [
        Variant(event_id="ev_gap_imb_3c", detect=detect_fvg, params=(("body_min", 0.5),)),
        Variant(event_id="ev_gap_imb_3c", detect=detect_fvg, params=(("body_min", 0.6),)),
    ]
    obs = scan_variants(panel, variants)
    assert obs["params_hash"].nunique() == 2  # two parameterisations stay separate

    decoded = decode_params(obs)

    assert len(decoded) == 2  # one row per params_hash
    assert "params_hash" in decoded.columns
    assert {"body_min", "z_imp", "atr_n"} <= set(decoded.columns)
    assert set(decoded["body_min"]) == {0.5, 0.6}


def test_decode_params_collapses_up_dn_to_one_row() -> None:
    # Both geometric branches share one params_hash + one params dict, so a panel
    # firing only one branch still yields exactly one decoded row per hash.
    panel = _panel_with_bullish_fvg()
    obs = scan_variants(panel, [Variant(event_id="ev_gap_imb_3c", detect=detect_fvg, params=())])
    decoded = decode_params(obs)
    assert len(decoded) == obs["params_hash"].nunique() == 1


def test_decode_params_none_valued_param_is_nan_clean() -> None:
    # A displacement variant leaves z_vol=None (gate off). decode must surface a
    # z_vol column that is NaN, with no coercion warning (filterwarnings=error).
    panel = _panel_with_bullish_fvg()  # the wide up-candle also fires displacement
    obs = scan_variants(
        panel, [Variant(event_id="ev_disp_bar", detect=detect_displacement, params=())]
    )
    assert not obs.empty

    decoded = decode_params(obs)

    assert "z_vol" in decoded.columns
    assert decoded["z_vol"].isna().all()


def test_decode_params_heterogeneous_keys_union() -> None:
    # Pooling displacement + FVG unions their param keys; the cell a detector does
    # not use reads NaN.
    panel = _panel_with_bullish_fvg()
    obs = scan_variants(
        panel,
        [
            Variant(event_id="ev_disp_bar", detect=detect_displacement, params=()),
            Variant(event_id="ev_gap_imb_3c", detect=detect_fvg, params=(("body_min", 0.5),)),
        ],
    )

    decoded = decode_params(obs)

    assert {"z_body", "body_min"} <= set(decoded.columns)
    # The FVG row has no z_body; the displacement row has no body_min.
    assert decoded["z_body"].isna().any()
    assert decoded["body_min"].isna().any()


def test_decode_params_empty_obs() -> None:
    decoded = decode_params(build_observations([]))
    assert decoded.empty
    assert list(decoded.columns) == ["params_hash"]


def test_count_variants_is_cartesian_product() -> None:
    assert count_variants(DEFAULT_GRIDS) == 20
    assert count_variants(FULL_GRIDS) == 582
    # An empty grid counts as one parameter-free variant.
    assert count_variants({"x": (detect_fvg, {})}) == 1
    assert count_variants({"x": (detect_fvg, {"a": [1, 2, 3], "b": [1, 2]})}) == 6


def test_full_grids_is_superset_of_defaults() -> None:
    # Every default detector is present in FULL_GRIDS, and for each shared param the
    # full value list contains the default's values (FULL widens, never narrows).
    assert set(DEFAULT_GRIDS) <= set(FULL_GRIDS)
    for base_id, (_detect, default_grid) in DEFAULT_GRIDS.items():
        full_grid = FULL_GRIDS[base_id][1]
        for key, default_vals in default_grid.items():
            assert set(default_vals) <= set(full_grid[key])
    # Spot-check the registry-canonical value lists guard against drift.
    assert FULL_GRIDS["ev_disp_bar"][1]["z_body"] == [0.8, 1.0, 1.25, 1.5, 2.0]
    assert FULL_GRIDS["ev_sweep_fail"][1]["eps"] == [0.0, 0.05, 0.10, 0.25, 0.50]


def test_max_variants_guard_raises_before_mining() -> None:
    # The wide grid is an explicit choice: a low cap fails loud rather than expand.
    with pytest.raises(ValueError, match="582 variants"):
        default_variants(FULL_GRIDS, max_variants=100)
    # A generous cap lets it through; no cap on the default is fine.
    assert len(default_variants(FULL_GRIDS, max_variants=1000)) == 582
    assert len(default_variants()) == 20
