"""Tests for the variant grid + neutral multi-detector scan.

Grid expansion must be a deterministic Cartesian product; ``scan_variants`` must
pool every variant's detected events into one observation frame without
assigning any direction or performance, and stay empty when nothing fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._displacement import detect_displacement
from fundcloud.research.events._fvg import detect_fvg
from fundcloud.research.events.schema import OBSERVATION_COLUMNS
from fundcloud.research.events.study import (
    Variant,
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
    # 2 (displacement) + 2 (fvg) + 2 (sweep) = 6.
    assert len(default_variants()) == 6


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
