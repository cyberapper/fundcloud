"""Research utilities for the event-study engine.

Step 1 ships the data layer: leak-free extraction of clean daily OHLCV from the
ClickHouse ``md_ohlcv_data`` feed (:func:`load_bars`, :func:`load_universe`) and a
quality gate + cleaner (:func:`check_quality`, :func:`clean_panel`) to run before any
research touches the data.

The event layer adds the single-asset detectors (:func:`detect_displacement`,
:func:`detect_fvg`, :func:`detect_sweep_fail`), the :func:`scan_panel` helper that
runs a detector across a ``(field, symbol)`` panel, and the schema surface
(:data:`OBSERVATION_COLUMNS`, :func:`to_events_frame`) needed to feed detected
events to :func:`fundcloud.metrics.feature_quality.evaluate`.
"""

from __future__ import annotations

from fundcloud.research.events import (
    FULL_GRIDS,
    OBSERVATION_COLUMNS,
    FrozenSplit,
    Variant,
    count_variants,
    decode_params,
    default_variants,
    detect_displacement,
    detect_donchian,
    detect_fvg,
    detect_inside_bar,
    detect_key_reversal,
    detect_nr_squeeze,
    detect_opening_gap,
    detect_order_block,
    detect_sr_touch_bounce,
    detect_sweep_fail,
    event_portfolio,
    evidence_table,
    forward_paths,
    frozen_split,
    outcome_profile,
    portfolio_by_event,
    return_distribution,
    scan_panel,
    scan_variants,
    tag_episodes,
    to_events_frame,
    variant_leaderboard,
)
from fundcloud.research.loader import load_bars, load_universe
from fundcloud.research.quality import QualityReport, check_quality, clean_panel

__all__ = [
    "FULL_GRIDS",
    "OBSERVATION_COLUMNS",
    "FrozenSplit",
    "QualityReport",
    "Variant",
    "check_quality",
    "clean_panel",
    "count_variants",
    "decode_params",
    "default_variants",
    "detect_displacement",
    "detect_donchian",
    "detect_fvg",
    "detect_inside_bar",
    "detect_key_reversal",
    "detect_nr_squeeze",
    "detect_opening_gap",
    "detect_order_block",
    "detect_sr_touch_bounce",
    "detect_sweep_fail",
    "event_portfolio",
    "evidence_table",
    "forward_paths",
    "frozen_split",
    "load_bars",
    "load_universe",
    "outcome_profile",
    "portfolio_by_event",
    "return_distribution",
    "scan_panel",
    "scan_variants",
    "tag_episodes",
    "to_events_frame",
    "variant_leaderboard",
]
