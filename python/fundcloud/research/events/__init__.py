"""Causal, leak-free OHLCV event detectors for the event-study research engine.

This package holds the per-asset event detectors specified in
``docs/guides/research/event-registry.md`` plus their shared foundation:

* :mod:`fundcloud.research.events.schema` — the observation row schema, the
  frame assembler, the ``params_hash`` helper, and the projection that lets the
  existing :func:`fundcloud.metrics.feature_quality.evaluate` consume events
  unchanged.
* :mod:`fundcloud.research.events._causality` — Wilder's ATR, neighbour-locked
  pivot confirmation, and the mandatory prefix-invariance harness every
  detector must pass.

The package surface re-exports the observation schema + reuse projection, the
causality primitives, and the three detectors plus :func:`scan_panel`.
"""

from __future__ import annotations

from fundcloud.research.events._causality import (
    assert_prefix_invariant,
    confirmed_pivots,
    wilder_atr,
)
from fundcloud.research.events.detectors import (
    detect_displacement,
    detect_fvg,
    detect_sweep_fail,
    scan_panel,
)
from fundcloud.research.events.explore import (
    event_portfolio,
    evidence_table,
    forward_paths,
    outcome_profile,
    portfolio_by_event,
    return_distribution,
    tag_episodes,
)
from fundcloud.research.events.schema import (
    OBSERVATION_COLUMNS,
    build_observations,
    params_hash,
    to_events_frame,
)
from fundcloud.research.events.split import FrozenSplit, frozen_split
from fundcloud.research.events.study import (
    DEFAULT_GRIDS,
    STUDY_HORIZONS,
    Variant,
    default_variants,
    expand_grid,
    scan_variants,
)

__all__ = [
    "DEFAULT_GRIDS",
    "OBSERVATION_COLUMNS",
    "STUDY_HORIZONS",
    "FrozenSplit",
    "Variant",
    "assert_prefix_invariant",
    "build_observations",
    "confirmed_pivots",
    "default_variants",
    "detect_displacement",
    "detect_fvg",
    "detect_sweep_fail",
    "event_portfolio",
    "evidence_table",
    "expand_grid",
    "forward_paths",
    "frozen_split",
    "outcome_profile",
    "params_hash",
    "portfolio_by_event",
    "return_distribution",
    "scan_panel",
    "scan_variants",
    "tag_episodes",
    "to_events_frame",
    "wilder_atr",
]
