"""Variant grid expansion + a neutral multi-detector scan.

This layer turns detectors into raw observations to feed the **evidence** views
in :mod:`fundcloud.research.events.explore`. It expands a pre-registered
parameter grid into concrete :class:`Variant` objects and runs them across a
``(field, symbol)`` panel, pooling every detected event into one observation
frame.

It deliberately assigns **no trade direction and computes no performance**. Each
detector splits its two geometric branches into distinct ``event_id``\\ s (the
``_up`` / ``_dn`` suffix *is* the detection equation, not a long/short mandate);
whether an event implies long, short, or nothing is decided downstream from the
*measured* forward path (see :func:`fundcloud.research.events.outcome_profile`
and ``side="auto"`` in :func:`fundcloud.research.events.event_portfolio`).

The grids are fixed up front (:data:`DEFAULT_GRIDS`) so the trial count is known
before mining — the precondition for honest multiple-testing correction later.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fundcloud.research.events.detectors import (
    detect_displacement,
    detect_fvg,
    detect_sweep_fail,
    scan_panel,
)
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = [
    "DEFAULT_GRIDS",
    "STUDY_HORIZONS",
    "Variant",
    "default_variants",
    "expand_grid",
    "scan_variants",
]

#: Pre-registered forward horizons (bars) the evidence views score at, from the
#: registry grid (``docs/guides/research/event-registry.md``). Fixed before mining.
STUDY_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 40, 60)

#: First-batch parameter grids per detector — a deliberately small slice of the
#: registry's pre-registered grids so the initial run stays tractable. Each value
#: is ``(detector, grid)`` where ``grid`` maps a kwarg name to the values swept.
#: Keyed by the detector **base id** (``ev_disp_bar`` / ``ev_gap_imb_3c`` /
#: ``ev_sweep_fail``) — the same id the detector feeds to ``params_hash`` — so a
#: variant's ``params_hash`` aligns with the per-branch (``_up`` / ``_dn``)
#: observations the detector pools under one shared hash.
DEFAULT_GRIDS: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, list[Any]]]] = {
    "ev_disp_bar": (detect_displacement, {"z_body": [1.0, 1.5], "clv_min": [0.7]}),
    "ev_gap_imb_3c": (detect_fvg, {"body_min": [0.5, 0.6], "z_imp": [1.0]}),
    "ev_sweep_fail": (detect_sweep_fail, {"pivot_k": [3, 5], "eps": [0.10]}),
}


@dataclass(frozen=True)
class Variant:
    """One fully-resolved detector configuration to score.

    Attributes
    ----------
    event_id
        Detector **base id** (e.g. ``"ev_gap_imb_3c"``) — the id the detector
        feeds to ``params_hash``. It is used here *only* to compute
        :attr:`params_hash`; it never appears as an emitted observation
        ``event_id`` (the detector writes the per-branch ``_up`` / ``_dn`` ids,
        both carrying this variant's shared ``params_hash``).
    detect
        The single-asset detector callable.
    params
        The resolved kwargs forwarded to ``detect`` (and to ``scan_panel``).
    """

    event_id: str
    detect: Callable[..., pd.DataFrame]
    params: tuple[tuple[str, Any], ...]

    @property
    def params_dict(self) -> dict[str, Any]:
        """The params as a plain dict (the tuple form keeps the variant hashable)."""
        return dict(self.params)

    @property
    def _resolved_params(self) -> dict[str, Any]:
        """The full params dict the detector folds into ``params_hash``.

        A detector hashes its swept kwargs *plus* the signature defaults it
        injects (e.g. ``atr_n``, and ``z_vol`` for displacement), so a variant
        that sweeps only a subset must bind the same defaults to land on the
        same digest. ``asset`` and ``logic_version`` are excluded because the
        detector never folds them into its hashed ``params`` dict.
        """
        resolved = {
            name: param.default
            for name, param in inspect.signature(self.detect).parameters.items()
            if param.default is not inspect.Parameter.empty
            and name not in ("asset", "logic_version")
        }
        resolved.update(self.params_dict)
        return resolved

    @property
    def params_hash(self) -> str:
        """Stable digest identifying this variant for grouping/pooling.

        Computed over the detector-resolved params (:attr:`_resolved_params`) and
        the detector **base id** (:attr:`event_id`), so it equals the
        ``params_hash`` the detector stamps onto this variant's emitted
        observations — letting callers join variants back to their detections.
        """
        p = self.params_dict
        return params_hash(self.event_id, self._resolved_params, int(p.get("logic_version", 1)))


def expand_grid(
    event_id: str,
    detect: Callable[..., pd.DataFrame],
    grid: Mapping[str, Sequence[Any]],
) -> list[Variant]:
    """Expand a parameter grid into the Cartesian product of :class:`Variant`.

    Parameters
    ----------
    event_id
        Catalog id stamped onto every variant.
    detect
        The detector callable shared by every variant.
    grid
        Maps each kwarg name to the list of values to sweep. An empty grid yields
        a single parameter-free variant.

    Returns
    -------
    list[Variant]
        One variant per combination, in deterministic (sorted-key) order.
    """
    keys = sorted(grid)
    if not keys:
        return [Variant(event_id=event_id, detect=detect, params=())]
    combos = itertools.product(*(grid[k] for k in keys))
    return [
        Variant(event_id=event_id, detect=detect, params=tuple(zip(keys, combo, strict=True)))
        for combo in combos
    ]


def default_variants() -> list[Variant]:
    """Expand :data:`DEFAULT_GRIDS` into the full first-batch variant list."""
    variants: list[Variant] = []
    for event_id, (detect, grid) in DEFAULT_GRIDS.items():
        variants.extend(expand_grid(event_id, detect, grid))
    return variants


def scan_variants(
    panel: pd.DataFrame,
    variants: Sequence[Variant] | None = None,
) -> pd.DataFrame:
    """Run every variant's detector across the panel and pool the observations.

    A neutral detection step: it assigns no direction and computes no
    performance. Each detected event carries its own per-branch ``event_id``
    (``_up`` / ``_dn``) and a ``params_hash`` (written by the detector; both
    branches of one call share one hash, keyed on the detector base id), so
    distinct parameterisations stay distinguishable in the pooled frame even
    though the two branches share a hash. The result feeds the evidence views in
    :mod:`fundcloud.research.events.explore`, which group by
    ``(event_id, params_hash)``.

    Parameters
    ----------
    panel
        Canonical ``(field, symbol)`` OHLCV panel (cleaned), as produced by
        :func:`fundcloud.research.load_bars` + :func:`fundcloud.research.clean_panel`.
    variants
        Variants to scan. Defaults to :func:`default_variants`.

    Returns
    -------
    pandas.DataFrame
        Observation frame (:data:`fundcloud.research.events.OBSERVATION_COLUMNS`)
        pooling every variant's events. Empty when nothing fired.
    """
    specs = list(variants) if variants is not None else default_variants()
    frames = [scan_panel(panel, v.detect, **v.params_dict) for v in specs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return build_observations([])
    return pd.concat(frames, ignore_index=True)
