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
    detect_donchian,
    detect_fvg,
    detect_inside_bar,
    detect_key_reversal,
    detect_nr_squeeze,
    detect_opening_gap,
    detect_order_block,
    detect_sr_touch_bounce,
    detect_sweep_fail,
    scan_panel,
)
from fundcloud.research.events.schema import build_observations, params_hash

__all__ = [
    "DEFAULT_GRIDS",
    "FULL_GRIDS",
    "STUDY_HORIZONS",
    "Variant",
    "count_variants",
    "decode_params",
    "default_variants",
    "expand_grid",
    "scan_variants",
]

#: Pre-registered forward horizons (bars) the evidence views score at, from the
#: registry grid (``docs/guides/research/event-registry.md``). Fixed before mining.
STUDY_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 40, 60)

#: Per-detector parameter grids — a deliberately small slice of each detector's
#: pre-registered grid so the default run stays tractable (see :data:`FULL_GRIDS`
#: for the registry-wide grids). Each value is ``(detector, grid)`` where ``grid``
#: maps a kwarg name to the values swept. Keyed by the detector **base id** (the
#: same id the detector feeds to ``params_hash``) — so a variant's ``params_hash``
#: aligns with the per-branch (``_up`` / ``_dn``) observations the detector pools
#: under one shared hash. Neutral detectors (``ev_inside_bar``, ``ev_nr_squeeze``)
#: emit a single suffix-less id but key the hash on the same base id identically.
DEFAULT_GRIDS: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, list[Any]]]] = {
    "ev_disp_bar": (detect_displacement, {"z_body": [1.0, 1.5], "clv_min": [0.7]}),
    "ev_gap_imb_3c": (detect_fvg, {"body_min": [0.5, 0.6], "z_imp": [1.0]}),
    "ev_sweep_fail": (detect_sweep_fail, {"pivot_k": [3, 5], "eps": [0.10]}),
    "ev_donchian_break": (detect_donchian, {"N": [20, 40], "buf": [0.0]}),
    "ev_outside_reversal": (detect_key_reversal, {"clv_min": [0.6, 0.7]}),
    "ev_opening_gap": (detect_opening_gap, {"k": [0.5, 1.0]}),
    "ev_inside_bar": (detect_inside_bar, {"strict": [False, True]}),
    "ev_nr_squeeze": (detect_nr_squeeze, {"n": [4, 7]}),
    "ev_sr_touch_bounce": (detect_sr_touch_bounce, {"pivot_k": [3, 5], "eps": [0.10]}),
    "ev_ob_impulse_last_opp": (detect_order_block, {"m": [3, 5]}),
}

#: The **full** pre-registered grids from the registry (``event-registry.md``
#: §"Pre-registered parameter grids" + the per-event spec blocks). Same shape as
#: :data:`DEFAULT_GRIDS` but sweeping the registry's complete value lists, so a
#: serious scan can tune every parameter. It is *not* the default precisely
#: because it is large: expanding it is a deliberate choice (each variant is a
#: separate trial, and more trials inflate the multiple-testing burden the engine
#: is built to respect). Use :func:`count_variants` to see the trial count first
#: and the ``max_variants`` guard on :func:`default_variants` to fail loud rather
#: than mine thousands of trials by accident. ``clv_min`` follows the displacement
#: spec block's ``{0.6, 0.7, 0.8}`` (it has no row in the summary grid table).
FULL_GRIDS: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, list[Any]]]] = {
    "ev_disp_bar": (
        detect_displacement,
        {
            "z_body": [0.8, 1.0, 1.25, 1.5, 2.0],
            "clv_min": [0.6, 0.7, 0.8],
            "z_vol": [None, 1.0, 1.25, 1.5, 2.0],
            "atr_n": [10, 14, 20],
        },
    ),
    "ev_gap_imb_3c": (
        detect_fvg,
        {
            "body_min": [0.5, 0.6, 0.7, 0.8],
            "z_imp": [0.0, 1.0, 1.25, 1.5],
            "atr_n": [10, 14, 20],
        },
    ),
    "ev_sweep_fail": (
        detect_sweep_fail,
        {
            "pivot_k": [2, 3, 5, 10],
            "eps": [0.0, 0.05, 0.10, 0.25, 0.50],
            "atr_n": [10, 14, 20],
        },
    ),
    "ev_donchian_break": (
        detect_donchian,
        {
            "N": [10, 20, 40, 60, 120, 252],
            "buf": [0.0, 0.05, 0.10, 0.25],
            "atr_n": [10, 14, 20],
        },
    ),
    "ev_outside_reversal": (
        detect_key_reversal,
        {"clv_min": [0.6, 0.7, 0.8], "atr_n": [10, 14, 20]},
    ),
    "ev_opening_gap": (
        detect_opening_gap,
        {"k": [0.25, 0.5, 1.0, 1.5], "atr_n": [10, 14, 20]},
    ),
    "ev_inside_bar": (
        detect_inside_bar,
        {"strict": [False, True], "atr_n": [10, 14, 20]},
    ),
    "ev_nr_squeeze": (
        detect_nr_squeeze,
        {"n": [4, 7, 10], "atr_n": [10, 14, 20]},
    ),
    "ev_sr_touch_bounce": (
        detect_sr_touch_bounce,
        {
            "pivot_k": [2, 3, 5, 10],
            "eps": [0.0, 0.05, 0.10, 0.25, 0.50],
            "atr_n": [10, 14, 20],
        },
    ),
    "ev_ob_impulse_last_opp": (
        detect_order_block,
        {
            "m": [3, 5, 10],
            "r": [1, 2, 3],
            "z_body": [1.0, 1.25, 1.5],
            "atr_n": [10, 14, 20],
        },
    ),
}


def count_variants(
    grids: Mapping[str, tuple[Callable[..., pd.DataFrame], Mapping[str, Sequence[Any]]]],
) -> int:
    """Total variant (trial) count a grid set expands to, *before* scanning.

    Each detector contributes the Cartesian-product size of its grid (an empty
    grid counts as one parameter-free variant, matching :func:`expand_grid`).
    Knowing the count up front is the precondition for honest multiple-testing
    correction — call this before :func:`default_variants` / :func:`scan_variants`
    to see the burden a wide grid (e.g. :data:`FULL_GRIDS`) implies.

    Parameters
    ----------
    grids
        A mapping of detector base id to ``(detector, grid)``, shaped like
        :data:`DEFAULT_GRIDS` / :data:`FULL_GRIDS`.

    Returns
    -------
    int
        The number of variants the grids expand to.
    """
    total = 0
    for _detect, grid in grids.values():
        combos = 1
        for values in grid.values():
            combos *= len(values)
        total += combos
    return total


def decode_params(obs: pd.DataFrame) -> pd.DataFrame:
    """Map each ``params_hash`` in an observation frame back to its params.

    Every view groups by ``params_hash`` (a SHA-1 digest) so distinct
    parameterisations stay separate — but a digest is unreadable. This recovers
    the human-readable parameters by deduping the observation frame's own
    ``params`` column (no recomputation): one row per ``params_hash`` with the
    ``params`` dict flattened into one column per parameter. Both ``_up`` / ``_dn``
    branches of a detector call share one hash *and* one params dict, so they
    collapse to a single row correctly. Joining this onto an evidence table (see
    :func:`fundcloud.research.events.variant_leaderboard`) is what lets a human
    read which parameters a row used.

    Parameters
    ----------
    obs
        Observation frame (:func:`scan_variants` output) carrying ``params`` and
        ``params_hash``.

    Returns
    -------
    pandas.DataFrame
        One row per distinct ``params_hash`` with a ``params_hash`` column plus
        one column per parameter key (union across detectors; a key absent for a
        detector — or an off optional gate like ``z_vol=None`` — reads as ``NaN``,
        never coerced). Param columns are sorted for determinism. Empty / params-
        free input yields a frame with just ``params_hash``.
    """
    if obs.empty:
        return pd.DataFrame(columns=["params_hash"])
    sub = obs.loc[obs["params"].notna(), ["params_hash", "params"]]
    if sub.empty:
        return pd.DataFrame(columns=["params_hash"])
    first = sub.groupby("params_hash", sort=True)["params"].first()
    flat = pd.DataFrame(list(first.to_numpy()), index=first.index)
    flat = flat.reindex(sorted(flat.columns), axis=1)
    flat.insert(0, "params_hash", first.index.to_numpy())
    return flat.reset_index(drop=True)


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


def default_variants(
    grids: Mapping[str, tuple[Callable[..., pd.DataFrame], Mapping[str, Sequence[Any]]]]
    | None = None,
    *,
    max_variants: int | None = None,
) -> list[Variant]:
    """Expand a grid set into a flat variant list, with an optional trial cap.

    Parameters
    ----------
    grids
        The grid set to expand. Defaults to the tractable :data:`DEFAULT_GRIDS`;
        pass :data:`FULL_GRIDS` (or a custom dict of the same shape) to tune more
        parameters. This is how a caller "tunes parameters in scanning" — by
        choosing the grid, never by silently widening the default.
    max_variants
        If set and the grids expand to more than this many variants, raise
        :class:`ValueError` rather than mining the trials. A guard against an
        accidental blow-up (each variant is a separate trial — more trials inflate
        the multiple-testing burden). Call :func:`count_variants` to see the count
        beforehand.

    Returns
    -------
    list[Variant]
        One variant per parameter combination across all detectors.

    Raises
    ------
    ValueError
        If ``max_variants`` is set and the expansion exceeds it.
    """
    chosen = DEFAULT_GRIDS if grids is None else grids
    total = count_variants(chosen)
    if max_variants is not None and total > max_variants:
        msg = (
            f"grid expansion yields {total} variants, exceeding max_variants="
            f"{max_variants}; pass a smaller grid or raise the cap (each variant is "
            f"a separate trial, so more trials inflate the multiple-testing burden)"
        )
        raise ValueError(msg)
    variants: list[Variant] = []
    for event_id, (detect, grid) in chosen.items():
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
