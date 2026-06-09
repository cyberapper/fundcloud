"""Detector registry + the panel-scan helper.

Re-exports the three single-asset detectors so callers import them from one
place, and adds :func:`scan_panel` to run any detector across every symbol of a
canonical ``(field, symbol)`` MultiIndex panel (as produced by
:func:`fundcloud.research.load_bars`) and concatenate the observation frames.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from fundcloud.research.events._displacement import detect_displacement
from fundcloud.research.events._fvg import detect_fvg
from fundcloud.research.events._sweep import detect_sweep_fail
from fundcloud.research.events.schema import build_observations

__all__ = [
    "detect_displacement",
    "detect_fvg",
    "detect_sweep_fail",
    "scan_panel",
]


def scan_panel(
    panel: pd.DataFrame,
    detect: Callable[..., pd.DataFrame],
    **params: Any,
) -> pd.DataFrame:
    """Run a detector over every symbol of a ``(field, symbol)`` panel.

    Parameters
    ----------
    panel
        Canonical OHLCV panel with ``(field, symbol)`` MultiIndex columns and a
        tz-aware UTC DatetimeIndex, as produced by
        :func:`fundcloud.research.load_bars`.
    detect
        A single-asset detector ``detect(bars, *, asset=..., **params)`` returning
        an observation frame.
    **params
        Forwarded verbatim to ``detect``.

    Returns
    -------
    pandas.DataFrame
        The per-symbol observation frames concatenated into one. Empty input
        yields an empty observation frame.
    """
    if panel.empty or not isinstance(panel.columns, pd.MultiIndex):
        return build_observations([])

    symbols = panel.columns.get_level_values(-1).unique()
    frames = [
        detect(panel.xs(sym, level=-1, axis=1), asset=str(sym), **params) for sym in symbols
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return build_observations([])
    return pd.concat(frames, ignore_index=True)
