"""``PatternIndicator`` — sklearn-compatible base for chart-pattern features.

Pattern indicators subclass :class:`fundcloud.features.indicators.IndicatorSpec`
to keep the ``fit`` / ``transform`` contract identical to TA-Lib indicators.
The detection itself runs in Rust via :mod:`fundcloud._core.scan_pattern`.

Each subclass declares a ``pattern_name`` (matches the
:class:`Pattern` enum value) and ships a sensible ``condition`` preset.
Per-instance overrides happen via ``__init__(condition=...)``.

Inputs / outputs:

* ``inputs = ("open", "high", "low", "close", "volume")`` — OHLCV
  required, hence MultiIndex ``Bars`` frames are the supported shape.
* ``outputs = ("signal",)`` — single per-bar float column. The events
  table (with pivots, target / stop, quality) is available via
  :meth:`events`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from fundcloud import _core  # type: ignore[attr-defined]
from fundcloud.features.indicators.base import IndicatorSpec
from fundcloud.features.patterns._condition import PatternCondition
from fundcloud.features.patterns._enums import SignalMode, coerce
from fundcloud.features.patterns._events import (
    EVENTS_COLUMNS,
    build_events_frame,
    events_to_signal,
)

__all__ = [
    "DEFAULT_PIVOT_TIERS",
    "PIVOT_TIER_LONG",
    "PIVOT_TIER_MEDIUM",
    "PIVOT_TIER_SHORT",
    "PatternIndicator",
]

# Pivot scales — one effective ``order`` per timeframe tier.
#
# ``argrel`` pivots at a smaller order are a strict superset of those at
# a larger order, so multi-element tiers like ``(3, 5, 8)`` collapse to
# ``(3,)`` after the multi-level dedup step — orders 5 and 8 contribute
# nothing inside the same tier. Keep one integer per tier; rely on the
# *tiers themselves* (run as independent scans, then unioned) to expose
# patterns at multiple horizons.
#
# Rough mental model on dailies:
#   SHORT  (order=3)  → ~weekly swings
#   MEDIUM (order=13) → ~monthly swings
#   LONG   (order=34) → multi-month swings
PIVOT_TIER_SHORT: tuple[int, ...] = (3,)
PIVOT_TIER_MEDIUM: tuple[int, ...] = (13,)
PIVOT_TIER_LONG: tuple[int, ...] = (34,)
DEFAULT_PIVOT_TIERS: tuple[tuple[int, ...], ...] = (
    PIVOT_TIER_SHORT,
    PIVOT_TIER_MEDIUM,
    PIVOT_TIER_LONG,
)


class PatternIndicator(IndicatorSpec):
    """Base class for every chart-pattern indicator.

    Subclasses set ``pattern_name`` (snake_case, matching the
    :class:`Pattern` enum value) and a default ``condition`` preset; the
    rest of the pipeline (input fields, output schema, scan invocation,
    event projection) is shared.
    """

    inputs: ClassVar[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
    outputs: ClassVar[tuple[str, ...]] = ("signal",)
    default_params: ClassVar[dict[str, Any]] = {
        # 0.0 = surface every detection that passes the geometric gates;
        # callers filter by quality themselves when they have a reason to.
        # Quality is a "textbookness" score — its correlation to forward
        # returns is empirical, so we don't pre-filter on it.
        "min_quality": 0.0,
        # Single-tier order. Used when ``pivot_tiers`` is empty/None or
        # when callers explicitly override. Multi-element tuples
        # (e.g. ``(3, 5, 8)``) collapse to the smallest order after dedup,
        # so a single integer is the canonical form — see the module-level
        # ``PIVOT_TIER_*`` constants for the timeframe nomenclature.
        "pivot_orders": PIVOT_TIER_SHORT,
        # Multi-tier scan: each inner sequence is one tier, runs an
        # independent scan, and the detections are unioned. Tiers exist
        # so disjoint scales surface patterns at multiple horizons —
        # without tiering, small-scale pivots dominate the alternating
        # sequence and the windowing logic never sees major swings as
        # consecutive peaks. Defaults to short / medium / long.
        # Set to ``()`` or ``None`` to disable tiering and fall back to
        # ``pivot_orders``.
        "pivot_tiers": DEFAULT_PIVOT_TIERS,
        "signal_mode": SignalMode.BREAKOUT,
        "decay_bars": 5,
    }

    #: Stable lowercase identifier — matches :class:`Pattern` enum value.
    pattern_name: ClassVar[str] = ""
    #: Default condition preset. Subclasses may override at the class level.
    condition: ClassVar[PatternCondition] = PatternCondition()
    #: Names of per-detector tuning attributes that are forwarded to the
    #: Rust detector via `_core.scan_pattern(..., detector_params=...)`.
    #: Subclasses override; see ``docs/guides/patterns/knobs.md``.
    detector_param_keys: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, condition: PatternCondition | None = None, **params: Any) -> None:
        super().__init__(**params)
        # Per-instance condition shadow keeps the preset intact at the class.
        self._condition: PatternCondition = (
            condition if condition is not None else type(self).condition
        )

    @property
    def effective_condition(self) -> PatternCondition:
        """Active :class:`PatternCondition` (per-instance override or preset)."""
        return self._condition

    # ----------------------------------------------------------------- compute
    def _compute(
        self,
        series_by_field: dict[str, pd.Series],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        events = self._scan(series_by_field, index, asset="<series>")
        signal = events_to_signal(
            events,
            index=index,
            mode=coerce(self.signal_mode, SignalMode),
            decay_bars=int(self.decay_bars),
        )
        return pd.DataFrame({"signal": signal})

    # ----------------------------------------------------------------- events
    def events(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the canonical events table for every asset in ``X``.

        Accepts the same frame shape as :meth:`transform`: a MultiIndex
        ``Bars`` frame with ``(field, asset)`` columns. Output is the
        union of per-asset event tables, with ``EVENTS_COLUMNS`` in the
        canonical order.
        """
        if not isinstance(X.columns, pd.MultiIndex):
            msg = (
                f"{type(self).__name__}.events expects a MultiIndex Bars frame "
                "with (field, asset) columns; got a flat-column DataFrame."
            )
            raise TypeError(msg)
        index = X.index
        if not isinstance(index, pd.DatetimeIndex):
            msg = "Bars frame must have a DatetimeIndex"
            raise TypeError(msg)

        assets = sorted(set(X.columns.get_level_values(-1)))
        per_asset: list[pd.DataFrame] = []
        for asset in assets:
            missing = [f for f in self.inputs if (f, asset) not in X.columns]
            if missing:
                msg = (
                    f"{type(self).__name__} requires fields {self.inputs!r}; "
                    f"asset {asset!r} is missing: {missing}"
                )
                raise KeyError(msg)
            fields = {f: X[(f, asset)] for f in self.inputs}
            per_asset.append(self._scan(fields, index, asset=asset))
        non_empty = [df for df in per_asset if not df.empty]
        if not non_empty:
            return pd.DataFrame(columns=EVENTS_COLUMNS)
        return pd.concat(non_empty, ignore_index=True)

    # ----------------------------------------------------------------- helpers
    def _scan(
        self,
        fields: dict[str, pd.Series],
        index: pd.DatetimeIndex,
        *,
        asset: str,
    ) -> pd.DataFrame:
        """Single-asset scan — bridges to the Rust binding and builds the
        events frame."""
        if not self.pattern_name:
            msg = (
                f"{type(self).__name__}: pattern_name is empty. "
                "Concrete PatternIndicator subclasses must set this class-level "
                "attribute to a registered Rust detector name."
            )
            raise NotImplementedError(msg)
        ts_ns = _index_to_ns(index)
        params = _ohlcv_arrays(fields, expected_len=len(index))
        finite_mask = np.logical_and.reduce([
            np.isfinite(params[name]) for name in ("open", "high", "low", "close", "volume")
        ])
        if not finite_mask.any():
            return pd.DataFrame(columns=EVENTS_COLUMNS)
        if not finite_mask.all():
            index = index[finite_mask]
            ts_ns = ts_ns[finite_mask]
            params = {name: arr[finite_mask] for name, arr in params.items()}

        tiers = _resolve_tiers(self.pivot_tiers, self.pivot_orders)
        detector_params = {
            k: float(getattr(self, k)) for k in self.detector_param_keys if hasattr(self, k)
        }
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        for orders in tiers:
            raw = _core.scan_pattern(
                self.pattern_name,
                ts_ns,
                params["open"],
                params["high"],
                params["low"],
                params["close"],
                params["volume"],
                list(orders),
                float(self.min_quality),
                detector_params,
            )
            for ev in raw:
                key = (int(ev["formation_start"]), int(ev["formation_end"]))
                # Same formation bounds at different scales → keep highest quality.
                existing = merged.get(key)
                if existing is None or float(ev.get("quality", 0.0)) > float(
                    existing.get("quality", 0.0)
                ):
                    merged[key] = ev
        return build_events_frame(list(merged.values()), asset=asset, index=index)


# ----------------------------------------------------------------------- helpers


def _resolve_tiers(
    pivot_tiers: Any,
    pivot_orders: Any,
) -> tuple[tuple[int, ...], ...]:
    """Pick the scan tiers to run.

    If ``pivot_tiers`` is non-empty, use it (each inner sequence is one
    tier). Otherwise fall back to a single tier built from ``pivot_orders``.
    Empty / None / falsy ``pivot_tiers`` is the explicit "single-tier" opt-out.
    """
    if pivot_tiers:
        out: list[tuple[int, ...]] = []
        for tier in pivot_tiers:
            tier_t = tuple(int(x) for x in tier)
            if tier_t:
                out.append(tier_t)
        if out:
            return tuple(out)
    return (tuple(int(x) for x in pivot_orders),)


def _index_to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the index as ``int64`` UTC nanoseconds since the Unix epoch.

    Naive timestamps are treated as UTC — this matches how pandas stores
    them internally when no ``tz`` is set.
    """
    ns = index.view("int64") if index.tz is None else index.tz_convert("UTC").view("int64")
    return np.ascontiguousarray(ns, dtype=np.int64)


def _ohlcv_arrays(fields: dict[str, pd.Series], *, expected_len: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close", "volume"):
        if name not in fields:
            msg = f"missing required OHLCV field: {name!r}"
            raise KeyError(msg)
        arr = np.ascontiguousarray(fields[name].to_numpy(dtype=np.float64), dtype=np.float64)
        if arr.shape[0] != expected_len:
            msg = (
                f"OHLCV field {name!r} length {arr.shape[0]} does not match "
                f"index length {expected_len}"
            )
            raise ValueError(msg)
        out[name] = arr
    return out
