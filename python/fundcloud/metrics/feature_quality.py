"""Feature-quality metrics for event-based features (chart patterns etc.).

This module is intentionally **not** flat-reexported from
:mod:`fundcloud.metrics` — feature quality is conceptually distinct from
portfolio metrics, and several names (``win_rate`` etc.) would collide.
Import as::

    from fundcloud.metrics import feature_quality as fq

The headline entry point is :func:`evaluate`, which takes an events
table (the canonical schema produced by
:meth:`fundcloud.features.patterns.PatternIndicator.events`) plus a
MultiIndex Bars frame and returns a per-horizon DataFrame summarising
the feature's effectiveness.

**Design choices** (documented per function; the most important here):

* **Forward path** = bars ``[breakout_ts + 1, breakout_ts + h]``. The
  breakout bar itself is excluded so we don't double-count its move
  into the realised return. Entry price = ``events["breakout_price"]``
  with a fallback to ``events["entry_price"]`` if the former is NaN.
* **Hit rate** is *close-based*: a bar at horizon ``h`` is a "hit" if
  the directional close-to-entry return is positive. Path-based
  outcomes (target-hit before stop-hit) are reported separately when
  they're available.
* **MFE / MAE** are intraday — max forward ``high`` / min forward
  ``low`` — not close-based. That's what would actually have triggered
  a stop in live trading.
* **R-multiple unit** = ``|entry − stop_price|`` if the events table
  carries a stop, otherwise ``1 × ATR(atr_window)`` at the breakout
  bar. This lets us ship without coupling to ``apply_condition``.
* **Baseline** = asset-weighted unconditional fraction of forward
  ``h``-bar returns whose sign matches the event's direction. The
  fairest "would a random entry have done the same?" yardstick.
* **Throwback** is fixed at ``lookahead=10`` bars and is *not* a
  per-horizon metric — it's a property of the breakout. Reported once
  per panel, repeated across the horizon column for table readability.
  Note: the v1 events table fires at ``formation_end`` (the last pivot
  bar) rather than at a *confirmed* close-through-neckline breakout, so
  throwback rates here are systematically inflated relative to
  Bulkowski's published numbers — the "breakout" bar is right next to
  the neckline by construction. This metric will be more meaningful
  once Phase 8 lands real breakout-confirmation in the events table.
* **n_events** is reported per horizon: events near the end of the bar
  series whose ``t + h`` exceeds the series length are dropped from
  that horizon's aggregates. This is honest about lookahead-availability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fundcloud.features.patterns._enums import Direction

__all__ = [
    "avg_mae_atr",
    "avg_mfe_atr",
    "baseline_hit_rate",
    "edge_ratio",
    "evaluate",
    "expectancy",
    "hit_rate",
    "icir",
    "information_coefficient",
    "mae_p95_atr",
    "per_asset",
    "quality_buckets",
    "throwback_rate",
    "time_stability",
]

# Default per-horizon cutoffs used by `evaluate`.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20, 60)
# Wilder's ATR period — Bulkowski's standard.
DEFAULT_ATR_WINDOW: int = 14
# Throwback look-ahead — Bulkowski's standard.
DEFAULT_THROWBACK_WINDOW: int = 10
# Allowed values for the trade_direction knob.
_TRADE_DIRECTIONS = ("long", "short")


def _resolve_sign(trade_direction: str) -> float:
    """Resolve ``trade_direction`` (``"long"`` | ``"short"``) to a ±1 sign
    used to flip MFE/MAE and hit-rate computations.
    """
    if trade_direction == "long":
        return 1.0
    if trade_direction == "short":
        return -1.0
    msg = f"unknown trade_direction: {trade_direction!r}; valid: {_TRADE_DIRECTIONS}"
    raise ValueError(msg)


# ----------------------------------------------------------------------- helpers


def _direction_sign(direction: Any) -> float:
    """Map a Direction (enum or string) to ``+1.0`` for bullish, ``-1.0``
    for bearish, ``0.0`` for neutral / unknown.
    """
    value = direction.value if isinstance(direction, Direction) else str(direction).lower()
    if value == "bullish":
        return 1.0
    if value == "bearish":
        return -1.0
    return 0.0


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Wilder's ATR. Returns ``np.nan`` for the first ``window`` bars.

    ``ATR[w] = mean(TR[1..w])``; subsequent bars use the recursive
    smoothing ``ATR[t] = (ATR[t-1] * (w-1) + TR[t]) / w``.
    """
    n = len(close)
    if n == 0 or window < 1:
        return np.full(n, np.nan)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ])
    atr = np.full(n, np.nan)
    if n < window:
        return atr
    atr[window - 1] = np.mean(tr[:window])
    for t in range(window, n):
        atr[t] = (atr[t - 1] * (window - 1) + tr[t]) / window
    return atr


class _EventPath:
    """A single event with its forward path arrays already aligned.

    Carries everything the per-horizon metric functions need so they can
    work on a list of these without re-walking the bars frame.
    """

    __slots__ = (
        "asset",
        "breakout_pos",
        "close",
        "direction_sign",
        "entry",
        "high",
        "low",
        "quality",
        "stop_distance",
        "ts",
    )

    def __init__(
        self,
        *,
        asset: str,
        breakout_pos: int,
        direction_sign: float,
        entry: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        stop_distance: float,
        quality: float,
        ts: pd.Timestamp,
    ) -> None:
        self.asset = asset
        self.breakout_pos = breakout_pos
        self.direction_sign = direction_sign
        self.entry = entry
        self.high = high
        self.low = low
        self.close = close
        self.stop_distance = stop_distance
        self.quality = quality
        self.ts = ts


def _select_asset(bars: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Pull a single asset's OHLCV columns out of the MultiIndex frame.

    Drops rows where any of the OHLC fields is NaN — assets in a
    multi-asset bars frame typically have leading NaN history before
    their listing date, which would poison Wilder's ATR seed.

    Raises ``KeyError`` if the asset isn't present.
    """
    if not isinstance(bars.columns, pd.MultiIndex):
        msg = "bars must have MultiIndex (field, asset) columns"
        raise TypeError(msg)
    sub = bars.xs(asset, level=-1, axis=1)
    return sub.dropna(subset=["open", "high", "low", "close"])


def _build_event_paths(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    max_horizon: int,
    atr_window: int,
    trade_direction: str = "long",
) -> list[_EventPath]:
    """Align each event with its forward path and ATR-based stop unit.

    Events whose ``breakout_ts`` isn't in the bar index are dropped
    silently. Events with no forward bars at all (breakout on the last
    bar) are also dropped. Events beyond ``max_horizon`` get truncated
    to the available bars — per-horizon callers re-check length.
    """
    out: list[_EventPath] = []
    asset_cache: dict[str, pd.DataFrame | None] = {}
    atr_cache: dict[str, np.ndarray] = {}

    for _, ev in events.iterrows():
        asset = str(ev["asset"])
        if asset not in asset_cache:
            try:
                asset_cache[asset] = _select_asset(bars, asset)
            except KeyError:
                asset_cache[asset] = None
        ab = asset_cache[asset]
        if ab is None:
            continue

        ts = ev.get("breakout_ts")
        if ts is None or pd.isna(ts):
            continue
        try:
            pos = ab.index.get_loc(ts)
        except KeyError:
            continue
        if isinstance(pos, slice):
            pos = pos.start
        if pos >= len(ab) - 1:
            continue

        if asset not in atr_cache:
            atr_cache[asset] = _wilder_atr(
                ab["high"].to_numpy(np.float64),
                ab["low"].to_numpy(np.float64),
                ab["close"].to_numpy(np.float64),
                atr_window,
            )
        atr_at_entry = float(atr_cache[asset][pos])

        end_pos = min(pos + max_horizon, len(ab) - 1)
        forward = ab.iloc[pos + 1 : end_pos + 1]

        entry_raw = ev.get("breakout_price")
        if entry_raw is None or pd.isna(entry_raw):
            entry_raw = ev.get("entry_price")
        if entry_raw is None or pd.isna(entry_raw):
            continue
        entry = float(entry_raw)

        stop_raw = ev.get("stop_price")
        if stop_raw is not None and not pd.isna(stop_raw):
            stop_distance = abs(entry - float(stop_raw))
        else:
            stop_distance = atr_at_entry
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            # No usable risk denominator — skip; would otherwise produce
            # ±inf R-multiples that contaminate aggregates.
            continue

        sign = _resolve_sign(trade_direction)

        quality = ev.get("quality")
        quality_f = float(quality) if quality is not None and not pd.isna(quality) else np.nan

        out.append(
            _EventPath(
                asset=asset,
                breakout_pos=int(pos),
                direction_sign=sign,
                entry=entry,
                high=forward["high"].to_numpy(np.float64),
                low=forward["low"].to_numpy(np.float64),
                close=forward["close"].to_numpy(np.float64),
                stop_distance=float(stop_distance),
                quality=quality_f,
                ts=ts,
            )
        )
    return out


def _truncate_to_horizon(paths: list[_EventPath], horizon: int) -> list[_EventPath]:
    """Keep only paths with ``len >= horizon``; the others lack lookahead."""
    return [p for p in paths if p.high.shape[0] >= horizon]


# --------------------------------------------------------------- scalar metrics


def hit_rate(paths: list[_EventPath], horizon: int) -> float:
    """Close-based hit rate at ``horizon`` bars.

    "Hit" = directional close return at ``t + horizon`` is strictly
    positive. Returns ``np.nan`` when no events have lookahead.
    """
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return float("nan")
    wins = 0
    for p in keep:
        ret = (p.close[horizon - 1] - p.entry) / p.entry * p.direction_sign
        if ret > 0:
            wins += 1
    return wins / len(keep)


def expectancy(paths: list[_EventPath], horizon: int) -> float:
    """Mean realised R-multiple at horizon.

    R-multiple = ``signed_return_at_horizon × entry / stop_distance``.
    Stop distance is taken from ``events.stop_price`` if present, else
    ``1 × ATR(atr_window)`` at the breakout bar.
    """
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return float("nan")
    rs = np.empty(len(keep))
    for i, p in enumerate(keep):
        signed_move = (p.close[horizon - 1] - p.entry) * p.direction_sign
        rs[i] = signed_move / p.stop_distance
    return float(np.mean(rs))


def avg_mfe_atr(paths: list[_EventPath], horizon: int) -> float:
    """Average maximum favourable excursion at horizon, in ATR units.

    Excursion is intraday: bullish events look at forward ``high``,
    bearish events look at forward ``low``.
    """
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return float("nan")
    vals = np.empty(len(keep))
    for i, p in enumerate(keep):
        high_window = p.high[:horizon]
        low_window = p.low[:horizon]
        if p.direction_sign > 0:
            mfe = float(high_window.max()) - p.entry
        else:
            mfe = p.entry - float(low_window.min())
        # MFE is by definition ≥ 0 — if price never moved in our favour
        # the favourable excursion is zero, not negative.
        vals[i] = max(mfe, 0.0) / p.stop_distance
    return float(np.mean(vals))


def avg_mae_atr(paths: list[_EventPath], horizon: int) -> float:
    """Average maximum adverse excursion at horizon, in ATR units."""
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return float("nan")
    vals = np.empty(len(keep))
    for i, p in enumerate(keep):
        high_window = p.high[:horizon]
        low_window = p.low[:horizon]
        if p.direction_sign > 0:
            mae = p.entry - float(low_window.min())
        else:
            mae = float(high_window.max()) - p.entry
        vals[i] = max(mae, 0.0) / p.stop_distance
    return float(np.mean(vals))


def mae_p95_atr(paths: list[_EventPath], horizon: int) -> float:
    """95th-percentile MAE in ATR units — the stop-sizing reference number.

    Mean MAE understates real stop pain because the distribution is
    right-skewed. The 95th percentile is what you'd actually need to
    survive in live trading.
    """
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return float("nan")
    vals = np.empty(len(keep))
    for i, p in enumerate(keep):
        high_window = p.high[:horizon]
        low_window = p.low[:horizon]
        if p.direction_sign > 0:
            mae = p.entry - float(low_window.min())
        else:
            mae = float(high_window.max()) - p.entry
        vals[i] = max(mae, 0.0) / p.stop_distance
    return float(np.quantile(vals, 0.95))


def edge_ratio(paths: list[_EventPath], horizon: int) -> float:
    """``avg_mfe_atr / avg_mae_atr`` — symmetry of payoff.

    > 1 means the typical winner runs further than the typical loser
    bleeds. < 1 means the opposite. Returns ``np.nan`` if MAE is zero
    (no adverse excursion in the sample — essentially impossible at
    realistic horizons but defended for cleanliness).
    """
    mae = avg_mae_atr(paths, horizon)
    if mae <= 0 or not np.isfinite(mae):
        return float("nan")
    return avg_mfe_atr(paths, horizon) / mae


def throwback_rate(paths: list[_EventPath], window: int = DEFAULT_THROWBACK_WINDOW) -> float:
    """Fraction of events that re-touch the breakout level within ``window``
    bars after the breakout.

    Bullish events: a "throwback" = forward ``low`` falls back to or
    below ``entry``. Bearish events: forward ``high`` rises back to or
    above ``entry``. Bulkowski's textbook finding: high throwback rates
    *reduce* post-breakout performance.

    Not a per-horizon metric — it's a property of the breakout.
    """
    if not paths:
        return float("nan")
    hits = 0
    counted = 0
    for p in paths:
        n = min(window, p.high.shape[0])
        if n <= 0:
            continue
        counted += 1
        if p.direction_sign > 0:
            if float(p.low[:n].min()) <= p.entry:
                hits += 1
        else:
            if float(p.high[:n].max()) >= p.entry:
                hits += 1
    if counted == 0:
        return float("nan")
    return hits / counted


def information_coefficient(paths: list[_EventPath], horizon: int) -> float:
    """Spearman rank correlation between event ``quality`` and signed
    forward return at ``horizon``.

    Returns ``np.nan`` if fewer than 5 events have non-NaN quality
    (Spearman with n < 5 is meaningless).
    """
    keep = _truncate_to_horizon(paths, horizon)
    if len(keep) < 5:
        return float("nan")
    qs = np.array([p.quality for p in keep])
    rs = np.array([(p.close[horizon - 1] - p.entry) / p.entry * p.direction_sign for p in keep])
    mask = np.isfinite(qs) & np.isfinite(rs)
    if mask.sum() < 5:
        return float("nan")
    qs = qs[mask]
    rs = rs[mask]
    q_ranks = pd.Series(qs).rank().to_numpy()
    r_ranks = pd.Series(rs).rank().to_numpy()
    if np.std(q_ranks) == 0 or np.std(r_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(q_ranks, r_ranks)[0, 1])


def icir(paths: list[_EventPath], horizon: int, *, period: str = "Y") -> float:
    """Mean / std of yearly ICs.

    A high mean IC paired with high IC volatility is unstable; ICIR
    captures both at once. Returns ``np.nan`` when there are fewer than
    3 periods with sufficient events.
    """
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return float("nan")
    rows = [
        {
            # Strip tz before to_period — pandas warns otherwise. UTC by
            # convention, so localising/dropping the offset is lossless.
            "period": pd.Timestamp(p.ts).tz_convert(None).to_period(period)
            if pd.Timestamp(p.ts).tzinfo is not None
            else pd.Timestamp(p.ts).to_period(period),
            "quality": p.quality,
            "ret": (p.close[horizon - 1] - p.entry) / p.entry * p.direction_sign,
        }
        for p in keep
    ]
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["quality", "ret"])
    ics: list[float] = []
    for _, group in df.groupby("period"):
        if len(group) < 5:
            continue
        q_ranks = group["quality"].rank()
        r_ranks = group["ret"].rank()
        if q_ranks.std() == 0 or r_ranks.std() == 0:
            continue
        ics.append(float(np.corrcoef(q_ranks, r_ranks)[0, 1]))
    if len(ics) < 3:
        return float("nan")
    arr = np.array(ics)
    if arr.std(ddof=0) == 0:
        return float("nan")
    return float(arr.mean() / arr.std(ddof=0))


# ----------------------------------------------------------------------- baseline


def baseline_hit_rate(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
    *,
    trade_direction: str = "long",
) -> float:
    """Asset-weighted unconditional ``P(directional forward return > 0)``
    at ``horizon``.

    For each asset present in the events table, computes the fraction of
    bars whose ``close[t+h] - close[t]`` has the sign implied by the
    caller-supplied ``trade_direction``. Aggregates across assets
    weighted by event count, so a pattern that fires more on AAPL
    contributes more AAPL-baseline weight.

    Notes:

    * Uses the same bar series the events came from. No filtering by
      formation window — the baseline is "any random entry on this
      asset" at horizon ``h``.
    * Returns ``np.nan`` if the events table has no usable rows.
    """
    if horizon <= 0:
        msg = "horizon must be > 0"
        raise ValueError(msg)
    if events.empty:
        return float("nan")
    if not isinstance(bars.columns, pd.MultiIndex):
        msg = "bars must have MultiIndex (field, asset) columns"
        raise TypeError(msg)

    sign = _resolve_sign(trade_direction)
    weights: dict[str, int] = {str(asset): len(group) for asset, group in events.groupby("asset")}
    if not weights:
        return float("nan")

    weighted_sum = 0.0
    weight_total = 0
    for asset in weights:
        try:
            close = _select_asset(bars, asset)["close"].to_numpy(np.float64)
        except KeyError:
            continue
        if close.size <= horizon:
            continue
        forward = close[horizon:] - close[:-horizon]
        directional = forward * sign
        if directional.size == 0:
            continue
        frac = float((directional > 0).mean())
        w = weights[asset]
        weighted_sum += frac * w
        weight_total += w
    if weight_total == 0:
        return float("nan")
    return weighted_sum / weight_total


# ----------------------------------------------------------------------- panel


def evaluate(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    atr_window: int = DEFAULT_ATR_WINDOW,
    throwback_window: int = DEFAULT_THROWBACK_WINDOW,
    baseline: bool = True,
    trade_direction: str = "long",
    condition: Any = None,
) -> pd.DataFrame:
    """Headline feature-quality panel — one row per horizon.

    Columns (in order):

    ``n_events`` (int), ``hit_rate``, [``baseline_hit``],
    ``expectancy``, ``edge_ratio``, ``mfe_atr``, ``mae_atr``,
    ``mae_p95_atr``, ``ic``, ``icir``, ``throwback``.

    The throwback column is filled with the same value across rows —
    it's a property of the breakout, not the horizon, but is included
    here so the table is self-contained.

    ``trade_direction`` is caller-supplied (``"long"`` or ``"short"``)
    and applied uniformly to every event in the input table. The baseline
    is computed against the same side so the apples-to-apples comparison
    stays honest.

    ``condition`` (optional :class:`PatternCondition`): when supplied,
    :func:`fundcloud.features.patterns.apply_condition` runs first to
    fill ``target_price`` / ``stop_price`` per the condition's target
    and stop methods. R-multiples (and therefore ``expectancy`` /
    ``edge_ratio``) then use the condition-derived stop distance
    instead of the 1×ATR fallback.

    Empty events frame → returns an all-NaN frame with the right shape.
    """
    if trade_direction not in _TRADE_DIRECTIONS:
        msg = f"unknown trade_direction: {trade_direction!r}; valid: {_TRADE_DIRECTIONS}"
        raise ValueError(msg)
    if not horizons:
        msg = "horizons must be non-empty"
        raise ValueError(msg)
    horizons_sorted = tuple(sorted(set(int(h) for h in horizons)))
    if any(h <= 0 for h in horizons_sorted):
        msg = "horizons must contain only positive integers"
        raise ValueError(msg)
    max_horizon = horizons_sorted[-1]

    columns = [
        "n_events",
        "hit_rate",
        "expectancy",
        "edge_ratio",
        "mfe_atr",
        "mae_atr",
        "mae_p95_atr",
        "ic",
        "icir",
        "throwback",
    ]
    if baseline:
        columns.insert(2, "baseline_hit")

    empty = pd.DataFrame(
        np.nan,
        index=pd.Index(horizons_sorted, name="horizon"),
        columns=columns,
    )
    empty["n_events"] = 0
    if events.empty:
        return empty

    if condition is not None:
        from fundcloud.features.patterns import apply_condition

        events = apply_condition(events, condition, bars)

    paths = _build_event_paths(
        events,
        bars,
        max_horizon=max_horizon,
        atr_window=atr_window,
        trade_direction=trade_direction,
    )
    if not paths:
        return empty

    tb = throwback_rate(paths, window=throwback_window)
    rows: list[dict[str, float]] = []
    for h in horizons_sorted:
        keep = _truncate_to_horizon(paths, h)
        row: dict[str, Any] = {
            "n_events": len(keep),
            "hit_rate": hit_rate(paths, h),
            "expectancy": expectancy(paths, h),
            "edge_ratio": edge_ratio(paths, h),
            "mfe_atr": avg_mfe_atr(paths, h),
            "mae_atr": avg_mae_atr(paths, h),
            "mae_p95_atr": mae_p95_atr(paths, h),
            "ic": information_coefficient(paths, h),
            "icir": icir(paths, h),
            "throwback": tb,
        }
        if baseline:
            row["baseline_hit"] = baseline_hit_rate(
                events, bars, h, trade_direction=trade_direction
            )
        rows.append(row)

    panel = pd.DataFrame(rows, index=pd.Index(horizons_sorted, name="horizon"))
    return panel[columns]


# --------------------------------------------------------- stratified diagnostics


def quality_buckets(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizon: int = 20,
    n_buckets: int = 5,
    atr_window: int = DEFAULT_ATR_WINDOW,
    trade_direction: str = "long",
) -> pd.DataFrame:
    """Bucket events by ``quality`` and report metrics per bucket.

    The single most useful diagnostic for whether the geometric scorer
    is doing useful work: if ``hit_rate`` and ``expectancy`` increase
    monotonically from the worst-quality bucket to the best, the
    scorer's 30/25/25/20 weighting earns its keep. Flat buckets are a
    signal to recalibrate.

    Returns a frame with one row per bucket (Q1 = worst, Q``n`` = best),
    columns: ``quality_min``, ``quality_max``, ``n_events``,
    ``hit_rate``, ``expectancy``, ``edge_ratio``, ``mfe_atr``,
    ``mae_atr``. NaN rows are returned when a bucket is empty (rare —
    happens when many events share the same quality and ``qcut`` can't
    split cleanly).

    Bucketing uses ``pd.qcut`` with ``duplicates="drop"`` so tied
    quality values don't crash; in extreme cases this yields fewer
    than ``n_buckets`` rows.
    """
    if trade_direction not in _TRADE_DIRECTIONS:
        msg = f"unknown trade_direction: {trade_direction!r}; valid: {_TRADE_DIRECTIONS}"
        raise ValueError(msg)
    if horizon <= 0:
        msg = "horizon must be > 0"
        raise ValueError(msg)
    if n_buckets < 2:
        msg = "n_buckets must be >= 2"
        raise ValueError(msg)

    columns = [
        "quality_min",
        "quality_max",
        "n_events",
        "hit_rate",
        "expectancy",
        "edge_ratio",
        "mfe_atr",
        "mae_atr",
    ]
    bucket_index = pd.Index([f"Q{i + 1}" for i in range(n_buckets)], name="bucket")
    empty = pd.DataFrame(np.nan, index=bucket_index, columns=columns)
    empty["n_events"] = 0
    if events.empty:
        return empty

    paths = _build_event_paths(
        events,
        bars,
        max_horizon=horizon,
        atr_window=atr_window,
        trade_direction=trade_direction,
    )
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return empty

    qualities = np.array([p.quality for p in keep])
    finite = np.isfinite(qualities)
    if finite.sum() < n_buckets:
        return empty
    keep_finite = [p for p, f in zip(keep, finite, strict=True) if f]
    qualities = qualities[finite]

    try:
        bucket_labels = pd.qcut(qualities, q=n_buckets, duplicates="drop")
    except ValueError:
        return empty

    # When ``duplicates="drop"`` collapses tied edges, qcut yields fewer
    # categories than requested. Re-label the surviving categories (sorted
    # by Interval, lowest quality first) Q1..Qk so callers always get a
    # densely-numbered axis.
    bucket_cat = pd.Categorical(bucket_labels)
    ordered_cats = list(bucket_cat.categories)
    n_actual = len(ordered_cats)
    label_map = {cat: f"Q{i + 1}" for i, cat in enumerate(ordered_cats)}
    bucket_labels = np.asarray(pd.Series(bucket_cat).map(label_map))
    actual_index = pd.Index([f"Q{i + 1}" for i in range(n_actual)], name="bucket")

    rows: list[dict[str, Any]] = []
    for label in actual_index:
        mask = bucket_labels == label
        bucket_paths = [p for p, m in zip(keep_finite, mask, strict=True) if m]
        if not bucket_paths:
            rows.append({col: np.nan for col in columns} | {"n_events": 0})
            continue
        bucket_qualities = qualities[mask]
        rows.append({
            "quality_min": float(np.min(bucket_qualities)),
            "quality_max": float(np.max(bucket_qualities)),
            "n_events": len(bucket_paths),
            "hit_rate": hit_rate(bucket_paths, horizon),
            "expectancy": expectancy(bucket_paths, horizon),
            "edge_ratio": edge_ratio(bucket_paths, horizon),
            "mfe_atr": avg_mfe_atr(bucket_paths, horizon),
            "mae_atr": avg_mae_atr(bucket_paths, horizon),
        })

    return pd.DataFrame(rows, index=actual_index)[columns]


def per_asset(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizon: int = 20,
    atr_window: int = DEFAULT_ATR_WINDOW,
    trade_direction: str = "long",
) -> pd.DataFrame:
    """Stratified view: one row per asset, same metrics as ``evaluate``.

    Discovery tool, not a sanity check — a feature working on AAPL but
    not MSFT is normal. Use this to build an asset-specific deployment
    list.

    Returns a frame indexed by asset (alphabetical), columns:
    ``n_events``, ``hit_rate``, ``baseline_hit``, ``expectancy``,
    ``edge_ratio``, ``mfe_atr``, ``mae_atr``. Empty events → empty
    frame with the right columns.
    """
    if trade_direction not in _TRADE_DIRECTIONS:
        msg = f"unknown trade_direction: {trade_direction!r}; valid: {_TRADE_DIRECTIONS}"
        raise ValueError(msg)
    if horizon <= 0:
        msg = "horizon must be > 0"
        raise ValueError(msg)

    columns = [
        "n_events",
        "hit_rate",
        "baseline_hit",
        "expectancy",
        "edge_ratio",
        "mfe_atr",
        "mae_atr",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    paths = _build_event_paths(
        events,
        bars,
        max_horizon=horizon,
        atr_window=atr_window,
        trade_direction=trade_direction,
    )
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return pd.DataFrame(columns=columns)

    rows: dict[str, dict[str, Any]] = {}
    for asset in sorted({p.asset for p in keep}):
        ap = [p for p in keep if p.asset == asset]
        events_asset = events[events["asset"] == asset]
        rows[asset] = {
            "n_events": len(ap),
            "hit_rate": hit_rate(ap, horizon),
            "baseline_hit": baseline_hit_rate(
                events_asset, bars, horizon, trade_direction=trade_direction
            ),
            "expectancy": expectancy(ap, horizon),
            "edge_ratio": edge_ratio(ap, horizon),
            "mfe_atr": avg_mfe_atr(ap, horizon),
            "mae_atr": avg_mae_atr(ap, horizon),
        }
    return pd.DataFrame.from_dict(rows, orient="index")[columns].rename_axis("asset")


def time_stability(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizon: int = 20,
    n_folds: int = 5,
    atr_window: int = DEFAULT_ATR_WINDOW,
    trade_direction: str = "long",
) -> pd.DataFrame:
    """Chronological folds: equal-sized splits of the events by date.

    A real sanity check — if a feature only worked in 2010-2014 and is
    flat everywhere else, the apparent edge is regime-bound and risky
    to deploy forward.

    Returns a frame indexed ``fold_1`` … ``fold_n``, columns:
    ``start``, ``end``, ``n_events``, ``hit_rate``, ``expectancy``,
    ``edge_ratio``. Empty events → empty frame with the right columns.
    """
    if trade_direction not in _TRADE_DIRECTIONS:
        msg = f"unknown trade_direction: {trade_direction!r}; valid: {_TRADE_DIRECTIONS}"
        raise ValueError(msg)
    if horizon <= 0:
        msg = "horizon must be > 0"
        raise ValueError(msg)
    if n_folds < 2:
        msg = "n_folds must be >= 2"
        raise ValueError(msg)

    columns = ["start", "end", "n_events", "hit_rate", "expectancy", "edge_ratio"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    paths = _build_event_paths(
        events,
        bars,
        max_horizon=horizon,
        atr_window=atr_window,
        trade_direction=trade_direction,
    )
    keep = _truncate_to_horizon(paths, horizon)
    if not keep:
        return pd.DataFrame(columns=columns)

    keep_sorted = sorted(keep, key=lambda p: p.ts)
    n = len(keep_sorted)
    # Equal-sized event-count folds (not equal-sized time folds): with
    # sparse pattern data, time-equal folds can leave folds with zero
    # events. Event-equal folds always have ~n/n_folds events each.
    fold_size = max(1, n // n_folds)
    rows: list[dict[str, Any]] = []
    for i in range(n_folds):
        lo = i * fold_size
        hi = (i + 1) * fold_size if i < n_folds - 1 else n
        slice_ = keep_sorted[lo:hi]
        if not slice_:
            rows.append({col: np.nan for col in columns} | {"n_events": 0})
            continue
        rows.append({
            "start": pd.Timestamp(slice_[0].ts),
            "end": pd.Timestamp(slice_[-1].ts),
            "n_events": len(slice_),
            "hit_rate": hit_rate(slice_, horizon),
            "expectancy": expectancy(slice_, horizon),
            "edge_ratio": edge_ratio(slice_, horizon),
        })

    fold_index = pd.Index([f"fold_{i + 1}" for i in range(n_folds)], name="fold")
    return pd.DataFrame(rows, index=fold_index)[columns]
