"""Causal building blocks shared by every event detector.

Three primitives enforce the registry's leak-free contract
(``docs/guides/research/event-registry.md``):

* :func:`wilder_atr` — Wilder's ATR, NaN over the warmup region, index-aligned to
  ``close``. The volatility unit for displacement thresholds, breakout buffers and
  the R unit recorded as ``atr_at_confirm``.
* :func:`confirmed_pivots` — pivot highs/lows of order ``k`` together with the bar
  ``p + k`` at which each becomes knowable. A pivot reads ``k`` *future* bars, so
  any level built from it neighbour-locks: it may only inform an event at bar
  ``t`` when ``p + k <= t``.
* :func:`assert_prefix_invariant` — the mandatory online-causality proof. A spec
  does not close a leak; this test does. Running a detector on ``bars[:T]`` must
  reproduce, on the signal columns, exactly the events the full run confirms at or
  before the cutoff. If the detector peeked forward, the truncated run differs and
  this raises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from fundcloud.research.events.schema import OBSERVATION_COLUMNS

__all__ = [
    "PivotConfirmation",
    "assert_prefix_invariant",
    "confirmed_pivots",
    "wilder_atr",
]

#: One confirmed pivot as ``(p, p_plus_k, price)``: the pivot bar index ``p``, the
#: index ``p + k`` at which it is first knowable (neighbour-lock anchor), and the
#: pivot price (``high[p]`` for highs, ``low[p]`` for lows).
PivotConfirmation = tuple[int, int, float]

#: Columns compared by the prefix-invariance test. ``execution_ts`` and
#: ``entry_ref_price`` are excluded: both legitimately depend on the bar *after*
#: ``confirmed_ts`` (a future bar at the cutoff), so their absence in a prefix run
#: is execution timing, not a signal leak.
_SIGNAL_COLUMNS: tuple[str, ...] = tuple(
    c for c in OBSERVATION_COLUMNS if c not in {"execution_ts", "entry_ref_price"}
)


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Wilder's Average True Range, index-aligned to ``close``.

    The first ``n`` entries are ``np.nan`` (warmup). ``ATR[n-1] = mean(TR[0..n-1])``
    seeds the series, then each later bar uses Wilder's recursive smoothing
    ``ATR[t] = (ATR[t-1] * (n-1) + TR[t]) / n``. ``TR[0] = high[0] - low[0]``.

    Parameters
    ----------
    high, low, close
        Equal-length 1-D float arrays for one asset, chronologically ordered.
    n
        ATR length (number of bars in the seed average).

    Returns
    -------
    numpy.ndarray
        Float array of ``len(close)``; ``np.nan`` where ATR is undefined.
    """
    length = len(close)
    if length == 0 or n < 1:
        return np.full(length, np.nan)
    tr = np.empty(length)
    tr[0] = high[0] - low[0]
    if length > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ])
    atr = np.full(length, np.nan)
    if length < n:
        return atr
    atr[n - 1] = np.mean(tr[:n])
    for t in range(n, length):
        atr[t] = (atr[t - 1] * (n - 1) + tr[t]) / n
    return atr


def confirmed_pivots(
    high: np.ndarray, low: np.ndarray, k: int
) -> tuple[list[PivotConfirmation], list[PivotConfirmation]]:
    """Locate pivot highs and lows of order ``k`` with their confirmation bars.

    A pivot high at ``p`` requires ``high[p] == max(high[p-k .. p+k])``; a pivot
    low at ``p`` requires ``low[p] == min(low[p-k .. p+k])`` (ties count as
    pivots — the ``==`` is inclusive). Because the window reaches ``k`` bars into
    the future, the pivot is only knowable at bar ``p + k``: that index is the
    neighbour-lock anchor a downstream level must respect.

    Only fully-windowed centres are considered: ``p`` ranges over
    ``[k, len-1-k]``, so every returned pivot has both a complete left and right
    neighbourhood.

    Parameters
    ----------
    high, low
        Equal-length 1-D float arrays for one asset, chronologically ordered.
    k
        Pivot order (neighbours on each side). Must be ``>= 1``.

    Returns
    -------
    tuple[list[PivotConfirmation], list[PivotConfirmation]]
        ``(pivot_highs, pivot_lows)``. Each list holds ``(p, p + k, price)``
        triples in ascending ``p`` order — ``price`` is ``high[p]`` for highs and
        ``low[p]`` for lows.
    """
    if k < 1:
        msg = f"pivot order k must be >= 1, got {k}"
        raise ValueError(msg)
    length = len(high)
    highs: list[PivotConfirmation] = []
    lows: list[PivotConfirmation] = []
    for p in range(k, length - k):
        window = slice(p - k, p + k + 1)
        if high[p] == np.max(high[window]):
            highs.append((p, p + k, float(high[p])))
        if low[p] == np.min(low[window]):
            lows.append((p, p + k, float(low[p])))
    return highs, lows


def assert_prefix_invariant(
    detect: Callable[..., pd.DataFrame],
    bars: pd.DataFrame,
    *,
    signal_cols: Sequence[str] | None = None,
    n_cutoffs: int = 6,
    **kwargs: object,
) -> None:
    """Prove a detector is online-causal via prefix invariance.

    For each of ``n_cutoffs`` cutoffs ``T`` spread across the series, the events
    a full run confirms at or before ``cutoff_ts = bars.index[T-1]`` must match,
    on ``signal_cols``, the events ``detect(bars.iloc[:T])`` confirms. If they
    differ the detector used a future bar.

    Both subsets are filtered to ``confirmed_ts <= cutoff_ts`` and sorted by
    ``confirmed_ts`` before comparison. Comparison is restricted to
    ``signal_cols`` (default: :data:`OBSERVATION_COLUMNS` minus ``execution_ts``
    and ``entry_ref_price``, which legitimately read the next bar).

    Parameters
    ----------
    detect
        Detector callable ``detect(bars, **kwargs) -> observation frame`` shaped
        like :func:`fundcloud.research.events.schema.build_observations` output.
    bars
        Single-asset OHLCV frame with a sorted tz-aware UTC DatetimeIndex.
    signal_cols
        Columns to compare. Defaults to the leak-sensitive signal columns.
    n_cutoffs
        Number of cutoffs to probe across the series.
    **kwargs
        Forwarded verbatim to every ``detect`` call.

    Raises
    ------
    AssertionError
        If any cutoff's prefix events differ from the full run's, naming the
        offending cutoff and the first differing row.
    """
    cols = list(signal_cols) if signal_cols is not None else list(_SIGNAL_COLUMNS)
    n_bars = len(bars)
    if n_bars == 0:
        return

    full = detect(bars, **kwargs)

    cutoffs = sorted({int(t) for t in np.linspace(1, n_bars, num=n_cutoffs, dtype=int)})
    for cutoff in cutoffs:
        cutoff_ts = bars.index[cutoff - 1]
        prefix = detect(bars.iloc[:cutoff], **kwargs)

        full_sub = _confirmed_subset(full, cutoff_ts, cols)
        prefix_sub = _confirmed_subset(prefix, cutoff_ts, cols)

        if len(full_sub) != len(prefix_sub):
            msg = (
                f"prefix-invariance violation at cutoff T={cutoff} "
                f"(cutoff_ts={cutoff_ts!r}): full run confirms {len(full_sub)} "
                f"event(s) at/<= cutoff but the prefix run yields "
                f"{len(prefix_sub)} — the detector reads future bars."
            )
            raise AssertionError(msg)

        if full_sub.empty:
            continue

        diff = _first_diff(full_sub, prefix_sub, cols)
        if diff is not None:
            row, column, full_val, prefix_val = diff
            msg = (
                f"prefix-invariance violation at cutoff T={cutoff} "
                f"(cutoff_ts={cutoff_ts!r}): row {row} column {column!r} differs "
                f"— full={full_val!r} vs prefix={prefix_val!r}. The detector "
                f"reads future bars."
            )
            raise AssertionError(msg)


def _confirmed_subset(
    obs: pd.DataFrame, cutoff_ts: pd.Timestamp, cols: list[str]
) -> pd.DataFrame:
    """Rows with ``confirmed_ts <= cutoff_ts``, sorted, projected onto ``cols``."""
    if obs.empty:
        return obs.reindex(columns=cols)
    sub = obs[obs["confirmed_ts"] <= cutoff_ts]
    sub = sub.sort_values("confirmed_ts", kind="stable").reset_index(drop=True)
    return sub.reindex(columns=cols)


def _first_diff(
    full_sub: pd.DataFrame, prefix_sub: pd.DataFrame, cols: list[str]
) -> tuple[int, str, object, object] | None:
    """First ``(row, column, full_value, prefix_value)`` that differs, or ``None``.

    NaN positions are treated as equal so floating-point gaps compare cleanly.
    """
    for row in range(len(full_sub)):
        for column in cols:
            full_val = full_sub.iloc[row][column]
            prefix_val = prefix_sub.iloc[row][column]
            if _values_equal(full_val, prefix_val):
                continue
            return row, column, full_val, prefix_val
    return None


def _values_equal(a: object, b: object) -> bool:
    """True if two cell values are equal, with NaN == NaN."""
    a_na = pd.isna(a)
    b_na = pd.isna(b)
    if bool(a_na) or bool(b_na):
        return bool(a_na) and bool(b_na)
    return bool(a == b)
