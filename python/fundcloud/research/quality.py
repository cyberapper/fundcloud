"""Data-quality gate + cleaner for daily OHLCV panels.

:func:`check_quality` inspects a canonical ``(field, symbol)`` panel and returns a
:class:`QualityReport`; :func:`clean_panel` removes the fatally-bad bars and returns
a panel guaranteed to pass the gate. Run one of these before any research touches the
data — "clean it before we use it".

Fatal checks (must be zero for :attr:`QualityReport.ok`):

* null OHLC — a *partial* bar (some of open/high/low/close present, some missing);
* ``high < low`` — an impossible bar;
* OHLC inconsistency — ``low > min(open, close)`` or ``high < max(open, close)``;
* non-positive price — any of open/high/low/close ``<= 0``;
* duplicate dates per symbol — index integrity (should be 0 after the loader's collapse);
* non-monotonic or non-UTC index.

Warning checks (reported, never block):

* ``volume <= 0`` — common for delisted-but-still-reporting tickers;
* missing trading days — business-day gaps (holidays make this noisy);
* symbols below ``min_days`` of history.

Comparisons are NaN-safe: a symbol that simply did not trade on a date has all fields
NaN (legitimate absence) and is not counted as a violation. The OHLC math operates on
the canonical ``(field, symbol)`` MultiIndex panel; a flat single-asset frame is treated
as one anonymous symbol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["QualityReport", "check_quality", "clean_panel"]

_OHLC = ("open", "high", "low", "close")


@dataclass(frozen=True)
class QualityReport:
    """Outcome of :func:`check_quality`. See module docstring for the checks."""

    n_symbols: int
    n_bars: int
    null_ohlc: int
    high_lt_low: int
    ohlc_inconsistent: int
    nonpositive_price: int
    duplicate_dates: int
    monotonic_index: bool
    utc_index: bool
    zero_volume: int
    missing_trading_days: int
    symbols_below_min_days: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True iff every *fatal* check passes (warnings are ignored)."""
        return (
            self.null_ohlc == 0
            and self.high_lt_low == 0
            and self.ohlc_inconsistent == 0
            and self.nonpositive_price == 0
            and self.duplicate_dates == 0
            and self.monotonic_index
            and self.utc_index
        )


def _field(bars: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Extract one field as a (date x symbol) frame, or ``None`` if absent.

    Flat single-asset frames are mapped to a single ``"_"`` column so that the
    OHLC fields align with each other under arithmetic.
    """
    cols = bars.columns
    if isinstance(cols, pd.MultiIndex):
        if name in cols.get_level_values(0):
            return bars.xs(name, axis=1, level=0)
        return None
    if name in cols:
        return bars[name].to_frame(name="_")
    return None


def _ohlc_frames(
    bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Return aligned (open, high, low, close) frames, or ``None`` if incomplete."""
    o, h, low, c = (_field(bars, f) for f in _OHLC)
    if o is None or h is None or low is None or c is None:
        return None
    return o, h, low, c


def _bad_mask(bars: pd.DataFrame) -> pd.DataFrame | None:
    """Per-(date, symbol) boolean mask of fatally-bad *bars* (NaN-safe)."""
    frames = _ohlc_frames(bars)
    if frames is None:
        return None
    o, h, low, c = frames
    present = o.notna() | h.notna() | low.notna() | c.notna()
    all_present = o.notna() & h.notna() & low.notna() & c.notna()
    min_oc = np.minimum(o, c)
    max_oc = np.maximum(o, c)
    bad = (
        (present & ~all_present)                          # partial bar
        | (h < low)                                       # impossible bar
        | (low > min_oc)                                  # low above body
        | (h < max_oc)                                    # high below body
        | (o <= 0) | (h <= 0) | (low <= 0) | (c <= 0)     # non-positive price
    )
    return bad.fillna(False).astype(bool)


def check_quality(bars: pd.DataFrame, *, min_days: int = 250) -> QualityReport:
    """Inspect a daily OHLCV panel and return a :class:`QualityReport`.

    Parameters
    ----------
    bars
        Canonical ``(field, symbol)`` MultiIndex panel on a DatetimeIndex.
    min_days
        Warn-threshold: symbols with fewer non-null closes are listed in
        :attr:`QualityReport.symbols_below_min_days`.
    """
    idx = bars.index
    monotonic = bool(getattr(idx, "is_monotonic_increasing", True))
    utc = isinstance(idx, pd.DatetimeIndex) and idx.tz is not None
    duplicate_dates = int(idx.duplicated().sum())

    frames = _ohlc_frames(bars)
    if frames is None or bars.empty:
        return QualityReport(
            n_symbols=0, n_bars=0, null_ohlc=0, high_lt_low=0, ohlc_inconsistent=0,
            nonpositive_price=0, duplicate_dates=duplicate_dates,
            monotonic_index=monotonic, utc_index=utc, zero_volume=0,
            missing_trading_days=0, symbols_below_min_days=(),
        )
    o, h, low, c = frames
    present = o.notna() | h.notna() | low.notna() | c.notna()
    all_present = o.notna() & h.notna() & low.notna() & c.notna()
    min_oc = np.minimum(o, c)
    max_oc = np.maximum(o, c)

    null_ohlc = int((present & ~all_present).to_numpy().sum())
    high_lt_low = int((h < low).to_numpy().sum())
    ohlc_inconsistent = int(((low > min_oc) | (h < max_oc)).to_numpy().sum())
    nonpositive = int(((o <= 0) | (h <= 0) | (low <= 0) | (c <= 0)).to_numpy().sum())

    vol = _field(bars, "volume")
    zero_volume = int((vol <= 0).to_numpy().sum()) if vol is not None else 0

    missing_days = 0
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        full = pd.bdate_range(idx.min(), idx.max(), tz=idx.tz)
        missing_days = len(full.difference(idx))

    counts = c.notna().sum()
    below = tuple(sorted(str(s) for s in counts.index[counts < min_days]))

    return QualityReport(
        n_symbols=int(c.shape[1]),
        n_bars=int(c.notna().to_numpy().sum()),
        null_ohlc=null_ohlc,
        high_lt_low=high_lt_low,
        ohlc_inconsistent=ohlc_inconsistent,
        nonpositive_price=nonpositive,
        duplicate_dates=duplicate_dates,
        monotonic_index=monotonic,
        utc_index=utc,
        zero_volume=zero_volume,
        missing_trading_days=missing_days,
        symbols_below_min_days=below,
    )


def clean_panel(bars: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Return a fatally-clean copy of ``bars`` plus its :class:`QualityReport`.

    Drops duplicate index rows (keeping the last), sorts the index, and NaNs out
    every field of any bar that fails a fatal OHLC check. Fully-empty rows are
    dropped, so ``report.ok`` is ``True`` whenever the index is monotonic + UTC.
    """
    df = bars[~bars.index.duplicated(keep="last")].sort_index()
    bad = _bad_mask(df)
    if bad is not None and bool(bad.to_numpy().any()):
        if isinstance(df.columns, pd.MultiIndex):
            full = pd.DataFrame(
                {col: bad[col[1]] for col in df.columns}, index=df.index
            )
            full.columns = df.columns
        else:
            full = pd.DataFrame(
                {col: bad.iloc[:, 0] for col in df.columns}, index=df.index
            )
        df = df.mask(full).dropna(how="all")
    return df, check_quality(df)
