"""Calendar-period return aggregates.

Pivot a daily (or higher-freq) return stream into monthly and yearly
tables, then report best/worst/positive/negative counts.
"""

from __future__ import annotations

import pandas as pd

from fundcloud._config import get_config

__all__ = [
    "best_month",
    "best_year",
    "monthly_returns",
    "negative_months",
    "period_returns",
    "positive_months",
    "worst_month",
    "worst_year",
    "yearly_returns",
]


def _compound(s: pd.Series) -> float:
    if s.empty:
        return float("nan")
    return float((1.0 + s).prod() - 1.0)


def _resample(returns: pd.Series, rule: str) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        msg = "calendar-period aggregates require a DatetimeIndex on the returns Series"
        raise TypeError(msg)
    return returns.resample(rule).apply(_compound).dropna()


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year × month pivot of compounded monthly returns."""
    monthly = _resample(returns, "ME")
    if monthly.empty:
        return pd.DataFrame()
    idx: pd.DatetimeIndex = pd.DatetimeIndex(monthly.index)
    table = monthly.to_frame(name="ret")
    table["year"] = idx.year
    table["month"] = idx.month
    wide = table.pivot(index="year", columns="month", values="ret")
    wide.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in wide.columns]
    return wide


def yearly_returns(returns: pd.Series) -> pd.Series:
    """Compounded return per calendar year."""
    yearly = _resample(returns, "YE")
    idx: pd.DatetimeIndex = pd.DatetimeIndex(yearly.index)
    yearly.index = idx.year
    yearly.name = "yearly_return"
    return yearly


def best_month(returns: pd.Series) -> float:
    monthly = _resample(returns, "ME")
    return float(monthly.max()) if not monthly.empty else float("nan")


def worst_month(returns: pd.Series) -> float:
    monthly = _resample(returns, "ME")
    return float(monthly.min()) if not monthly.empty else float("nan")


def best_year(returns: pd.Series) -> float:
    yearly = _resample(returns, "YE")
    return float(yearly.max()) if not yearly.empty else float("nan")


def worst_year(returns: pd.Series) -> float:
    yearly = _resample(returns, "YE")
    return float(yearly.min()) if not yearly.empty else float("nan")


def positive_months(returns: pd.Series) -> int:
    monthly = _resample(returns, "ME")
    return int((monthly > 0).sum())


def negative_months(returns: pd.Series) -> int:
    monthly = _resample(returns, "ME")
    return int((monthly < 0).sum())


# Period-ending windows for :func:`period_returns`.
# Each tuple is ``(display_label, cutoff_spec, annualize_flag)``:
# * ``"mtd"`` / ``"ytd"`` are calendar-anchor sentinels,
# * a :class:`pandas.DateOffset` instance means ``anchor - offset``,
# * ``None`` means "all history".
_PERIOD_SPECS: list[tuple[str, object, bool]] = [
    ("MTD", "mtd", False),
    ("3M", pd.DateOffset(months=3), False),
    ("6M", pd.DateOffset(months=6), False),
    ("YTD", "ytd", False),
    ("1Y", pd.DateOffset(years=1), False),
    ("3Y (ann.)", pd.DateOffset(years=3), True),
    ("5Y (ann.)", pd.DateOffset(years=5), True),
    ("10Y (ann.)", pd.DateOffset(years=10), True),
    ("All-time (ann.)", None, True),
]


def _period_cutoff(anchor: pd.Timestamp, spec: object) -> pd.Timestamp | None:
    if spec == "mtd":
        return anchor.normalize().replace(day=1)
    if spec == "ytd":
        return anchor.normalize().replace(month=1, day=1)
    if spec is None:
        return None
    return anchor - spec  # pd.DateOffset


def _window_return(
    returns: pd.Series, cutoff: pd.Timestamp | None, *, annualize: bool, ppy: int
) -> float:
    window = returns if cutoff is None else returns.loc[cutoff:]
    if window.empty:
        return float("nan")
    total = float((1.0 + window).prod() - 1.0)
    if not annualize:
        return total
    years = len(window) / ppy
    if years <= 0:
        return float("nan")
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def period_returns(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    periods_per_year: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Compact MTD / 3M / 6M / YTD / 1Y / 3Y / 5Y / 10Y / All-time return table.

    Rows tagged ``(ann.)`` are annualised over their own window (CAGR).
    Windows shorter than 1Y annualise as ``NaN`` to avoid the
    exploding-CAGR artefact on tiny samples.

    Parameters
    ----------
    returns
        One-strategy :class:`pandas.Series` or panel :class:`pandas.DataFrame`
        (one column per strategy). Must use a ``DatetimeIndex``.
    benchmark
        Optional benchmark return :class:`pandas.Series`. When supplied,
        the output prepends a column for the benchmark.
    periods_per_year
        Annualisation factor; defaults to
        :func:`fundcloud._config.get_config().periods_per_year`.

    Returns
    -------
    ``pd.Series`` (single strategy, no benchmark) or ``pd.DataFrame``
    (otherwise) indexed by the nine period labels above.
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        msg = "period_returns requires a DatetimeIndex"
        raise TypeError(msg)
    ppy = periods_per_year if periods_per_year is not None else get_config().periods_per_year
    anchor = returns.index[-1] if len(returns) else pd.Timestamp.utcnow()

    def _per_series(s: pd.Series) -> pd.Series:
        vals = {
            label: _window_return(s, _period_cutoff(anchor, spec), annualize=ann, ppy=ppy)
            for label, spec, ann in _PERIOD_SPECS
        }
        return pd.Series(vals, name=s.name or "strategy")

    if isinstance(returns, pd.Series):
        strat = _per_series(returns)
        if benchmark is None:
            return strat
        bench = _per_series(benchmark)
        strat_name = str(returns.name) if returns.name is not None else "strategy"
        bench_name = str(benchmark.name) if benchmark.name is not None else "benchmark"
        return pd.concat([bench.rename(bench_name), strat.rename(strat_name)], axis=1)

    # DataFrame — one column per strategy.
    cols = {str(c): _per_series(returns[c]) for c in returns.columns}
    if benchmark is not None:
        bench_name = str(benchmark.name) if benchmark.name is not None else "benchmark"
        cols = {bench_name: _per_series(benchmark), **cols}
    return pd.DataFrame(cols)
