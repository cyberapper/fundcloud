"""One-shot summary: every metric we ship, in one call.

:func:`metrics` gathers ~50 standard portfolio statistics into a single
``pd.Series`` (one strategy) or ``pd.DataFrame`` (panel of strategies).
Meant to be the single entry point behind ``Portfolio.metrics()`` and
``.fc.metrics()``.

:func:`drawdown_details` returns one row per drawdown episode in the
sample (peak date, valley date, recovery date, depth, durations).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fundcloud._config import get_config
from fundcloud.metrics import benchmark as _bench
from fundcloud.metrics import core
from fundcloud.metrics import periods as _periods_mod

__all__ = ["drawdown_details", "metrics", "runup_details"]


def _periods(periods_per_year: int | None) -> int:
    return periods_per_year if periods_per_year is not None else get_config().periods_per_year


def _series_metrics(
    r: pd.Series,
    *,
    benchmark: pd.Series | None,
    risk_free: float,
    periods_per_year: int,
    cvar_alpha: float,
) -> dict[str, float]:
    d: dict[str, float] = {}
    # Return metrics
    d["periods"] = float(len(r))
    d["start"] = r.index.min() if len(r) else pd.NaT
    d["end"] = r.index.max() if len(r) else pd.NaT
    d["total_return"] = float(core.total_return(r))
    d["cagr"] = float(core.cagr(r, periods_per_year=periods_per_year))
    d["ann_volatility"] = float(core.volatility(r, periods_per_year=periods_per_year))
    d["downside_volatility"] = float(
        core.downside_volatility(r, periods_per_year=periods_per_year)
    )
    d["avg_return"] = float(core.avg_return(r))
    d["best"] = float(core.best(r))
    d["worst"] = float(core.worst(r))
    d["win_rate"] = float(core.win_rate(r))
    d["avg_win"] = float(core.avg_win(r))
    d["avg_loss"] = float(core.avg_loss(r))
    d["payoff_ratio"] = float(core.payoff_ratio(r))
    d["profit_factor"] = float(core.profit_factor(r))
    d["exposure"] = float(core.exposure(r))
    d["consecutive_wins"] = int(core.consecutive_wins(r))
    d["consecutive_losses"] = int(core.consecutive_losses(r))
    d["kelly_criterion"] = float(core.kelly_criterion(r))
    d["risk_of_ruin"] = float(core.risk_of_ruin(r))

    # Risk metrics
    d["skew"] = float(core.skew(r))
    d["kurtosis"] = float(core.kurtosis(r))
    d["tail_ratio"] = float(core.tail_ratio(r))
    d["common_sense_ratio"] = float(core.common_sense_ratio(r))
    d["pain_index"] = float(core.pain_index(r))
    d["pain_ratio"] = float(
        core.pain_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year)
    )
    d["gain_to_pain_ratio"] = float(core.gain_to_pain_ratio(r))
    d["max_drawdown"] = float(core.max_drawdown(r))
    d["ulcer_index"] = float(core.ulcer_index(r))
    d["ulcer_performance_index"] = float(
        core.ulcer_performance_index(r, risk_free=risk_free, periods_per_year=periods_per_year)
    )
    d["value_at_risk"] = float(core.value_at_risk(r, alpha=cvar_alpha))
    d["cvar"] = float(core.cvar(r, alpha=cvar_alpha))

    # Risk-adjusted
    d["sharpe"] = float(core.sharpe(r, risk_free=risk_free, periods_per_year=periods_per_year))
    d["sortino"] = float(core.sortino(r, periods_per_year=periods_per_year))
    d["calmar"] = float(core.calmar(r, periods_per_year=periods_per_year))
    d["omega"] = float(core.omega(r))
    d["adjusted_sortino"] = float(
        core.adjusted_sortino(r, periods_per_year=periods_per_year)
    )
    d["probabilistic_sharpe"] = float(
        core.probabilistic_sharpe_ratio(r, periods_per_year=periods_per_year)
    )
    d["smart_sharpe"] = float(
        core.smart_sharpe(r, risk_free=risk_free, periods_per_year=periods_per_year)
    )
    d["smart_sortino"] = float(core.smart_sortino(r, periods_per_year=periods_per_year))

    # Calendar-period stats (only when the index is datetime).
    if isinstance(r.index, pd.DatetimeIndex):
        try:
            d["best_month"] = _periods_mod.best_month(r)
            d["worst_month"] = _periods_mod.worst_month(r)
            d["best_year"] = _periods_mod.best_year(r)
            d["worst_year"] = _periods_mod.worst_year(r)
            d["positive_months"] = float(_periods_mod.positive_months(r))
            d["negative_months"] = float(_periods_mod.negative_months(r))
        except TypeError:
            pass

    # Benchmark-relative
    if benchmark is not None:
        d["alpha"] = float(
            _bench.alpha(
                r, benchmark, risk_free=risk_free, periods_per_year=periods_per_year
            )
        )
        d["beta"] = float(_bench.beta(r, benchmark))
        d["correlation"] = float(_bench.correlation(r, benchmark))
        d["r_squared"] = float(_bench.r_squared(r, benchmark))
        d["information_ratio"] = float(_bench.information_ratio(r, benchmark))
        d["tracking_error"] = float(
            _bench.tracking_error(r, benchmark, periods_per_year=periods_per_year)
        )
        d["up_capture"] = float(_bench.up_capture(r, benchmark))
        d["down_capture"] = float(_bench.down_capture(r, benchmark))
        d["capture_ratio"] = float(_bench.capture_ratio(r, benchmark))
        d["treynor_ratio"] = float(
            _bench.treynor_ratio(
                r, benchmark, risk_free=risk_free, periods_per_year=periods_per_year
            )
        )
    return d


def metrics(
    returns: pd.Series | pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
    cvar_alpha: float = 0.95,
) -> pd.Series | pd.DataFrame:
    """Return a single-pass bundle of every metric Fundcloud ships.

    Parameters
    ----------
    returns
        One-strategy ``Series`` or multi-strategy ``DataFrame``.
    benchmark
        Optional benchmark return ``Series``. When supplied, the output
        also includes alpha, beta, r_squared, information_ratio,
        tracking_error, up_capture, down_capture, capture_ratio, and
        treynor_ratio.
    risk_free
        Annualised risk-free rate. Defaults to
        :func:`fundcloud._config.get_config().risk_free_rate` (0).
    periods_per_year
        Annualisation factor; defaults to the global config.
    cvar_alpha
        Confidence level for VaR / CVaR (default 0.95).

    Returns
    -------
    ``pd.Series`` (single strategy) or ``pd.DataFrame`` (one column per
    strategy) with ~50 metrics. NaN is used for metrics that need a
    benchmark when none is provided.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("2024-01-02", periods=252, freq="B")
    >>> r = pd.Series(rng.normal(0.0005, 0.012, 252), index=idx, name="my_strat")
    >>> bench = pd.Series(rng.normal(0.0003, 0.010, 252), index=idx, name="SPY")
    >>> m = metrics(r, benchmark=bench)
    >>> len(m) >= 45
    True
    >>> {"sharpe", "max_drawdown", "cvar", "alpha", "beta"}.issubset(m.index)
    True
    """
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    ppy = _periods(periods_per_year)

    if isinstance(returns, pd.Series):
        data = _series_metrics(
            returns,
            benchmark=benchmark,
            risk_free=rf,
            periods_per_year=ppy,
            cvar_alpha=cvar_alpha,
        )
        name = returns.name or "strategy"
        return pd.Series(data, name=name)

    cols = {}
    for col in returns.columns:
        cols[col] = _series_metrics(
            returns[col],
            benchmark=benchmark,
            risk_free=rf,
            periods_per_year=ppy,
            cvar_alpha=cvar_alpha,
        )
    return pd.DataFrame(cols)


def drawdown_details(returns: pd.Series) -> pd.DataFrame:
    """Table of drawdown episodes.

    Each row is one peak-to-valley-to-recovery episode with columns:
    ``start`` (previous peak), ``valley`` (trough timestamp),
    ``recovery`` (first timestamp reaching the prior peak again; NaT if
    unrecovered within the sample), ``max_drawdown``, ``duration_days``,
    ``days_to_recover``.
    """
    if not isinstance(returns, pd.Series):
        msg = "drawdown_details requires a Series; use Population.summary() for panels."
        raise TypeError(msg)
    if returns.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "valley",
                "recovery",
                "max_drawdown",
                "duration_days",
                "days_to_recover",
            ]
        )
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    in_dd = drawdown < 0
    rows: list[dict[str, object]] = []
    cur_start: pd.Timestamp | None = None
    for ts in drawdown.index:
        if in_dd.loc[ts]:
            if cur_start is None:
                cur_start = ts
        else:
            if cur_start is not None:
                segment = drawdown.loc[cur_start:ts]
                valley = segment.idxmin()
                # "start" of this drawdown = last timestamp where peak changed,
                # which is the bar immediately before cur_start.
                prior_idx = returns.index.get_indexer([cur_start])[0] - 1
                start_ts = (
                    returns.index[prior_idx] if prior_idx >= 0 else returns.index[0]
                )
                rows.append(
                    {
                        "start": start_ts,
                        "valley": valley,
                        "recovery": ts,
                        "max_drawdown": float(drawdown.loc[valley]),
                        "duration_days": float((ts - start_ts).days),
                        "days_to_recover": float((ts - valley).days),
                    }
                )
                cur_start = None
    if cur_start is not None:
        segment = drawdown.loc[cur_start:]
        valley = segment.idxmin()
        prior_idx = returns.index.get_indexer([cur_start])[0] - 1
        start_ts = returns.index[prior_idx] if prior_idx >= 0 else returns.index[0]
        rows.append(
            {
                "start": start_ts,
                "valley": valley,
                "recovery": pd.NaT,
                "max_drawdown": float(drawdown.loc[valley]),
                "duration_days": float(
                    (returns.index[-1] - start_ts).days
                ),
                "days_to_recover": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("max_drawdown").reset_index(drop=True)


def runup_details(returns: pd.Series) -> pd.DataFrame:
    """Table of run-up (rally) episodes between drawdowns.

    Each row is one trough → peak → retreat episode with columns:
    ``start`` (prior trough; series start for the first episode),
    ``peak`` (highest wealth timestamp within the segment),
    ``end`` (first bar after ``peak`` where the next drawdown begins;
    ``NaT`` when the series ends in a still-ascending leg),
    ``max_runup``, ``duration_days``, ``days_after_peak``.

    The episodes are the exact complements of those returned by
    :func:`drawdown_details` — a drawdown fills the gap between two
    runups and vice versa.
    """
    if not isinstance(returns, pd.Series):
        msg = "runup_details requires a Series; use Population.summary() for panels."
        raise TypeError(msg)
    if returns.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "peak",
                "end",
                "max_runup",
                "duration_days",
                "days_after_peak",
            ]
        )
    wealth = (1.0 + returns).cumprod()
    dd = drawdown_details(returns)
    dd_sorted = (
        dd.sort_values("start").reset_index(drop=True) if not dd.empty else dd
    )

    rows: list[dict[str, object]] = []
    cur_ts: pd.Timestamp = wealth.index[0]
    for _, row in dd_sorted.iterrows():
        peak_ts = row["start"]
        if peak_ts > cur_ts:
            _append_runup_row(rows, wealth, start_ts=cur_ts, retreat_ts=peak_ts)
        cur_ts = (
            row["recovery"] if pd.notna(row["recovery"]) else wealth.index[-1]
        )
    if cur_ts < wealth.index[-1]:
        _append_runup_row(rows, wealth, start_ts=cur_ts, retreat_ts=pd.NaT)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("max_runup", ascending=False).reset_index(drop=True)


def _append_runup_row(
    rows: list[dict[str, object]],
    wealth: pd.Series,
    *,
    start_ts: pd.Timestamp,
    retreat_ts: pd.Timestamp | Any,
) -> None:
    """Emit one runup row if the peak within the segment is above the start."""
    end_ts = retreat_ts if pd.notna(retreat_ts) else wealth.index[-1]
    segment = wealth.loc[start_ts:end_ts]
    if segment.empty:
        return
    peak_ts = segment.idxmax()
    start_val = float(segment.iloc[0])
    peak_val = float(segment.loc[peak_ts])
    if peak_val <= start_val:
        return
    days_after = (
        float((retreat_ts - peak_ts).days) if pd.notna(retreat_ts) else float("nan")
    )
    rows.append(
        {
            "start": start_ts,
            "peak": peak_ts,
            "end": retreat_ts,
            "max_runup": peak_val / start_val - 1.0,
            "duration_days": float((peak_ts - start_ts).days),
            "days_after_peak": days_after,
        }
    )
