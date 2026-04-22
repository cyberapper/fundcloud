"""Benchmark-relative metrics: alpha, beta, capture ratios, Treynor, IR.

Every function accepts a portfolio return ``Series`` and a benchmark
``Series``. When a ``DataFrame`` of strategies is passed, the metric is
computed column-by-column against the same benchmark and returned as a
``Series`` indexed by strategy name.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from fundcloud._config import get_config

__all__ = [
    "alpha",
    "beta",
    "capture_ratio",
    "correlation",
    "down_capture",
    "information_ratio",
    "r_squared",
    "tracking_error",
    "treynor_ratio",
    "up_capture",
]


def _align(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    r = returns.to_frame() if isinstance(returns, pd.Series) else returns
    b = benchmark.reindex(r.index).dropna()
    r = r.loc[b.index]
    return r, b


def _periods(periods_per_year: int | None) -> int:
    return periods_per_year if periods_per_year is not None else get_config().periods_per_year


def _rf_per_period(risk_free: float | None, periods_per_year: int) -> float:
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    return rf / periods_per_year if rf else 0.0


def _collapse(out: pd.Series, original: pd.Series | pd.DataFrame) -> float | pd.Series:
    if isinstance(original, pd.Series):
        return float(out.iloc[0])
    return out


@overload
def beta(returns: pd.Series, benchmark: pd.Series) -> float: ...


@overload
def beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series: ...


def beta(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Regression beta: ``cov(r, benchmark) / var(benchmark)``."""
    r, b = _align(returns, benchmark)
    var_b = b.var(ddof=1)
    if not var_b or not np.isfinite(var_b):
        out = pd.Series(np.nan, index=r.columns, dtype=float)
        return _collapse(out, returns)
    covs = pd.Series({c: r[c].cov(b) for c in r.columns}, dtype=float)
    out = (covs / var_b).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def alpha(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Jensen's alpha (annualised).

    ``alpha = ann(r - rf) - beta * ann(benchmark - rf)`` where ``rf`` is the
    per-period risk-free rate. Positive alpha = the strategy beat what beta
    alone would predict.
    """
    ppy = _periods(periods_per_year)
    rf_pp = _rf_per_period(risk_free, ppy)
    r, b = _align(returns, benchmark)
    beta_vals = beta(r, b)
    if isinstance(beta_vals, float):
        beta_vals = pd.Series([beta_vals], index=r.columns)
    port_ann = (r.mean() - rf_pp) * ppy
    bench_ann = (b.mean() - rf_pp) * ppy
    out = (port_ann - beta_vals * bench_ann).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def r_squared(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Coefficient of determination of the strategy vs. the benchmark."""
    r, b = _align(returns, benchmark)
    out = pd.Series(
        {c: float(r[c].corr(b) ** 2) for c in r.columns},
        dtype=float,
    )
    return _collapse(out, returns)


def correlation(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Pearson correlation of ``returns`` with ``benchmark``.

    Unlike :func:`r_squared` which squares the coefficient, this keeps the
    sign so users can see anti-correlated strategies. Useful alongside beta
    for a quick read on market dependence.
    """
    r, b = _align(returns, benchmark)
    out = pd.Series(
        {c: float(r[c].corr(b)) for c in r.columns},
        dtype=float,
    )
    return _collapse(out, returns)


def information_ratio(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Active-return / active-volatility ratio.

    Mean of ``returns - benchmark`` divided by its sample standard deviation.
    """
    r, b = _align(returns, benchmark)
    active = r.sub(b, axis=0)
    mu = active.mean()
    sigma = active.std(ddof=1)
    out = (mu / sigma).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def tracking_error(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Annualised standard deviation of ``returns - benchmark``."""
    ppy = _periods(periods_per_year)
    r, b = _align(returns, benchmark)
    active = r.sub(b, axis=0)
    out = active.std(ddof=1) * np.sqrt(ppy)
    return _collapse(out, returns)


def up_capture(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Mean strategy return / mean benchmark return on benchmark-up periods."""
    r, b = _align(returns, benchmark)
    up_mask = b > 0
    if not up_mask.any():
        out = pd.Series(np.nan, index=r.columns, dtype=float)
        return _collapse(out, returns)
    up_bench_mean = b.loc[up_mask].mean()
    out = pd.Series(
        {
            c: r.loc[up_mask, c].mean() / up_bench_mean if up_bench_mean else np.nan
            for c in r.columns
        },
        dtype=float,
    )
    return _collapse(out, returns)


def down_capture(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Mean strategy return / mean benchmark return on benchmark-down periods.

    Lower is better (less participation in the drawdown).
    """
    r, b = _align(returns, benchmark)
    down_mask = b < 0
    if not down_mask.any():
        out = pd.Series(np.nan, index=r.columns, dtype=float)
        return _collapse(out, returns)
    down_bench_mean = b.loc[down_mask].mean()
    out = pd.Series(
        {
            c: r.loc[down_mask, c].mean() / down_bench_mean if down_bench_mean else np.nan
            for c in r.columns
        },
        dtype=float,
    )
    return _collapse(out, returns)


def capture_ratio(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """``up_capture / down_capture`` — Morningstar-style single number."""
    up = up_capture(returns, benchmark)
    down = down_capture(returns, benchmark)
    if isinstance(returns, pd.Series):
        if down == 0.0 or not np.isfinite(down):
            return float("nan")
        return float(up) / float(down)
    out = up / down
    return out.replace([np.inf, -np.inf], np.nan)


def treynor_ratio(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """``(annualised excess return) / beta``.

    Per-unit-of-market-risk premium, analogous to Sharpe but using beta in
    place of volatility.
    """
    ppy = _periods(periods_per_year)
    rf_pp = _rf_per_period(risk_free, ppy)
    r, b = _align(returns, benchmark)
    ann_excess = (r.mean() - rf_pp) * ppy
    beta_vals = beta(r, b)
    if isinstance(beta_vals, float):
        beta_vals = pd.Series([beta_vals], index=r.columns)
    out = (ann_excess / beta_vals).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)
