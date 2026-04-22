"""Rolling-window metric series.

Each function returns a ``Series`` (or ``DataFrame`` if the input was
multi-column) of rolling metric values — handy as inputs to the plot
builders or as raw columns you can drop into a feature pipeline.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from fundcloud._config import get_config

__all__ = [
    "rolling_alpha",
    "rolling_beta",
    "rolling_drawdown",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_volatility",
]


def _periods(periods_per_year: int | None) -> int:
    return periods_per_year if periods_per_year is not None else get_config().periods_per_year


@overload
def rolling_sharpe(
    returns: pd.Series,
    *,
    window: int = ...,
    risk_free: float | None = ...,
    periods_per_year: int | None = ...,
) -> pd.Series: ...


@overload
def rolling_sharpe(
    returns: pd.DataFrame,
    *,
    window: int = ...,
    risk_free: float | None = ...,
    periods_per_year: int | None = ...,
) -> pd.DataFrame: ...


def rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 63,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Rolling annualised Sharpe ratio."""
    ppy = _periods(periods_per_year)
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    rf_pp = rf / ppy if rf else 0.0
    excess = returns - rf_pp
    mu = excess.rolling(window).mean()
    sigma = excess.rolling(window).std(ddof=1)
    out = (mu / sigma) * np.sqrt(ppy)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_sortino(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 63,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Rolling annualised Sortino ratio (downside-only denominator)."""
    ppy = _periods(periods_per_year)
    diff = returns - target
    downside = diff.clip(upper=0.0)
    mu = diff.rolling(window).mean()
    dd = np.sqrt((downside**2).rolling(window).mean())
    out = (mu / dd) * np.sqrt(ppy)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    *,
    window: int = 63,
    periods_per_year: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Rolling annualised sample volatility."""
    ppy = _periods(periods_per_year)
    out = returns.rolling(window).std(ddof=1) * np.sqrt(ppy)
    return out


def rolling_beta(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 63,
) -> pd.Series | pd.DataFrame:
    """Rolling regression beta vs ``benchmark``.

    Returns and benchmark are inner-aligned before the rolling cov/var is
    taken — otherwise every date the benchmark doesn't trade introduces
    ``NaN`` into both operands, and pandas' ``rolling().cov()`` propagates
    the NaN across the entire window. That caused the flat/jagged lines
    on mixed-calendar pairs (e.g. BTC-USD vs NQ=F) before this fix.
    """
    if isinstance(returns, pd.Series):
        aligned = (
            pd.concat([returns.rename("__r"), benchmark.rename("__b")], axis=1, join="inner")
            .dropna()
        )
        r = aligned["__r"]
        b = aligned["__b"]
        cov_ab = r.rolling(window).cov(b)
        var_b = b.rolling(window).var(ddof=1)
        return (cov_ab / var_b).replace([np.inf, -np.inf], np.nan)

    out_df = pd.DataFrame(dtype=float)
    for col in returns.columns:
        aligned = (
            pd.concat(
                [returns[col].rename("__r"), benchmark.rename("__b")], axis=1, join="inner"
            )
            .dropna()
        )
        r = aligned["__r"]
        b = aligned["__b"]
        cov_ab = r.rolling(window).cov(b)
        var_b = b.rolling(window).var(ddof=1)
        out_df[col] = (cov_ab / var_b).replace([np.inf, -np.inf], np.nan)
    return out_df


def rolling_alpha(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    *,
    window: int = 63,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Rolling annualised Jensen alpha vs ``benchmark``.

    Computed on the common trading calendar — see :func:`rolling_beta` for
    why the inner-join matters on mixed-calendar pairs (e.g. 7-day crypto
    vs 5-day equity futures).
    """
    ppy = _periods(periods_per_year)
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    rf_pp = rf / ppy if rf else 0.0

    if isinstance(returns, pd.Series):
        aligned = (
            pd.concat([returns.rename("__r"), benchmark.rename("__b")], axis=1, join="inner")
            .dropna()
        )
        r = aligned["__r"]
        b = aligned["__b"]
        cov_ab = r.rolling(window).cov(b)
        var_b = b.rolling(window).var(ddof=1)
        beta_w = cov_ab / var_b
        r_mean = (r - rf_pp).rolling(window).mean()
        b_mean = (b - rf_pp).rolling(window).mean()
        out = (r_mean - beta_w * b_mean) * ppy
        return out.replace([np.inf, -np.inf], np.nan)

    out_df = pd.DataFrame(dtype=float)
    for col in returns.columns:
        aligned = (
            pd.concat(
                [returns[col].rename("__r"), benchmark.rename("__b")], axis=1, join="inner"
            )
            .dropna()
        )
        r = aligned["__r"]
        b = aligned["__b"]
        cov_ab = r.rolling(window).cov(b)
        var_b = b.rolling(window).var(ddof=1)
        beta_w = cov_ab / var_b
        r_mean = (r - rf_pp).rolling(window).mean()
        b_mean = (b - rf_pp).rolling(window).mean()
        out_df[col] = ((r_mean - beta_w * b_mean) * ppy).replace([np.inf, -np.inf], np.nan)
    return out_df


def rolling_drawdown(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Path-wise drawdown series: ``wealth / running_max - 1`` (always ≤ 0)."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0
