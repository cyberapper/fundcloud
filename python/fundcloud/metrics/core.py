"""Core portfolio performance metrics.

Free functions on returns. Every function accepts a ``pd.Series`` (single
strategy) or a ``pd.DataFrame`` (panel of strategies as columns) and returns
the same shape: a scalar becomes a ``pd.Series`` indexed by column name.

Naming and formulas are chosen to match the finance standard and to line up
with skfolio / quantstats where those two agree.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from fundcloud._config import get_config

__all__ = [
    "adjusted_sortino",
    "avg_loss",
    "avg_return",
    "avg_win",
    "best",
    "cagr",
    "calmar",
    "common_sense_ratio",
    "consecutive_losses",
    "consecutive_wins",
    "cvar",
    "downside_volatility",
    "drawdown_series",
    "exposure",
    "gain_to_pain_ratio",
    "kelly_criterion",
    "kurtosis",
    "max_drawdown",
    "omega",
    "pain_index",
    "pain_ratio",
    "payoff_ratio",
    "probabilistic_sharpe_ratio",
    "profit_factor",
    "returns_stats",
    "risk_of_ruin",
    "sharpe",
    "skew",
    "smart_sharpe",
    "smart_sortino",
    "sortino",
    "tail_ratio",
    "total_return",
    "ulcer_index",
    "ulcer_performance_index",
    "value_at_risk",
    "volatility",
    "win_rate",
    "worst",
]


# ---------------------------------------------------------------------------
# internal helpers


def _to_df(r: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(r, pd.Series):
        return r.to_frame(name=r.name or "strategy")
    return r


def _collapse(out: pd.Series, original: pd.Series | pd.DataFrame) -> float | pd.Series:
    if isinstance(original, pd.Series):
        return float(out.iloc[0])
    return out


def _periods(periods_per_year: int | None) -> int:
    return periods_per_year if periods_per_year is not None else get_config().periods_per_year


def _rf_per_period(risk_free: float | None, periods_per_year: int) -> float:
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    return rf / periods_per_year if rf else 0.0


# ---------------------------------------------------------------------------
# public API


@overload
def sharpe(
    returns: pd.Series,
    *,
    risk_free: float | None = ...,
    periods_per_year: int | None = ...,
) -> float: ...


@overload
def sharpe(
    returns: pd.DataFrame,
    *,
    risk_free: float | None = ...,
    periods_per_year: int | None = ...,
) -> pd.Series: ...


def sharpe(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Annualised Sharpe ratio.

    Uses the **sample** standard deviation (``ddof=1``). Returns are assumed
    to be simple per-period returns; for log returns the formula is the same
    numerator and denominator.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r = pd.Series(rng.normal(0.0005, 0.01, 252))
    >>> round(sharpe(r, periods_per_year=252), 2)  # doctest: +SKIP
    0.7
    >>> # Works on a DataFrame too — returns a Series indexed by column:
    >>> panel = pd.DataFrame({"a": r, "b": -r})
    >>> isinstance(sharpe(panel), pd.Series)
    True
    """
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    rf_pp = _rf_per_period(risk_free, ppy)
    excess = df - rf_pp
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    out = (mu / sigma) * np.sqrt(ppy)
    out = out.replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


@overload
def sortino(
    returns: pd.Series,
    *,
    target: float = ...,
    periods_per_year: int | None = ...,
) -> float: ...


@overload
def sortino(
    returns: pd.DataFrame,
    *,
    target: float = ...,
    periods_per_year: int | None = ...,
) -> pd.Series: ...


def sortino(
    returns: pd.Series | pd.DataFrame,
    *,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Annualised Sortino ratio.

    Downside deviation uses only periods with returns strictly below
    ``target`` and divides by the sample count (``ddof=0``).
    """
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    diff = df - target
    downside = diff.clip(upper=0.0)
    # pop std with mean=0 => sqrt(mean(x^2))
    dd = np.sqrt((downside**2).mean())
    mu = diff.mean()
    out = (mu / dd) * np.sqrt(ppy)
    out = out.replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


@overload
def drawdown_series(returns: pd.Series) -> pd.Series: ...


@overload
def drawdown_series(returns: pd.DataFrame) -> pd.DataFrame: ...


def drawdown_series(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Drawdown at each timestamp: ``wealth / running_max - 1``.

    Always ≤ 0.
    """
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


@overload
def max_drawdown(returns: pd.Series) -> float: ...


@overload
def max_drawdown(returns: pd.DataFrame) -> pd.Series: ...


def max_drawdown(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Largest peak-to-trough loss (negative number)."""
    dd = drawdown_series(_to_df(returns))
    out = dd.min()
    return _collapse(out, returns)


@overload
def calmar(
    returns: pd.Series,
    *,
    periods_per_year: int | None = ...,
) -> float: ...


@overload
def calmar(
    returns: pd.DataFrame,
    *,
    periods_per_year: int | None = ...,
) -> pd.Series: ...


def calmar(
    returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Annualised return divided by absolute max drawdown."""
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    ann_ret = (1.0 + df).prod() ** (ppy / max(len(df), 1)) - 1.0
    mdd = max_drawdown(df).abs()
    out = ann_ret / mdd
    out = out.replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


@overload
def ulcer_index(returns: pd.Series) -> float: ...


@overload
def ulcer_index(returns: pd.DataFrame) -> pd.Series: ...


def ulcer_index(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Ulcer Index: RMS of drawdowns, in percent."""
    dd_pct = drawdown_series(_to_df(returns)) * 100.0
    out = np.sqrt((dd_pct**2).mean())
    return _collapse(out, returns)


@overload
def value_at_risk(returns: pd.Series, *, alpha: float = ...) -> float: ...


@overload
def value_at_risk(returns: pd.DataFrame, *, alpha: float = ...) -> pd.Series: ...


def value_at_risk(returns: pd.Series | pd.DataFrame, *, alpha: float = 0.95) -> float | pd.Series:
    """Historical Value-at-Risk at confidence ``alpha``.

    Returns a **loss** as a negative number (the (1-alpha) quantile of returns).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    df = _to_df(returns)
    out = df.quantile(1.0 - alpha)
    return _collapse(out, returns)


@overload
def cvar(returns: pd.Series, *, alpha: float = ...) -> float: ...


@overload
def cvar(returns: pd.DataFrame, *, alpha: float = ...) -> pd.Series: ...


def cvar(returns: pd.Series | pd.DataFrame, *, alpha: float = 0.95) -> float | pd.Series:
    """Conditional Value-at-Risk (Expected Shortfall) at confidence ``alpha``.

    Returns a **loss** as a negative number — the mean of returns below the
    ``(1 - alpha)`` quantile.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    df = _to_df(returns)
    q = df.quantile(1.0 - alpha)
    out = pd.Series(index=df.columns, dtype=float)
    for c in df.columns:
        mask = df[c] <= q[c]
        out[c] = df.loc[mask, c].mean() if mask.any() else np.nan
    return _collapse(out, returns)


@overload
def omega(
    returns: pd.Series,
    *,
    target: float = ...,
) -> float: ...


@overload
def omega(
    returns: pd.DataFrame,
    *,
    target: float = ...,
) -> pd.Series: ...


def omega(returns: pd.Series | pd.DataFrame, *, target: float = 0.0) -> float | pd.Series:
    """Omega ratio at ``target`` threshold.

    Ratio of the expected gain above target to expected loss below.
    """
    df = _to_df(returns)
    diff = df - target
    gains = diff.clip(lower=0.0).sum()
    losses = -diff.clip(upper=0.0).sum()
    out = gains / losses
    out = out.replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def returns_stats(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
    cvar_alpha: float = 0.95,
) -> pd.DataFrame:
    """Bundle of the common metrics into a single, scannable summary table.

    Rows are metrics, columns are strategies.
    """
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    n = len(df)
    total_return = (1.0 + df).prod() - 1.0
    cagr = (1.0 + df).prod() ** (ppy / max(n, 1)) - 1.0
    ann_vol = df.std(ddof=1) * np.sqrt(ppy)
    rows = {
        "periods": pd.Series(n, index=df.columns),
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe(df, risk_free=risk_free, periods_per_year=ppy),
        "sortino": sortino(df, periods_per_year=ppy),
        "calmar": calmar(df, periods_per_year=ppy),
        "max_drawdown": max_drawdown(df),
        "ulcer_index": ulcer_index(df),
        "cvar": cvar(df, alpha=cvar_alpha),
        "omega": omega(df),
    }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Extended scalar metrics: return, risk, risk-adjusted, moments.
#
# Formulas match the quantstats / PyPortfolioOpt / empyrical consensus and are
# vectorised across DataFrame columns. Every function accepts a Series or a
# DataFrame and collapses to a scalar / Series respectively.


def total_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Cumulative return over the sample.

    Compounded: ``prod(1 + r) - 1``.
    """
    df = _to_df(returns)
    out = (1.0 + df).prod() - 1.0
    return _collapse(out, returns)


def cagr(
    returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Compound annual growth rate.

    ``(1 + total_return) ** (periods_per_year / n) - 1``.
    """
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    n = max(len(df), 1)
    out = (1.0 + df).prod() ** (ppy / n) - 1.0
    return _collapse(out, returns)


def volatility(
    returns: pd.Series | pd.DataFrame,
    *,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Annualised sample standard deviation of returns."""
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    out = df.std(ddof=1) * np.sqrt(ppy)
    return _collapse(out, returns)


def downside_volatility(
    returns: pd.Series | pd.DataFrame,
    *,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Annualised downside deviation below ``target``."""
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    downside = (df - target).clip(upper=0.0)
    out = np.sqrt((downside**2).mean()) * np.sqrt(ppy)
    return _collapse(out, returns)


def skew(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Fisher skewness (pandas default, bias-adjusted)."""
    df = _to_df(returns)
    out = df.skew()
    return _collapse(out, returns)


def kurtosis(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Excess kurtosis (Fisher definition: normal distribution → 0)."""
    df = _to_df(returns)
    out = df.kurtosis()
    return _collapse(out, returns)


def avg_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Arithmetic mean per-period return."""
    df = _to_df(returns)
    return _collapse(df.mean(), returns)


def best(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Best single-period return in the sample."""
    df = _to_df(returns)
    return _collapse(df.max(), returns)


def worst(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Worst single-period return in the sample."""
    df = _to_df(returns)
    return _collapse(df.min(), returns)


def win_rate(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Fraction of periods with a strictly positive return."""
    df = _to_df(returns)
    out = (df > 0.0).mean()
    return _collapse(out, returns)


def avg_win(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Mean of winning-period returns."""
    df = _to_df(returns)
    out = pd.Series(
        {c: df.loc[df[c] > 0, c].mean() if (df[c] > 0).any() else np.nan for c in df.columns},
        dtype=float,
    )
    return _collapse(out, returns)


def avg_loss(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Mean of losing-period returns (negative)."""
    df = _to_df(returns)
    out = pd.Series(
        {c: df.loc[df[c] < 0, c].mean() if (df[c] < 0).any() else np.nan for c in df.columns},
        dtype=float,
    )
    return _collapse(out, returns)


def payoff_ratio(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """``avg_win / |avg_loss|`` — expected $ gained per $ lost on a single trade."""
    df = _to_df(returns)
    wins = avg_win(df)
    losses = avg_loss(df)
    if not isinstance(wins, pd.Series):
        wins = pd.Series([wins], index=df.columns)
        losses = pd.Series([losses], index=df.columns)
    out = (wins / losses.abs()).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def profit_factor(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """``sum(gains) / |sum(losses)|`` — dollar-weighted version of payoff_ratio."""
    df = _to_df(returns)
    gains = df.clip(lower=0.0).sum()
    losses = -df.clip(upper=0.0).sum()
    out = (gains / losses).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def exposure(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Fraction of periods with a non-zero return (strategy deployed)."""
    df = _to_df(returns)
    out = (df != 0.0).mean()
    return _collapse(out, returns)


def _max_streak(series: pd.Series, positive: bool) -> int:
    mask = (series > 0.0) if positive else (series < 0.0)
    best_streak = 0
    current = 0
    for hit in mask.to_numpy():
        if hit:
            current += 1
            best_streak = max(best_streak, current)
        else:
            current = 0
    return best_streak


def consecutive_wins(returns: pd.Series | pd.DataFrame) -> int | pd.Series:
    """Longest streak of consecutive positive-return periods."""
    df = _to_df(returns)
    out = pd.Series({c: _max_streak(df[c], positive=True) for c in df.columns}, dtype=int)
    if isinstance(returns, pd.Series):
        return int(out.iloc[0])
    return out


def consecutive_losses(returns: pd.Series | pd.DataFrame) -> int | pd.Series:
    """Longest streak of consecutive negative-return periods."""
    df = _to_df(returns)
    out = pd.Series({c: _max_streak(df[c], positive=False) for c in df.columns}, dtype=int)
    if isinstance(returns, pd.Series):
        return int(out.iloc[0])
    return out


def kelly_criterion(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Kelly fraction: ``win_rate - (1 - win_rate) / payoff_ratio``."""
    wr = win_rate(returns)
    pr = payoff_ratio(returns)
    if isinstance(returns, pd.Series):
        wr_s, pr_s = float(wr), float(pr)
        if not np.isfinite(pr_s) or pr_s == 0.0:
            return float("nan")
        return wr_s - (1.0 - wr_s) / pr_s
    wr_ = wr
    pr_ = pr
    out = wr_ - (1.0 - wr_) / pr_
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def risk_of_ruin(
    returns: pd.Series | pd.DataFrame,
    *,
    starting_capital: float = 1.0,
    ruin_level: float = 0.0,
) -> float | pd.Series:
    """Empirical risk of ruin over the sample.

    Fraction of sliding-window wealth paths that dip below ``ruin_level``
    (as a share of ``starting_capital``). Useful as a soft check on
    strategies that otherwise look good on Sharpe.
    """
    df = _to_df(returns)
    out_vals = {}
    threshold = starting_capital * (1.0 - ruin_level)
    for c in df.columns:
        wealth = starting_capital * (1.0 + df[c]).cumprod()
        hits = (wealth < threshold).sum()
        out_vals[c] = float(hits) / max(len(wealth), 1)
    out = pd.Series(out_vals, dtype=float)
    return _collapse(out, returns)


def tail_ratio(
    returns: pd.Series | pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> float | pd.Series:
    """Right-tail quantile / absolute left-tail quantile.

    ``alpha=0.05`` gives the 95%/5% ratio. Higher is better (fatter upside).
    """
    df = _to_df(returns)
    lower = df.quantile(alpha).abs()
    upper = df.quantile(1.0 - alpha).abs()
    out = (upper / lower).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def common_sense_ratio(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """``tail_ratio × profit_factor``.

    A pragmatic 'is this strategy really good?' score. Popularised by
    Laurent Bernut: punishes fat-left-tail / thin-right-tail distributions
    that happen to look statistically fine.
    """
    tr = tail_ratio(returns)
    pf = profit_factor(returns)
    if isinstance(returns, pd.Series):
        return float(tr) * float(pf)
    out = tr * pf
    return out.replace([np.inf, -np.inf], np.nan)


def pain_index(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Mean absolute drawdown over the sample."""
    df = _to_df(returns)
    out = drawdown_series(df).abs().mean()
    return _collapse(out, returns)


def pain_ratio(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Zephyr pain ratio: ``(CAGR - risk_free) / pain_index``."""
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    growth = cagr(df, periods_per_year=ppy)
    pain = pain_index(df)
    if isinstance(growth, float):
        growth = pd.Series([growth], index=df.columns)
        pain = pd.Series([pain], index=df.columns)
    out = ((growth - rf) / pain).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def gain_to_pain_ratio(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Jack Schwager's Gain-to-Pain: ``sum(returns) / |sum(negative returns)|``."""
    df = _to_df(returns)
    gains = df.sum()
    losses = df.clip(upper=0.0).sum().abs()
    out = (gains / losses).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def ulcer_performance_index(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Martin ratio: ``(CAGR - risk_free) / ulcer_index``."""
    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    rf = risk_free if risk_free is not None else get_config().risk_free_rate
    growth = cagr(df, periods_per_year=ppy)
    ulcer = ulcer_index(df)
    if isinstance(growth, float):
        growth = pd.Series([growth], index=df.columns)
        ulcer = pd.Series([ulcer], index=df.columns)
    # ulcer_index returns percent; rescale to match the CAGR scale.
    out = ((growth - rf) / (ulcer / 100.0)).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def adjusted_sortino(
    returns: pd.Series | pd.DataFrame,
    *,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Pedersen-adjusted Sortino: ``Sortino / sqrt(2)``.

    Makes Sortino comparable in scale with Sharpe when the downside
    standard deviation halves the full-sample standard deviation by
    construction.
    """
    sr = sortino(returns, target=target, periods_per_year=periods_per_year)
    if isinstance(sr, float):
        return sr / np.sqrt(2.0)
    return sr / np.sqrt(2.0)


def probabilistic_sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    *,
    target_sharpe: float = 0.0,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Bailey & López de Prado's Probabilistic Sharpe Ratio.

    Probability that the observed Sharpe exceeds ``target_sharpe``, given
    the sample length and the non-normality of returns (skew + excess
    kurtosis). Returns a value in [0, 1].
    """
    from scipy.stats import norm

    df = _to_df(returns)
    ppy = _periods(periods_per_year)
    sr = sharpe(df, periods_per_year=ppy)
    if isinstance(sr, float):
        sr = pd.Series([sr], index=df.columns)
    sr_per_period = sr / np.sqrt(ppy)
    target_per_period = target_sharpe / np.sqrt(ppy)
    n = len(df)
    sk = df.skew()
    ku = df.kurtosis()  # Fisher excess
    denom = np.sqrt(1.0 - sk * sr_per_period + ((ku) / 4.0) * (sr_per_period**2))
    z = (sr_per_period - target_per_period) * np.sqrt(n - 1) / denom
    out = pd.Series(norm.cdf(z), index=sr.index).replace([np.inf, -np.inf], np.nan)
    return _collapse(out, returns)


def _autocorr_penalty(series: pd.Series, *, max_lag: int = 3) -> float:
    if len(series) < max_lag + 2:
        return 1.0
    rhos = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]
    rhos = [r for r in rhos if np.isfinite(r)]
    if not rhos:
        return 1.0
    # Lo (2002) autocorrelation penalty: shrink by sqrt(1 + 2 * sum(rho)).
    rho_sum = float(np.sum(rhos))
    penalty = np.sqrt(max(1.0 + 2.0 * rho_sum, 1e-8))
    return 1.0 / penalty


def smart_sharpe(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free: float | None = None,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Sharpe scaled by Lo's serial-correlation penalty.

    Strategies whose returns autocorrelate have inflated naïve Sharpes;
    ``smart_sharpe`` dampens by ``1 / sqrt(1 + 2 * sum(rho_k))``.
    """
    df = _to_df(returns)
    base = sharpe(df, risk_free=risk_free, periods_per_year=periods_per_year)
    if isinstance(base, float):
        base = pd.Series([base], index=df.columns)
    penalty = pd.Series({c: _autocorr_penalty(df[c]) for c in df.columns}, dtype=float)
    out = base * penalty
    return _collapse(out, returns)


def smart_sortino(
    returns: pd.Series | pd.DataFrame,
    *,
    target: float = 0.0,
    periods_per_year: int | None = None,
) -> float | pd.Series:
    """Sortino scaled by Lo's serial-correlation penalty (see :func:`smart_sharpe`)."""
    df = _to_df(returns)
    base = sortino(df, target=target, periods_per_year=periods_per_year)
    if isinstance(base, float):
        base = pd.Series([base], index=df.columns)
    penalty = pd.Series({c: _autocorr_penalty(df[c]) for c in df.columns}, dtype=float)
    out = base * penalty
    return _collapse(out, returns)
