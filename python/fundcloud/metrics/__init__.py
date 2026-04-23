"""Portfolio analytics.

Free functions on returns. Every function accepts a ``pd.Series`` (single
strategy) or a ``pd.DataFrame`` (panel of strategies as columns) and
returns the same shape: a scalar stays a scalar for Series input, a
Series indexed by column name for DataFrame input.

Organised into six concerns:

* :mod:`fundcloud.metrics.core` — scalar metrics independent of a
  benchmark (return, risk, risk-adjusted, higher moments).
* :mod:`fundcloud.metrics.benchmark` — benchmark-relative metrics
  (alpha, beta, capture ratios, Treynor, information ratio).
* :mod:`fundcloud.metrics.periods` — calendar-period aggregates
  (monthly / yearly tables, best/worst, positive/negative counts).
* :mod:`fundcloud.metrics.rolling` — rolling-window metric series
  (Sharpe / Sortino / volatility / beta / drawdown).
* :mod:`fundcloud.metrics.summary` — :func:`metrics` one-shot bundle
  and :func:`drawdown_details` episode table.
* :mod:`fundcloud.metrics.batch` — GIL-released batch variants over
  large panels via the Rust kernels.
"""

from __future__ import annotations

from fundcloud.metrics.batch import (
    batch_cvar,
    batch_max_drawdown,
    batch_sharpe,
    batch_sortino,
    batch_summary,
)
from fundcloud.metrics.benchmark import (
    alpha,
    beta,
    capture_ratio,
    correlation,
    down_capture,
    information_ratio,
    r_squared,
    tracking_error,
    treynor_ratio,
    up_capture,
)
from fundcloud.metrics.core import (
    adjusted_sortino,
    avg_loss,
    avg_return,
    avg_win,
    best,
    cagr,
    calmar,
    common_sense_ratio,
    consecutive_losses,
    consecutive_wins,
    cvar,
    downside_volatility,
    drawdown_series,
    exposure,
    gain_to_pain_ratio,
    kelly_criterion,
    kurtosis,
    max_drawdown,
    omega,
    pain_index,
    pain_ratio,
    payoff_ratio,
    probabilistic_sharpe_ratio,
    profit_factor,
    returns_stats,
    risk_of_ruin,
    sharpe,
    skew,
    smart_sharpe,
    smart_sortino,
    sortino,
    tail_ratio,
    total_return,
    ulcer_index,
    ulcer_performance_index,
    value_at_risk,
    volatility,
    win_rate,
    worst,
)
from fundcloud.metrics.periods import (
    best_month,
    best_year,
    monthly_returns,
    negative_months,
    period_returns,
    positive_months,
    worst_month,
    worst_year,
    yearly_returns,
)
from fundcloud.metrics.rolling import (
    rolling_alpha,
    rolling_beta,
    rolling_drawdown,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
)
from fundcloud.metrics.summary import drawdown_details, metrics, runup_details

__all__ = [
    "adjusted_sortino",
    "alpha",
    "avg_loss",
    "avg_return",
    "avg_win",
    "batch_cvar",
    "batch_max_drawdown",
    "batch_sharpe",
    "batch_sortino",
    "batch_summary",
    "best",
    "best_month",
    "best_year",
    "beta",
    "cagr",
    "calmar",
    "capture_ratio",
    "common_sense_ratio",
    "consecutive_losses",
    "consecutive_wins",
    "correlation",
    "cvar",
    "down_capture",
    "downside_volatility",
    "drawdown_details",
    "drawdown_series",
    "exposure",
    "gain_to_pain_ratio",
    "information_ratio",
    "kelly_criterion",
    "kurtosis",
    "max_drawdown",
    "metrics",
    "monthly_returns",
    "negative_months",
    "omega",
    "pain_index",
    "pain_ratio",
    "payoff_ratio",
    "period_returns",
    "positive_months",
    "probabilistic_sharpe_ratio",
    "profit_factor",
    "r_squared",
    "returns_stats",
    "risk_of_ruin",
    "rolling_alpha",
    "rolling_beta",
    "rolling_drawdown",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_volatility",
    "runup_details",
    "sharpe",
    "skew",
    "smart_sharpe",
    "smart_sortino",
    "sortino",
    "tail_ratio",
    "total_return",
    "tracking_error",
    "treynor_ratio",
    "ulcer_index",
    "ulcer_performance_index",
    "up_capture",
    "value_at_risk",
    "volatility",
    "win_rate",
    "worst",
    "worst_month",
    "worst_year",
    "yearly_returns",
]
