"""Metric display metadata shared by HTML / PDF / Excel renderers.

Every metric emitted by :func:`fundcloud.metrics.metrics` can declare:

* the user-facing label (spelled with finance-standard casing — ``CAGR``,
  ``CVaR``, not ``Cagr`` / ``Cvar``);
* a short plain-language definition;
* a one-line formula rendered in text (no MathJax — we want tear sheets to
  stay self-contained);
* the category it belongs to (Return / Risk / Risk-adjusted / Drawdown /
  Distribution / Trade / Calendar / Benchmark);
* a format hint — ``pct``, ``ratio``, ``int``, ``pct4`` (4 decimals).
* whether positive is good, negative is bad, or neutral — drives the
  green/red styling on stat cards and sidebar rows.

The registry is intentionally a plain dict so users can extend it from a
notebook without touching library internals::

    from fundcloud.reports.metric_info import METRIC_INFO, MetricInfo, Category
    METRIC_INFO["my_metric"] = MetricInfo(
        label="My metric",
        definition="...",
        formula="...",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["METRIC_INFO", "Category", "MetricInfo", "category_order"]


class Category(str, Enum):
    """Ordered categories in the tear-sheet sidebar."""

    RETURN = "Return"
    RISK = "Risk"
    RISK_ADJUSTED = "Risk-adjusted"
    DRAWDOWN = "Drawdown"
    DISTRIBUTION = "Distribution"
    TRADE = "Trade"
    CALENDAR = "Calendar"
    BENCHMARK = "Benchmark"


def category_order() -> list[Category]:
    """Return the canonical rendering order."""
    return [
        Category.RETURN,
        Category.RISK_ADJUSTED,
        Category.RISK,
        Category.DRAWDOWN,
        Category.DISTRIBUTION,
        Category.TRADE,
        Category.CALENDAR,
        Category.BENCHMARK,
    ]


@dataclass(slots=True, frozen=True)
class MetricInfo:
    """Static metadata for a single metric key."""

    label: str
    definition: str
    formula: str
    category: Category
    # Display hint: "pct" / "pct4" / "ratio" / "int".
    fmt: str = "ratio"
    # "pos" → green when value > 0; "neg" → red when value < 0; "signed"
    # → green when positive, red when negative; "neutral" → no colouring.
    sign: str = "neutral"


# --------------------------------------------------------------------- registry


METRIC_INFO: dict[str, MetricInfo] = {
    # --- Return ---
    "total_return": MetricInfo(
        label="Total return",
        definition="Compounded return over the whole sample, gross.",
        formula="∏(1 + r) − 1",
        category=Category.RETURN,
        fmt="pct",
        sign="signed",
    ),
    "cagr": MetricInfo(
        label="CAGR",
        definition="Compound annual growth rate — the constant annual rate that reproduces the total return.",
        formula="(1 + total_return)^(periods_per_year / n) − 1",
        category=Category.RETURN,
        fmt="pct",
        sign="signed",
    ),
    "avg_return": MetricInfo(
        label="Mean return",
        definition="Arithmetic mean of per-period returns.",
        formula="mean(r)",
        category=Category.RETURN,
        fmt="pct4",
        sign="signed",
    ),
    "best": MetricInfo(
        label="Best period",
        definition="Single best per-period return in the sample.",
        formula="max(r)",
        category=Category.RETURN,
        fmt="pct",
        sign="signed",
    ),
    "worst": MetricInfo(
        label="Worst period",
        definition="Single worst per-period return in the sample.",
        formula="min(r)",
        category=Category.RETURN,
        fmt="pct",
        sign="signed",
    ),
    # --- Risk ---
    "ann_volatility": MetricInfo(
        label="Volatility",
        definition="Annualised sample standard deviation of returns.",
        formula="σ(r) · √periods_per_year",
        category=Category.RISK,
        fmt="pct",
    ),
    "downside_volatility": MetricInfo(
        label="Downside volatility",
        definition="Annualised std-dev of returns below the target (0% by default).",
        formula="√mean(min(r − target, 0)²) · √periods_per_year",
        category=Category.RISK,
        fmt="pct",
    ),
    "skew": MetricInfo(
        label="Skewness",
        definition="Fisher skewness. Negative = fat left tail (big losses more frequent than big gains).",
        formula="E[(r − μ)³] / σ³",
        category=Category.RISK,
        fmt="ratio",
    ),
    "kurtosis": MetricInfo(
        label="Excess kurtosis",
        definition="Fisher excess kurtosis (0 = normal, > 0 = heavy tails).",
        formula="E[(r − μ)⁴] / σ⁴ − 3",
        category=Category.RISK,
        fmt="ratio",
    ),
    "tail_ratio": MetricInfo(
        label="Tail ratio",
        definition="Right-tail quantile over left-tail quantile — higher is better.",
        formula="|q(1 − α)| / |q(α)| (α = 0.05)",
        category=Category.RISK,
        fmt="ratio",
    ),
    "common_sense_ratio": MetricInfo(
        label="Common-sense ratio",
        definition="tail_ratio × profit_factor (Laurent Bernut). Punishes fat left tails.",
        formula="tail_ratio · profit_factor",
        category=Category.RISK,
        fmt="ratio",
    ),
    # --- Risk-adjusted ---
    "sharpe": MetricInfo(
        label="Sharpe",
        definition="Excess return over volatility, annualised. > 1 is strong, > 2 exceptional.",
        formula="(mean(r − rf) / σ(r − rf)) · √periods_per_year",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
        sign="signed",
    ),
    "sortino": MetricInfo(
        label="Sortino",
        definition="Like Sharpe but denominator uses downside deviation only.",
        formula="(mean(r) / downside_vol) · √periods_per_year",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
        sign="signed",
    ),
    "calmar": MetricInfo(
        label="Calmar",
        definition="CAGR divided by the absolute max drawdown.",
        formula="CAGR / |max_drawdown|",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
        sign="signed",
    ),
    "omega": MetricInfo(
        label="Omega",
        definition="Probability-weighted ratio of upside vs downside, relative to a target.",
        formula="Σ max(r − t, 0) / Σ max(t − r, 0)",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
    ),
    "adjusted_sortino": MetricInfo(
        label="Adjusted Sortino",
        definition="Pedersen-adjusted Sortino, comparable in scale with Sharpe.",
        formula="Sortino / √2",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
        sign="signed",
    ),
    "probabilistic_sharpe": MetricInfo(
        label="Probabilistic Sharpe",
        definition="Probability that the true Sharpe exceeds the target, given non-normality.",
        formula="Φ((Ŝ − S★) · √(n − 1) / ν), ν = √(1 − γ₃·Ŝ + γ₄/4·Ŝ²)",
        category=Category.RISK_ADJUSTED,
        fmt="pct",
    ),
    "smart_sharpe": MetricInfo(
        label="Smart Sharpe",
        definition="Sharpe scaled by Lo's serial-correlation penalty.",
        formula="Sharpe · (1 + 2·Σρₖ)^(−½)",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
        sign="signed",
    ),
    "smart_sortino": MetricInfo(
        label="Smart Sortino",
        definition="Sortino with the same serial-correlation penalty.",
        formula="Sortino · (1 + 2·Σρₖ)^(−½)",
        category=Category.RISK_ADJUSTED,
        fmt="ratio",
        sign="signed",
    ),
    # --- Drawdown ---
    "max_drawdown": MetricInfo(
        label="Max drawdown",
        definition="Largest peak-to-trough loss, as a negative percent.",
        formula="min(wealth / cummax(wealth) − 1)",
        category=Category.DRAWDOWN,
        fmt="pct",
        sign="neg",
    ),
    "ulcer_index": MetricInfo(
        label="Ulcer index",
        definition="Root-mean-square of drawdowns in percent — pain depth.",
        formula="√mean(dd%²)",
        category=Category.DRAWDOWN,
        fmt="ratio",
    ),
    "ulcer_performance_index": MetricInfo(
        label="UPI (Martin ratio)",
        definition="Risk-adjusted return using the ulcer index as risk proxy.",
        formula="(CAGR − rf) / (ulcer_index / 100)",
        category=Category.DRAWDOWN,
        fmt="ratio",
        sign="signed",
    ),
    "pain_index": MetricInfo(
        label="Pain index",
        definition="Mean absolute drawdown over the sample.",
        formula="mean(|drawdown|)",
        category=Category.DRAWDOWN,
        fmt="pct4",
    ),
    "pain_ratio": MetricInfo(
        label="Pain ratio",
        definition="Zephyr pain ratio — CAGR minus rf divided by pain index.",
        formula="(CAGR − rf) / pain_index",
        category=Category.DRAWDOWN,
        fmt="ratio",
        sign="signed",
    ),
    # --- Distribution / tail ---
    "value_at_risk": MetricInfo(
        label="VaR (95%)",
        definition="Historical value-at-risk — 5%-quantile loss (negative).",
        formula="q₍1 − α₎(r), α = 0.95",
        category=Category.DISTRIBUTION,
        fmt="pct",
        sign="neg",
    ),
    "cvar": MetricInfo(
        label="CVaR (95%)",
        definition="Conditional VaR — mean loss beyond the VaR threshold (expected shortfall).",
        formula="E[r | r ≤ VaR]",
        category=Category.DISTRIBUTION,
        fmt="pct",
        sign="neg",
    ),
    "gain_to_pain_ratio": MetricInfo(
        label="Gain-to-pain",
        definition="Schwager's gain-to-pain — sum of returns over sum of negative returns.",
        formula="Σr / |Σmin(r, 0)|",
        category=Category.DISTRIBUTION,
        fmt="ratio",
        sign="signed",
    ),
    # --- Trade stats ---
    "win_rate": MetricInfo(
        label="Win rate",
        definition="Fraction of periods with a strictly positive return.",
        formula="#{r > 0} / n",
        category=Category.TRADE,
        fmt="pct",
    ),
    "avg_win": MetricInfo(
        label="Avg win",
        definition="Mean return on winning periods.",
        formula="mean(r | r > 0)",
        category=Category.TRADE,
        fmt="pct4",
    ),
    "avg_loss": MetricInfo(
        label="Avg loss",
        definition="Mean return on losing periods (negative).",
        formula="mean(r | r < 0)",
        category=Category.TRADE,
        fmt="pct4",
        sign="neg",
    ),
    "payoff_ratio": MetricInfo(
        label="Payoff ratio",
        definition="avg_win ÷ |avg_loss| — expected dollars gained per dollar lost on a trade.",
        formula="avg_win / |avg_loss|",
        category=Category.TRADE,
        fmt="ratio",
    ),
    "profit_factor": MetricInfo(
        label="Profit factor",
        definition="Dollar-weighted version of payoff ratio.",
        formula="Σmax(r, 0) / |Σmin(r, 0)|",
        category=Category.TRADE,
        fmt="ratio",
    ),
    "exposure": MetricInfo(
        label="Exposure",
        definition="Fraction of periods with a non-zero return — how much the strategy was actually deployed.",
        formula="#{r ≠ 0} / n",
        category=Category.TRADE,
        fmt="pct",
    ),
    "kelly_criterion": MetricInfo(
        label="Kelly",
        definition="Kelly fraction implied by win rate and payoff ratio.",
        formula="p − (1 − p) / payoff_ratio",
        category=Category.TRADE,
        fmt="pct",
    ),
    "risk_of_ruin": MetricInfo(
        label="Risk of ruin",
        definition="Empirical share of sliding-window wealth paths that dip below the ruin level.",
        formula="#{wealth_t < starting · (1 − ruin_level)} / n",
        category=Category.TRADE,
        fmt="pct",
        sign="neg",
    ),
    "consecutive_wins": MetricInfo(
        label="Max consecutive wins",
        definition="Longest streak of consecutive positive-return periods.",
        formula="max streak of r > 0",
        category=Category.TRADE,
        fmt="int",
    ),
    "consecutive_losses": MetricInfo(
        label="Max consecutive losses",
        definition="Longest streak of consecutive negative-return periods.",
        formula="max streak of r < 0",
        category=Category.TRADE,
        fmt="int",
    ),
    # --- Calendar ---
    "best_month": MetricInfo(
        label="Best month",
        definition="Highest monthly compounded return in the sample.",
        formula="max month ∏(1 + r) − 1",
        category=Category.CALENDAR,
        fmt="pct",
        sign="signed",
    ),
    "worst_month": MetricInfo(
        label="Worst month",
        definition="Lowest monthly compounded return in the sample.",
        formula="min month ∏(1 + r) − 1",
        category=Category.CALENDAR,
        fmt="pct",
        sign="neg",
    ),
    "best_year": MetricInfo(
        label="Best year",
        definition="Highest yearly compounded return in the sample.",
        formula="max year ∏(1 + r) − 1",
        category=Category.CALENDAR,
        fmt="pct",
        sign="signed",
    ),
    "worst_year": MetricInfo(
        label="Worst year",
        definition="Lowest yearly compounded return in the sample.",
        formula="min year ∏(1 + r) − 1",
        category=Category.CALENDAR,
        fmt="pct",
        sign="neg",
    ),
    "positive_months": MetricInfo(
        label="Positive months",
        definition="Count of monthly-compounded returns > 0.",
        formula="#{monthly return > 0}",
        category=Category.CALENDAR,
        fmt="int",
    ),
    "negative_months": MetricInfo(
        label="Negative months",
        definition="Count of monthly-compounded returns < 0.",
        formula="#{monthly return < 0}",
        category=Category.CALENDAR,
        fmt="int",
    ),
    # --- Benchmark-relative ---
    "alpha": MetricInfo(
        label="Alpha",
        definition="Jensen's annualised alpha — excess return vs what beta predicts.",
        formula="ann(r − rf) − β · ann(bench − rf)",
        category=Category.BENCHMARK,
        fmt="pct",
        sign="signed",
    ),
    "beta": MetricInfo(
        label="Beta",
        definition="Regression slope of strategy on benchmark.",
        formula="cov(r, bench) / var(bench)",
        category=Category.BENCHMARK,
        fmt="ratio",
    ),
    "correlation": MetricInfo(
        label="Correlation",
        definition="Pearson correlation of strategy vs benchmark returns.",
        formula="cov(r, bench) / (σ(r) · σ(bench))",
        category=Category.BENCHMARK,
        fmt="ratio",
    ),
    "r_squared": MetricInfo(
        label="R²",
        definition="Share of strategy variance explained by benchmark variance.",
        formula="correlation²",
        category=Category.BENCHMARK,
        fmt="ratio",
    ),
    "information_ratio": MetricInfo(
        label="Information ratio",
        definition="Active mean return over active standard deviation (per-period).",
        formula="mean(r − bench) / σ(r − bench)",
        category=Category.BENCHMARK,
        fmt="ratio",
        sign="signed",
    ),
    "tracking_error": MetricInfo(
        label="Tracking error",
        definition="Annualised standard deviation of active returns.",
        formula="σ(r − bench) · √periods_per_year",
        category=Category.BENCHMARK,
        fmt="pct",
    ),
    "up_capture": MetricInfo(
        label="Up capture",
        definition="Strategy mean return on benchmark-up periods, over benchmark's mean.",
        formula="mean(r | bench > 0) / mean(bench | bench > 0)",
        category=Category.BENCHMARK,
        fmt="pct",
    ),
    "down_capture": MetricInfo(
        label="Down capture",
        definition="Same as up-capture but for benchmark-down periods. Lower is better.",
        formula="mean(r | bench < 0) / mean(bench | bench < 0)",
        category=Category.BENCHMARK,
        fmt="pct",
    ),
    "capture_ratio": MetricInfo(
        label="Capture ratio",
        definition="Morningstar-style single number = up_capture / down_capture.",
        formula="up_capture / down_capture",
        category=Category.BENCHMARK,
        fmt="ratio",
        sign="signed",
    ),
    "treynor_ratio": MetricInfo(
        label="Treynor",
        definition="Annualised excess return per unit of market beta.",
        formula="ann(r − rf) / β",
        category=Category.BENCHMARK,
        fmt="ratio",
        sign="signed",
    ),
}
