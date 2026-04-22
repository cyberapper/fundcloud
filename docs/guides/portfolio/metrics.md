# Portfolio metrics

Fundcloud exposes the same metric in three different surfaces so you can reach for whichever one matches the shape of the work you're doing, without ever rewriting the numerical core. Results are numerically identical across surfaces, and the Rust kernels accelerate all three when available.

| Surface | Best for | Example |
|---|---|---|
| Free function (`fundcloud.metrics.core`) | scripting, reuse inside custom estimators | `sharpe(returns, periods_per_year=252)` |
| `Portfolio` method | post-simulation analytics on a live `Portfolio` | `pf.sharpe()` |
| `.fc` pandas accessor | one-liners in a notebook, on any Series/DataFrame | `returns.fc.sharpe()` |

All three resolve to the same underlying kernel, and the `returns_stats` / `summary()` bundle (used by the tear sheet) is shared across surfaces — so a notebook exploration and a production estimator will report identical numbers.

## Core scalars

```python
from fundcloud.metrics.core import (
    sharpe, sortino, calmar, omega,
    max_drawdown, drawdown_series, ulcer_index,
    cvar, value_at_risk, returns_stats,
)

sharpe(returns, risk_free=0.0, periods_per_year=252)
sortino(returns, target=0.0)
calmar(returns)
omega(returns, target=0.0)
max_drawdown(returns)          # negative scalar
drawdown_series(returns)       # per-bar series, ≤ 0
cvar(returns, alpha=0.95)
value_at_risk(returns, alpha=0.95)
returns_stats(returns)         # bundle of all of the above
```

`returns_stats` is the one that powers tear-sheet tables.

## Batch / panel metrics

For running the same metric across a strategy grid:

```python
from fundcloud.metrics.batch import batch_sharpe, batch_summary

batch_sharpe({"slow": slow_returns, "fast": fast_returns})
batch_summary(dict_of_returns)  # one row per strategy
```

## The `.fc` accessor

```python
import fundcloud  # registers the accessor on pandas

returns.fc.sharpe()
returns.fc.drawdown_series()
returns.fc.metrics()           # a named Series of metrics
```

## `Portfolio` in a hurry

When you already have a `Portfolio` object (e.g. from the Simulator):

```python
pf = result.portfolio
pf.sharpe(); pf.max_drawdown(); pf.turnover(); pf.attribution()
pf.summary()       # identical shape to ``returns_stats``
pf.metrics()       # full ~55-metric bundle (fundcloud.metrics.metrics)
```

## Period & yearly breakdowns

For the "How did we do MTD / 3M / … / All-time?" and "what was 2024?"
surfaces that appear in the tear sheet:

```python
pf.period_returns(benchmark=spy_returns)
#                           SPY       Strategy
# MTD                  -0.01860      0.00255
# 3M                    0.12283     -0.04327
# 6M                   -0.05239     -0.07259
# YTD                  -0.09673     -0.05836
# 1Y                    0.23250     -0.03096
# 3Y (ann.)             0.14481     -0.05300
# 5Y (ann.)             0.21262      0.02923
# 10Y (ann.)            0.11988      0.02068
# All-time (ann.)       0.11988      0.02068

pf.yearly_returns(benchmark=spy_returns)  # one row per calendar year
```

`pf.period_returns` defaults to the `benchmark=` set at Portfolio
construction when no explicit benchmark is passed.

## Drawdowns & runups

Every equity curve has two symmetric structural stories — the drawdowns
(peak → valley → recovery) and the runups (trough → peak → retreat) that
fill the gaps between them. Both are available as full episode tables
and as display-formatted "worst-N" / "top-N" views:

```python
pf.drawdown_details()       # start / valley / recovery / max_drawdown / duration_days / days_to_recover
pf.runup_details()          # start / peak / end / max_runup / duration_days / days_after_peak

pf.worst_drawdowns(top=10)  # Started / Recovered / Drawdown / Days
pf.worst_runups(top=10)     # Started / Peaked / Runup / Days
```

The tear sheet renders the latter two as tables in HTML, PDF, and Excel.

The Rust backend transparently accelerates the batch variants when
`fundcloud.kernels.HAS_RUST` is True — see
[Rust kernels](../accelerators/rust-kernels.md).

## Trade statistics

These metrics are most informative for **active trading strategies** where the
number and quality of individual trades matter, not just the equity curve shape.

```python
import fundcloud  # registers .fc on pandas

# What fraction of periods had positive returns?
returns.fc.win_rate()        # e.g. 0.54 → 54% of days were green

# Average magnitude of winners vs losers
returns.fc.avg_win()         # e.g. 0.0082 → +0.82% on winning days
returns.fc.avg_loss()        # e.g. -0.0065 → –0.65% on losing days

# Payoff quality
returns.fc.payoff_ratio()    # avg_win / |avg_loss| — >1 means winners bigger than losers
returns.fc.profit_factor()   # total gains / total losses — >1 means net profitable

# Streak analysis
returns.fc.consecutive_wins()    # longest consecutive winning period
returns.fc.consecutive_losses()  # longest consecutive losing period

# Market exposure
returns.fc.exposure()        # fraction of time with non-zero returns (in market)
```

**Interpreting payoff ratio vs win rate together:**

| Win rate | Payoff ratio | Implication |
|---|---|---|
| High (>0.6) | Any | Good trend-following or momentum |
| Low (<0.4) | High (>2.5) | Typical mean-reversion / options selling profile |
| ~0.5 | ~1.0 | Break-even before costs — review strategy |

## Advanced Sharpe variants

Beyond the standard Sharpe, Fundcloud includes variants that correct for known
weaknesses in the basic formulation:

```python
# Corrects for positive autocorrelation in returns (Sharpe inflation from smooth curves)
returns.fc.smart_sharpe()
returns.fc.smart_sortino()

# Probabilistic Sharpe Ratio: tests whether Sharpe exceeds a benchmark
# Returns a probability (0–1); >0.95 is a strong signal
returns.fc.probabilistic_sharpe(target_sharpe=0.5)

# Robust to small samples (fewer than ~252 observations)
returns.fc.adjusted_sortino()

# Kelly criterion: optimal fraction of capital to allocate
# Use conservatively — multiply by 0.25–0.5 in practice
returns.fc.kelly_criterion()

# Probability of ruin: chance of losing ruin_level fraction of capital
returns.fc.risk_of_ruin(ruin_level=0.5)  # prob of losing 50%+
```

When to use each:

- **smart_sharpe/sortino**: strategies with smooth NAVs (monthly rebalancing, trend-following) — standard Sharpe overstates performance when autocorrelation is high.
- **PSR**: comparing a backtest Sharpe against a benchmark — guards against lucky results from multiple testing.
- **Kelly**: position sizing. Full Kelly is almost always too aggressive; use half- or quarter-Kelly in production.

## Tail risk and pain measures

These metrics quantify the shape and severity of the loss distribution — useful
when CVaR alone isn't enough to differentiate two strategies with similar Sharpe.

```python
# Tail ratio: |P95 of gains| / |P5 of losses| — >1 means right tail bigger than left
returns.fc.tail_ratio()

# Gain-to-pain: total net gain / sum of all individual period losses
# Higher is better; insensitive to extreme one-off events unlike Sharpe
returns.fc.gain_to_pain_ratio()

# Pain index: mean of squared drawdowns over the full history
returns.fc.pain_index()

# Pain ratio: excess return / pain index (like Sharpe but for drawdown severity)
returns.fc.pain_ratio()

# Common sense ratio: blends tail ratio and gain-to-pain into a single figure
returns.fc.common_sense_ratio()

# Ulcer performance index: CAGR / ulcer index — penalises deep AND prolonged drawdowns
returns.fc.ulcer_performance_index()
```

**Pain index vs Ulcer index:**

| Metric | Definition | Sensitivity |
|---|---|---|
| Ulcer index | RMS of running drawdown | Duration-weighted — long recoveries hurt more |
| Pain index | Mean of squared drawdowns | Severity-weighted — deep single drawdowns hurt more |
| Pain ratio | Excess return / pain index | Like Calmar but uses all drawdown episodes, not just the max |

## Rolling metrics

Rolling metrics reveal how a strategy's characteristics evolve over time — regimes,
crowding, and factor exposure all show up here before they appear in terminal metrics.

```python
window = 63  # ~1 quarter of trading days

# Rolling factor exposure vs a benchmark
rolling_beta  = returns.fc.rolling_beta(benchmark, window=window)   # pd.Series

# Rolling risk
rolling_vol = returns.fc.rolling_volatility(window=21)  # ~1 month
rolling_dd  = returns.fc.rolling_drawdown(window=window)

# Rolling risk-adjusted return
rolling_sharpe = returns.fc.rolling_sharpe(window=window)
rolling_sortino = returns.fc.rolling_sortino(window=window)
```

These all return `pd.Series` indexed identically to the input. Use them for:

- **Regime detection**: beta > 1 in 2020-style crashes signals factor crowding
- **Parameter stability testing**: a strategy whose rolling Sharpe collapses in recent quarters may be overfitted
- **Live monitoring**: check that rolling vol hasn't spiked above historical bounds

```python
# Example: flag when rolling Sharpe drops more than 1 from its 6-month average
rs = returns.fc.rolling_sharpe(window=63)
alert = rs < (rs.rolling(126).mean() - 1.0)
print(f"Alert triggered on {alert.sum()} days")
```
