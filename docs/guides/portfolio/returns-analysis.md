---
title: Returns analysis
description: Instant performance analytics on any pandas Series or DataFrame — no simulator required.
---

# Returns analysis

If you have a returns `Series` — from a broker export, a notebook experiment,
or the output of the Simulator — one `import fundcloud` is the only setup
required to run a full metric suite.

---

## One call to start

```python
import numpy as np
import pandas as pd
import fundcloud  # registers .fc on pandas Series and DataFrame

rng = np.random.default_rng(42)
idx = pd.bdate_range("2022-01-03", periods=504)           # two years of business days
returns = pd.Series(
    rng.normal(0.0007, 0.011, len(idx)), index=idx, name="my_strategy"
)

returns.fc.metrics()
```

Typical output:

```
cagr               0.1843
ann_volatility     0.1748
sharpe             1.0543
sortino            1.4912
calmar             0.7201
omega              1.2887
max_drawdown      -0.2560
ulcer_index        0.0621
cvar               0.0151
win_rate           0.5476
avg_return         0.0007
Name: my_strategy, dtype: float64
```

### What each metric tells you

**CAGR** — Compound Annual Growth Rate. The annualised geometric return.
Comparable across strategies with different track-record lengths, unlike
`total_return`. A CAGR of 0.18 means +18 % per year, compounded.

**ann_volatility** — Annualised standard deviation of daily returns. The
denominator of Sharpe. High vol is not automatically bad — it depends whether
the return justifies it.

**Sharpe** — `(CAGR − risk_free) / ann_volatility`. The single most widely
quoted risk-adjusted return measure. A Sharpe above 1.0 is considered strong
for a live strategy; many retail strategies sit between 0.3 and 0.8.

**Sortino** — Like Sharpe, but the denominator is *downside* deviation only
(returns below the target). Penalises strategies that lose more than they
should, while not penalising upside volatility. Usually higher than Sharpe for
trend-following.

**Calmar** — `CAGR / |max_drawdown|`. Annualised return per unit of peak loss.
A Calmar of 0.5 means the strategy earns half its worst drawdown back per year.
CTAs and managed-futures funds typically target Calmars between 0.5 and 1.5.

**Omega** — Ratio of gains above a threshold to losses below it. A probability-
weighted measure of the whole return distribution. Above 1 means the strategy
earns more than it loses relative to the threshold.

**max_drawdown** — Peak-to-trough decline in the wealth curve. One of the most
psychologically important metrics — it determines whether you can stay invested.

**ulcer_index** — Root-mean-square of the running drawdown path. Captures both
depth *and* duration of underwater periods. A portfolio with a quick -15 %
recovery has a lower ulcer index than one with a slow -8 % grind.

**CVaR** — Conditional Value at Risk at 95 %. The expected loss on the worst
5 % of days. More informative than VaR for tail-risk budgeting. A CVaR of
0.015 means losses average 1.5 % on the worst days.

**win_rate** — Fraction of periods with positive returns. A high win rate
doesn't guarantee profitability if losses are large; pair it with
`payoff_ratio`.

**avg_return** — Average per-period return. Multiply by 252 for a rough
annualised figure (not the same as CAGR due to compounding, but useful for
quick comparisons).

### Same call on a DataFrame

When `returns` is a multi-column DataFrame, `.fc.summary()` returns a
**DataFrame** with metrics as rows and strategies as columns.

```python
rng = np.random.default_rng(11)
idx = pd.bdate_range("2022-01-03", periods=504)
df = pd.DataFrame(
    {
        "conservative": rng.normal(0.0003, 0.006, len(idx)),
        "balanced":     rng.normal(0.0007, 0.011, len(idx)),
        "aggressive":   rng.normal(0.0012, 0.019, len(idx)),
    },
    index=idx,
)

df.fc.summary()
# Returns a DataFrame: rows = metrics, columns = strategy names
```

Pick a single row to compare all three at once:

```python
df.fc.summary().loc["sharpe"]
# conservative    0.81
# balanced        1.05
# aggressive      0.94
# Name: sharpe, dtype: float64
```

---

## The risk story: drawdowns

`max_drawdown` is a start, not an answer. A -20 % drawdown that lasts three
weeks is painful but manageable. The same loss spread over three years erodes
confidence, triggers investor redemptions, and — for levered strategies — can
force liquidation before recovery.

Duration is the missing dimension.

```python
from fundcloud.portfolio import Portfolio

pf = Portfolio(returns=returns, name="my_strategy")

# Worst N drawdown episodes, display-formatted
pf.worst_drawdowns(top=5)
#    Started     Recovered   Drawdown    Days
#  2022-06-14   2023-01-09    -0.256     209
#  2022-02-28   2022-04-01    -0.118      32
#  2023-10-02   2023-11-17    -0.097      46
#  2023-07-18   2023-08-21    -0.071      34
#  2022-11-04   2022-12-01    -0.063      27

# Full episode table: start / valley / recovery / depth / duration / recovery time
pf.drawdown_details()
# Returns a DataFrame with columns:
#   start, valley, recovery, max_drawdown, duration_days, days_to_recover
```

The `recovery` column is `NaT` for any episode still underwater at the end of
the sample. `days_to_recover` is `NaN` for the same reason.

```python
# Running drawdown series for plotting or alerting
dd = returns.fc.drawdown_series()       # pd.Series, always <= 0
deepest = dd.idxmin()
print(f"Deepest point: {dd.min():.1%} on {deepest.date()}")

# Plotly drawdown chart — one line per column for a DataFrame
returns.fc.plot_drawdown().show()
```

!!! tip "Duration is the psychological test"
    A -20 % drawdown lasting 3 months is a bump. The same loss lasting
    3 years destroys the investor relationship. When screening strategies,
    check `days_to_recover` in `drawdown_details()`, not just `max_drawdown`.

!!! warning "Still underwater at end of sample"
    If the most recent episode has `recovery = NaT`, the strategy has not
    recovered yet. The reported `max_drawdown` and `duration_days` are current
    as of the last data point, not necessarily the final state of the episode.

---

## Period returns: the institutional view

MTD and YTD are what every performance report leads with. The 3Y and 5Y
annualised figures are what institutional allocators actually care about when
deciding whether to invest.

```python
rng = np.random.default_rng(0)
idx = pd.date_range("2015-01-02", periods=2500, freq="B")
returns = pd.Series(rng.normal(0.0005, 0.012, 2500), index=idx, name="Strategy")
spy_rets = pd.Series(rng.normal(0.0003, 0.010, 2500), index=idx, name="SPY")

pf = Portfolio(returns=returns, name="Strategy")

# MTD / 3M / 6M / YTD / 1Y / 3Y / 5Y / 10Y / All-time (vs benchmark)
pf.period_returns(benchmark=spy_rets)
#                       SPY    Strategy
# MTD               -0.0186      0.0026
# 3M                 0.1228     -0.0433
# 6M                -0.0524     -0.0726
# YTD               -0.0967     -0.0584
# 1Y                 0.2325     -0.0310
# 3Y (ann.)          0.1448     -0.0530
# 5Y (ann.)          0.2126      0.0292
# 10Y (ann.)         0.1199      0.0207
# All-time (ann.)    0.1199      0.0207

# Calendar-year returns, one row per year
pf.yearly_returns(benchmark=spy_rets)
#           SPY    Strategy
# 2015    0.0121      0.0064
# 2016    0.1195      0.1438
# ...
```

!!! note "3Y / 5Y are annualised"
    Multi-year rows use `(1 + total_return)^(1/years) - 1` internally so they
    are directly comparable to the 1Y and MTD rows. Do not mix these with
    cumulative figures without rescaling.

---

## Multi-asset comparison

=== "Summary table"

    ```python
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-03", periods=504)
    df = pd.DataFrame(
        {
            "conservative": rng.normal(0.0003, 0.006, len(idx)),
            "balanced":     rng.normal(0.0007, 0.011, len(idx)),
            "aggressive":   rng.normal(0.0012, 0.019, len(idx)),
        },
        index=idx,
    )

    # 11-metric table — metrics as rows, strategies as columns
    summary = df.fc.summary()

    # Slice specific rows
    summary.loc[["cagr", "sharpe", "max_drawdown", "cvar"]]
    #               conservative  balanced  aggressive
    # cagr              0.0748    0.1843      0.3267
    # sharpe            0.8106    1.0543      1.0219
    # max_drawdown     -0.0812   -0.2560     -0.4487
    # cvar              0.0087    0.0154      0.0267
    ```

=== "Full 55-metric bundle"

    ```python
    # Full metric-by-strategy table (optional benchmark for alpha/beta/etc.)
    rng_spy = np.random.default_rng(99)
    spy = pd.Series(rng_spy.normal(0.0004, 0.009, len(idx)), index=idx, name="SPY")

    full = df.fc.metrics(benchmark=spy)
    # Shape: ~55 rows × 3 columns
    # With benchmark= included, also has: alpha, beta, correlation,
    # r_squared, information_ratio, tracking_error, up/down capture, treynor_ratio

    full.loc[["alpha", "beta", "up_capture", "down_capture"]]
    #               conservative  balanced  aggressive
    # alpha             0.0631    0.1502      0.2918
    # beta              0.0443    0.0872      0.1461
    # up_capture        1.0062    1.1032      1.1785
    # down_capture      1.0143    1.0718      1.1512
    ```

=== "Plots"

    ```python
    # Cumulative wealth curves — one line per column
    df.fc.plot_cumulative(title="Strategy comparison").show()

    # Drawdown corridors — one trace per column
    df.fc.plot_drawdown().show()

    # Rolling 63-day Sharpe
    df.fc.plot_rolling_sharpe(window=63).show()
    ```

### Rolling metrics on a single strategy

Rolling metrics show how characteristics evolve over time — invaluable for
regime detection and overfitting checks.

```python
window = 63  # ~1 quarter

rs = returns.fc.rolling_sharpe(window=window)          # pd.Series
rv = returns.fc.rolling_volatility(window=21)          # ~1 month vol
rd = returns.fc.rolling_drawdown()                     # current drawdown series
rb = returns.fc.rolling_beta(spy_rets, window=window)  # pd.Series

# Flag when rolling Sharpe drops more than 1 standard deviation below its 6-month average
alert = rs < (rs.rolling(126).mean() - rs.rolling(126).std())
print(f"Rolling Sharpe alert on {alert.sum()} days")
```

---

## Tear sheet in one line

```python
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet

rng = np.random.default_rng(42)
idx = pd.bdate_range("2022-01-03", periods=504)
returns = pd.Series(rng.normal(0.0007, 0.011, len(idx)), index=idx, name="my_strategy")

pf = Portfolio(returns=returns, name="my_strategy")
Tearsheet(pf, title="My Strategy").render_html("tearsheet.html")

# PDF and Excel use identical syntax
Tearsheet(pf, title="My Strategy").render_pdf("report.pdf")
Tearsheet(pf, title="My Strategy").render_excel("report.xlsx")
```

The tear sheet includes:

- Cumulative returns vs benchmark
- Drawdown corridor
- Rolling Sharpe (63-day)
- Monthly heatmap
- Yearly returns bar chart
- Period performance table (MTD → All-time)
- Worst drawdowns episode table
- Full metric summary

!!! tip "Jupyter shortcut — no file needed"
    For inline notebook exploration, skip the `Portfolio` / `Tearsheet`
    constructors entirely:

    ```python
    import fundcloud

    # Composite summary figure (cumulative, drawdown, rolling Sharpe, …)
    returns.fc.plot_summary(title="My Strategy").show()

    # Or call the module-level function directly
    import fundcloud.plots as fc_plots
    fc_plots.summary(returns, title="My Strategy").show()
    ```

### Shortcut: tear sheet from the accessor

```python
# Returns Series → HTML (no Portfolio object required)
returns.fc.render_html("tearsheet.html", title="My Strategy")
returns.fc.render_html("vs_spy.html", benchmark=spy_rets, title="vs SPY")

# DataFrame → one tab per column
df.fc.render_html("multi.html", title="All strategies")
```

---

## Metric cheat sheet

### Risk-adjusted returns

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `sharpe` | `(CAGR − rf) / ann_vol` — excess return per unit of total volatility | Higher is better |
| `sortino` | `(CAGR − rf) / downside_vol` — penalises downside only | Higher is better |
| `calmar` | `CAGR / |max_drawdown|` — return per unit of peak loss | Higher is better |
| `omega` | Probability-weighted ratio of gains to losses relative to threshold | Higher is better |
| `smart_sharpe` | Sharpe adjusted for positive autocorrelation in returns; corrects for smooth NAV inflation | Higher is better |
| `smart_sortino` | Sortino adjusted for autocorrelation | Higher is better |
| `probabilistic_sharpe` | Probability (0–1) that the true Sharpe exceeds a benchmark Sharpe | Higher is better |
| `adjusted_sortino` | Sortino corrected for small-sample bias | Higher is better |

### Return metrics

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `total_return` | Cumulative growth: `(1 + r₁)(1 + r₂)… − 1` | Higher is better |
| `cagr` | Annualised compound return | Higher is better |
| `avg_return` | Arithmetic average of per-period returns | Higher is better |
| `best` | Single best period return | Context-dependent |
| `worst` | Single worst period return | Context-dependent |
| `volatility` | Annualised standard deviation of returns (total vol) | Lower is better |
| `downside_volatility` | Annualised standard deviation of returns below target | Lower is better |

### Drawdown and underwater risk

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `max_drawdown` | Largest peak-to-trough decline in the wealth curve | Lower magnitude is better |
| `drawdown_series` | Per-bar running drawdown (wealth / cummax − 1), always ≤ 0 | n/a (series, not scalar) |
| `ulcer_index` | Root-mean-square of the running drawdown path; penalises long underwater periods | Lower is better |

### Tail risk and pain

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `cvar` | Expected loss on the worst α % of periods (Conditional VaR / Expected Shortfall) | Lower magnitude is better |
| `value_at_risk` | Loss threshold exceeded on the worst α % of periods | Lower magnitude is better |
| `tail_ratio` | `|P95 of gains| / |P5 of losses|` — right tail vs left tail | Higher is better |
| `common_sense_ratio` | `tail_ratio × gain_to_pain_ratio` combined into one figure | Higher is better |
| `gain_to_pain_ratio` | Total net gain / sum of all individual period losses | Higher is better |
| `pain_index` | Mean of squared running drawdowns; severity-weighted | Lower is better |
| `pain_ratio` | `(CAGR − rf) / pain_index` — like Sharpe but with drawdown severity as denominator | Higher is better |
| `ulcer_performance_index` | `CAGR / ulcer_index` — penalises both depth and duration of drawdowns | Higher is better |

### Trade statistics

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `win_rate` | Fraction of periods with positive returns | Higher is better (paired with payoff_ratio) |
| `avg_win` | Average return on positive periods | Higher is better |
| `avg_loss` | Average return on negative periods (negative value) | Lower magnitude is better |
| `payoff_ratio` | `avg_win / |avg_loss|` — average win vs average loss size | Higher is better |
| `profit_factor` | Total gross gains / total gross losses | Higher is better (>1 = net profitable) |
| `consecutive_wins` | Longest winning streak (count of periods) | Higher is better |
| `consecutive_losses` | Longest losing streak (count of periods) | Lower is better |
| `exposure` | Fraction of periods with non-zero returns (time in market) | Context-dependent |

### Distribution

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `skew` | Return distribution skewness; positive = right tail heavier | Positive preferred (right-skewed gains) |
| `kurtosis` | Excess kurtosis; measures tail heaviness relative to normal | Lower is better (fat tails = surprise risk) |

### Position sizing

| Metric | What it measures | Direction |
|--------|-----------------|-----------|
| `kelly_criterion` | Optimal fraction of capital: `win_rate − (1 − win_rate) / payoff_ratio` | n/a — use as input, not absolute signal |
| `risk_of_ruin` | Probability of losing a specified fraction of capital given current win/loss stats | Lower is better |

!!! note "Kelly in practice"
    Full Kelly is almost never used directly — it assumes stationary returns and
    maximises long-run log-wealth, which implies very high variance. Multiply the
    Kelly fraction by 0.25–0.5 ("quarter-Kelly" or "half-Kelly") for a smoother
    equity curve at the cost of lower expected long-run growth.

!!! note "Paired metrics matter"
    `win_rate` and `payoff_ratio` should always be read together. A 40 % win
    rate with a payoff ratio of 3.0 is a profitable distribution. A 70 % win
    rate with a payoff ratio of 0.4 is not.
