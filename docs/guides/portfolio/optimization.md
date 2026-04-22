---
title: Portfolio optimisation
description: From HRP quick win to MeanRisk with constraints and walk-forward validation.
---

# Portfolio optimisation

This guide covers the full optimizer stack — from a 15-line HRP backtest to
constrained Mean-Risk with transaction costs, and through to walk-forward
cross-validation with `GridSearchCV`.

---

## Part 1: Quick win — HRP in 15 lines

Hierarchical Risk Parity clusters assets by correlation using hierarchical
agglomerative clustering, then allocates inversely proportional to cluster
variance. It avoids matrix inversion entirely, which makes it more stable
than MVO when the number of assets is large or the estimation window is
short. Turnover is low because cluster structure is relatively persistent
across rebalances.

!!! note "Installation"
    HRP and all skfolio-backed optimizers require the optional extra:

    ```
    uv add 'fundcloud[pf]'
    ```

    Core optimizers (`EqualWeighted`, `InverseVolatility`, `MVO`) are always
    available without any extras.

```python
import numpy as np
import pandas as pd
import fundcloud  # noqa: F401
from fundcloud.optimize import HierarchicalRiskParity
from fundcloud.sim import Simulator
from fundcloud.reports import Tearsheet

# --- synthetic 4-asset panel (replace with real bars from YF etc.) ---
rng = np.random.default_rng(42)
idx = pd.bdate_range("2020-01-02", periods=1260)  # 5 years

def _asset(p0, vol):
    c = p0 + np.cumsum(rng.normal(0, vol, len(idx)))
    return {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1e6}

bars = pd.concat(
    {
        "US_EQ": pd.DataFrame(_asset(400, 2.0), index=idx),
        "EU_EQ": pd.DataFrame(_asset(280, 2.4), index=idx),
        "BONDS": pd.DataFrame(_asset(100, 0.4), index=idx),
        "GOLD":  pd.DataFrame(_asset(180, 1.2), index=idx),
    },
    axis=1,
).pipe(lambda df: df.set_axis(df.columns.swaplevel(), axis=1)).sort_index(axis=1)

# --- optimise ---
returns = bars.xs("close", level=0, axis=1).pct_change().dropna()

hrp = HierarchicalRiskParity()
hrp.fit(returns)
opt_pf = hrp.predict(returns)
print("HRP weights:\n", opt_pf.weights.round(3))  # pd.DataFrame, shape (1, n_assets)

# --- simulate with optimised weights ---
result = Simulator(bars, cash=100_000).run_weights(opt_pf.weights)
print(result.portfolio.summary())

# --- tear sheet ---
Tearsheet(result.portfolio, title="HRP 5Y backtest").render_html("hrp.html")
```

---

## Part 2: Choosing the right optimizer

Use this table as a quick triage. Start with the simplest optimizer that
satisfies your constraint set and only reach for `MeanRisk` when you need
specific objective functions or hard constraints.

| Optimizer | When to use | Key strength | Requires |
|---|---|---|---|
| `EqualWeighted` | Baseline, null hypothesis | Zero estimation risk | core |
| `InverseVolatility` | Simple risk parity without correlation | Intuitive, robust | core |
| `MVO` | Max Sharpe when you trust μ/Σ | Theoretically optimal under normality | core |
| `HierarchicalRiskParity` | Stable allocation, many assets | Low turnover, no matrix inversion | `[pf]` |
| `HierarchicalEqualRiskContribution` | HRP + equal risk per cluster | More balanced than plain HRP | `[pf]` |
| `MeanRisk` | Full control: any objective, any constraint | Maximum flexibility | `[pf]` |
| `RiskBudgeting` | Known target risk contributions | Client-mandate driven | `[pf]` |
| `MaximumDiversification` | Maximise diversification ratio | Alternative to max Sharpe | `[pf]` |
| `NestedClustersOptimization` | Two-level cluster hierarchy | Combines HRP structure with MVO precision | `[pf]` |

### Core optimizers (no extras)

All three follow the same `fit` / `predict` API:

```python
from fundcloud.optimize import EqualWeighted, InverseVolatility, MVO

EqualWeighted().fit(returns).predict(returns).weights
InverseVolatility().fit(returns).predict(returns).weights
MVO().fit(returns).predict(returns).weights  # max Sharpe
```

!!! tip "Use EqualWeighted as your baseline"
    Before doing anything clever, compare against `EqualWeighted`. It beats
    most optimized strategies over long horizons because it has zero
    estimation risk. If your optimizer can't beat it, the estimation error
    is eating the theoretical gains.

---

## Part 3: MeanRisk — full configuration

`MeanRisk` is the most flexible optimizer. It wraps skfolio's `MeanRisk`
directly — all parameters are passed through unchanged. The sections below
cover the most commonly needed levers.

### 3.1 Objective and risk measures

```python
from fundcloud.optimize import MeanRisk, RiskMeasure

# Minimise variance (classical Markowitz)
MeanRisk(risk_measure=RiskMeasure.VARIANCE)

# Minimise tail risk — recommended for equities with non-normal returns
MeanRisk(risk_measure=RiskMeasure.CVAR)

# Minimise maximum drawdown — drawdown-aware but computationally intensive
MeanRisk(risk_measure=RiskMeasure.MAX_DRAWDOWN)

# Maximise Sharpe ratio instead of minimising risk
from skfolio.optimization import ObjectiveFunction
MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
)
```

**When to use each risk measure:**

- **VARIANCE** — the standard Markowitz objective. Works well when returns
  are approximately normal and you have a long estimation window (500+
  observations). Sensitive to outliers and mean estimation error.

- **CVAR** (Conditional Value at Risk, expected shortfall at the 5% tail) —
  the recommended default for equity portfolios. Robust to non-normal
  distributions, explicitly penalises tail losses, and has desirable
  sub-additivity properties for risk budgeting.

- **MAX_DRAWDOWN** — directly optimises the worst peak-to-trough loss.
  Use when drawdown is the primary constraint for the mandate (e.g. absolute
  return funds with -15% stop-loss). Computationally expensive — expect
  10-50x slower solve times than VARIANCE.

### 3.2 Weight constraints

```python
# Basic bounds: every asset between 2% and 25%
MeanRisk(min_weights=0.02, max_weights=0.25)

# Standard long-only (default)
MeanRisk(min_weights=0.0, max_weights=1.0)

# Allow up to 10% short per asset
MeanRisk(min_weights=-0.10, max_weights=0.40)

# Limit to maximum 8 assets (cardinality — forces sparsity)
MeanRisk(cardinality=8, min_weights=0.02)

# Regularisation: penalise concentrated weights
MeanRisk(l2_coef=0.01)   # ridge — shrinks weights toward equal-weight
MeanRisk(l1_coef=0.005)  # lasso — induces sparsity (some weights go to zero)
```

!!! tip "Regularisation vs cardinality"
    `l2_coef` is a soft constraint that costs nothing in terms of solver
    complexity — prefer it as a first line of defence against over-concentration.
    `cardinality` introduces a mixed-integer program (MILP) which can be
    substantially slower; only use it when you need a hard asset-count limit.

### 3.3 Leverage and budget control

```python
# Fully invested (default)
MeanRisk(budget=1.0)

# Cash buffer: only 80% in risky assets
MeanRisk(budget=0.80)

# Leveraged portfolio (net long 130%)
MeanRisk(budget=1.30, max_long=1.30)
```

### 3.4 Transaction-cost-aware rebalancing

Passing `previous_weights` and `transaction_costs` turns the optimizer
into a turnover-penalised solver. It will trade off small expected-return
gains against the cost of getting there, producing a more stable
allocation.

```python
current_weights = pd.Series(
    {"US_EQ": 0.35, "EU_EQ": 0.25, "BONDS": 0.25, "GOLD": 0.15}
)

MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    transaction_costs=0.0010,          # 10 bps one-way
    previous_weights=current_weights,  # penalise deviations from current
)

# Hard turnover cap: never trade more than 20% of portfolio in one rebalance
MeanRisk(max_turnover=0.20)
```

### 3.5 Return constraints

```python
# Ensure portfolio return >= 8% annualised
MeanRisk(min_return=0.08 / 252)  # daily equivalent
```

!!! warning "Return constraints tighten the feasible set"
    `min_return` can make the optimisation infeasible if the constraint is
    too tight relative to the estimation window. Always wrap the solver call
    in a try/except and fall back to an unconstrained solve when infeasibility
    is detected.

---

## Part 4: From optimizer to backtest

### Pattern A — Static allocation (one-shot)

Fit on a training window, apply the weights to the full history. Simple to
reason about; appropriate when you believe the regime is stable.

```python
from fundcloud.optimize import MeanRisk, RiskMeasure
from fundcloud.sim import Simulator

train_end = "2022-12-31"
train_returns = returns.loc[:train_end]

opt = MeanRisk(risk_measure=RiskMeasure.CVAR, min_weights=0.02, max_weights=0.30)
opt.fit(train_returns)
static_weights = opt.predict(train_returns).weights

result = Simulator(bars, cash=100_000).run_weights(static_weights)
print(f"OOS Sharpe: {result.portfolio.sharpe():.2f}")
```

### Pattern B — Walk-forward (rolling re-fit)

Re-fit the optimizer periodically as new data arrives. Captures regime
shifts; the cost is higher turnover and estimation noise at each refit.

```python
from fundcloud.strategies import BaseStrategy, Context
from fundcloud.sim import Order

class RollingMeanRiskStrategy(BaseStrategy):
    def __init__(self, lookback: int = 252) -> None:
        self.lookback = lookback
        self._opt = MeanRisk(risk_measure=RiskMeasure.CVAR)
        self._weights: dict[str, float] = {}

    def decide(self, ctx: Context) -> list[Order]:
        if len(ctx.history) < self.lookback:
            return []
        closes = ctx.history.xs("close", level=0, axis=1)
        rets = closes.pct_change().dropna().iloc[-self.lookback:]
        self._opt.fit(rets)
        new_weights = self._opt.predict(rets).weights.to_dict()
        orders = []
        for asset, w in new_weights.items():
            orders.append(
                Order(ts=ctx.ts, asset=asset, side="buy",
                      notional=ctx.portfolio.equity * w)
            )
        return orders
```

!!! warning "Refit frequency vs overfitting"
    The example above re-fits at every bar, which is unrealistically
    expensive and can introduce look-ahead bias if not handled carefully.
    In practice, refit monthly or quarterly — use `ctx.ts.month` or a
    day-of-month check inside `decide` to gate the refit.

---

## Part 5: Walk-forward validation (cross-validated OOS)

Walk-forward validation answers the question "how would this optimizer
have performed on data it never saw during fitting?" It is the minimum
credible bar for reporting a backtest Sharpe.

### 5.1 PurgedKFold vs EmbargoedKFold

Standard K-fold leaks because adjacent train and test observations share
autocorrelated information. Purged and embargoed variants cut that link.

```
Timeline →
┌──────────────┬──────┬──────────┬─────────┬──────────────┐
│  Train fold  │ Purge│  Test    │ Embargo │  Train fold  │
│  (fit here)  │ gap  │ (score)  │  gap    │  (next fold) │
└──────────────┴──────┴──────────┴─────────┴──────────────┘
```

- **Purge gap** — observations at the train/test boundary are dropped from
  training to prevent label leakage from autocorrelated features.
- **Embargo gap** — observations at the test/next-train boundary are dropped
  to prevent the next training fold from "seeing" the future.

```python
from fundcloud.validate import PurgedKFold, EmbargoedKFold

# PurgedKFold: removes purge days from train/test boundary
# Use when returns are autocorrelated over short windows (momentum, trend)
cv = PurgedKFold(n_splits=5, purge=21)  # 21 trading days ~ 1 month gap

# EmbargoedKFold: also silences the start of the NEXT train fold
# Use for strategies where leakage can propagate forward (lookahead labels)
cv = EmbargoedKFold(n_splits=5, purge=21, embargo=10)
```

!!! tip "Which one to use?"
    Start with `PurgedKFold`. Add `EmbargoedKFold` when your features
    include any forward-looking computation (e.g. labels derived from
    future returns, smoothed signals with long decay). For a pure
    price-returns pipeline, `PurgedKFold` with `purge=21` is the standard
    choice.

### 5.2 Manual walk-forward

The fold loop gives you full control over what happens in each split —
useful when you need per-fold diagnostics or want to swap optimizers
between folds.

```python
import numpy as np
import pandas as pd
from fundcloud.optimize import MeanRisk, RiskMeasure
from fundcloud.portfolio import Portfolio
from fundcloud.validate import PurgedKFold

cv = PurgedKFold(n_splits=5, purge=21)
oos_sharpes = []

for fold, (train_idx, test_idx) in enumerate(cv.split(returns), 1):
    train = returns.iloc[train_idx]
    test  = returns.iloc[test_idx]
    opt = MeanRisk(risk_measure=RiskMeasure.CVAR, min_weights=0.02)
    opt.fit(train)
    w = opt.predict(train).weights.to_numpy().squeeze()  # shape (n_assets,)
    oos_rets = test.to_numpy() @ w
    oos_pf = Portfolio(returns=pd.Series(oos_rets, index=test.index))
    oos_sharpes.append(oos_pf.sharpe())
    print(f"Fold {fold}: OOS Sharpe = {oos_pf.sharpe():.2f}")

print(f"\nMean OOS Sharpe: {np.mean(oos_sharpes):.2f} ± {np.std(oos_sharpes):.2f}")
```

### 5.3 GridSearchCV for constraint search

Because every Fundcloud optimizer is a drop-in sklearn estimator, you can
plug it directly into `GridSearchCV` with a purged splitter as the `cv`
argument.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from fundcloud.optimize import MeanRisk, RiskMeasure
from fundcloud.validate import PurgedKFold

pipe = Pipeline([("optim", MeanRisk(risk_measure=RiskMeasure.CVAR))])

param_grid = {
    "optim__min_weights": [0.0, 0.02, 0.05],
    "optim__max_weights": [0.25, 0.40],
    "optim__l2_coef":     [0.0, 0.01],
}

search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=PurgedKFold(n_splits=5, purge=21),
    n_jobs=-1,
)
search.fit(returns)

print("Best params:", search.best_params_)
best_weights = search.best_estimator_.predict(returns).weights
```

!!! tip "Parallelise across folds"
    `n_jobs=-1` distributes folds across all CPU cores via joblib. On a
    8-core machine this reduces a 12-parameter grid from ~5 minutes to
    ~45 seconds with 5 folds.

!!! warning "Multiple testing inflates Sharpe"
    Searching over many parameter combinations is a form of in-sample
    optimisation. Each combination tried is another draw from the
    multiple-testing distribution. Validate the best set of parameters
    on a held-out period that was never part of the grid search — or
    compute the Probabilistic Sharpe Ratio (`pf.probabilistic_sharpe()`)
    to assess whether the best result is statistically distinguishable
    from luck.

### 5.4 Advanced optimizers

=== "RiskBudgeting"

    ```python
    from fundcloud.optimize import RiskBudgeting

    # Equal risk contribution per asset (risk parity)
    rb = RiskBudgeting()

    # Target a specific risk budget (e.g. 40% equities, 60% bonds by risk)
    rb_custom = RiskBudgeting(
        risk_budget={"US_EQ": 0.40, "EU_EQ": 0.20, "BONDS": 0.30, "GOLD": 0.10}
    )
    rb_custom.fit(returns)
    print(rb_custom.predict(returns).weights.round(3))
    ```

    Use when a client mandate specifies target risk contributions rather than
    target weights. The portfolio weights will shift to equalise (or match)
    each asset's marginal contribution to total portfolio risk.

=== "MaximumDiversification"

    ```python
    from fundcloud.optimize import MaximumDiversification

    # Maximises the diversification ratio (Choueifaty & Coignard, 2008):
    # ratio of weighted-average asset volatility to portfolio volatility
    md = MaximumDiversification()
    md.fit(returns)
    print(md.predict(returns).weights.round(3))
    ```

    An alternative to max Sharpe that does not require a return estimate.
    Tends to produce more stable weights than MVO under estimation error
    because it only depends on the covariance matrix, not the mean vector.

=== "NestedClustersOptimization"

    ```python
    from fundcloud.optimize import NestedClustersOptimization

    # Two-level hierarchy: cluster assets first, then optimise within
    # each cluster and across clusters separately
    nco = NestedClustersOptimization()
    nco.fit(returns)
    print(nco.predict(returns).weights.round(3))
    ```

    Combines the cluster stability of HRP with the optimality of MVO within
    each cluster. Especially useful when you have a natural grouping (sectors,
    geographies) that you want the optimizer to respect.
