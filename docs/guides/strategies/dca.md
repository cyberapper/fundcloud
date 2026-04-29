# DCA and Hold strategies

A `BaseStrategy` is Fundcloud's abstraction for anything that decides, on a bar-by-bar basis, what positions you should be holding. It produces either weights, signals, or explicit orders — the simulator will accept any of the three. Two presets ship in core:

- `Hold` — set a target allocation once, optionally rebalance when drift crosses a tolerance band.
- `DCA` — buy a fixed cash amount (or a fraction of current equity) across a target mix on a daily, weekly, or monthly cadence, optionally within a window and optionally selling out on exit.

Both are useful baselines on their own (DCA is a surprisingly hard strategy to beat on a risk-adjusted basis over long horizons) and serve as the reference implementations you subclass when you need something bespoke. The last section shows exactly that.

## Backtest configuration cheat sheet

Both strategies are typically run via the `pd.DataFrame.fc.run_hold` /
`run_dca` accessors, which forward keyword arguments to `Simulator`.
Every backtest sees the same set of friction / execution knobs — these
are the defaults you're agreeing to when you call `run_hold()` /
`run_dca()` without overriding them:

| Keyword     | Default                                | What it controls                                          |
|-------------|----------------------------------------|-----------------------------------------------------------|
| `cash`      | `1_000_000.0`                          | Starting cash balance — the basis for the entire backtest. |
| `costs`     | `FixedBps(5)`                          | 5 bps fee on every fill, applied symmetrically buy/sell.   |
| `slippage`  | `NoSlippage()`                         | Fills hit the reference price exactly. Swap in `HalfSpread(bps)` for a more realistic execution.  |
| `execution` | `NextBarOpen()`                        | Orders queued on bar `t` fill at the open of bar `t+1`. Swap in `SameBarClose()` to fill at the close of the bar that emitted the order. |

```python
from fundcloud.sim import FixedBps, HalfSpread, SameBarClose

bars.fc.run_hold(                       # override every default
    {"SPY": 0.6, "AGG": 0.4},
    cash=100_000,
    costs=FixedBps(10),                 # 10 bps instead of 5
    slippage=HalfSpread(2.0),           # 1 bp half-spread
    execution=SameBarClose(),           # fill on emit-bar close
)
```

See [`reference/strategies.md`](../../reference/strategies.md) for the
full per-class API.

## `Hold` — buy once, optionally rebalance

```python
from fundcloud.strategies import Hold, RebalanceSpec

# Equal-weight default: spread evenly across every asset in the bars frame.
Hold()

# One-shot buy & hold with explicit weights.
Hold(weights={"AAPL": 0.6, "MSFT": 0.4})

# Same thing but rebalance monthly when drift > 5 %.
Hold(
    weights={"AAPL": 0.6, "MSFT": 0.4},
    rebalance=RebalanceSpec(horizon="monthly", tolerance=0.05),
)
```

Weights can be a dict, a `pd.Series`, or a callable that receives the
full `Bars` frame at `init` time and returns a dict — handy for weights
computed from a skfolio optimiser warm-up window. When omitted entirely,
`Hold` falls back to equal weights across every asset in the bars
frame.

## `DCA` — dollar-cost averaging

```python
from fundcloud.strategies import DCA

# Buy 1,000 USD across AAPL/MSFT every week.
DCA(amount=1_000, horizon="weekly", weights={"AAPL": 0.5, "MSFT": 0.5})

# Per-asset dollar amounts bypass the weights argument entirely.
DCA(amount={"AAPL": 500, "MSFT": 500}, horizon="monthly")

# Fire only inside a window; close everything on the way out.
DCA(
    amount=500,
    horizon="daily",
    weights={"BTC/USDT": 1.0},
    start="2024-01-01",
    end="2024-12-31",
    sell_on_end=True,
)
```

### Sizing by percentage of equity (`amount_pct`)

`amount_pct` is the equity-relative twin of `amount`: instead of fixing
the deposit in dollars, you commit a fraction of the **current**
portfolio equity at each fire. The simulator recomputes the dollar size
from `Portfolio.equity_curve` on every cadence trigger, so the deposit
scales naturally as the portfolio grows or shrinks.

```python
from fundcloud.strategies import DCA

# Deploy 1 % of equity every month, equal-split across whatever
# assets are in the bars frame.
DCA(amount_pct=0.01, horizon="monthly")

# Per-asset percentages — handy when contribution rules differ per leg.
DCA(amount_pct={"SPY": 0.012, "AGG": 0.008}, horizon="monthly")
```

`amount` and `amount_pct` are **mutually exclusive** — exactly one must
be supplied. On the very first fire (before any mark-to-market), DCA
falls back to starting cash so the first deposit is well-defined even
when `equity_curve` is still empty.

## Horizon semantics

| Horizon | Meaning |
|---|---|
| `"daily"` | Every trading day in the data source. |
| `"weekly"` | Every **7 calendar days** from the anchor. Matches the PRD's "(7 days)" wording — we explicitly **do not** use ISO weekday 1. |
| `"monthly"` | Same day-of-month as the anchor, snapped **backwards** to the most recent trading day ≤ that day in the same month. Falls back to the last trading day of the month when the anchor day comes before the first bar. |
| any pandas offset like `"30D"`, `"2W"` | Treated as a raw cadence step anchored at the start. |
| `Cadence(step=..., anchor=...)` | Explicit construction for odd cadences. |

## Custom strategies

```python
from fundcloud.strategies import BaseStrategy, Context, register_strategy
from fundcloud.sim import Order

@register_strategy("ma_crossover")
class MACrossover(BaseStrategy):
    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        self.fast, self.slow = fast, slow

    def decide(self, ctx: Context) -> list[Order]:
        if len(ctx.history) < self.slow:
            return []
        close = ctx.history.xs("close", axis=1, level=0)
        fast_ma = close.rolling(self.fast).mean().iloc[-1]
        slow_ma = close.rolling(self.slow).mean().iloc[-1]
        orders = []
        for asset in ctx.assets:
            if fast_ma[asset] > slow_ma[asset]:
                orders.append(Order(ts=ctx.ts, asset=asset, side="buy", qty=100))
            elif fast_ma[asset] < slow_ma[asset]:
                orders.append(Order(ts=ctx.ts, asset=asset, side="sell", qty=100))
        return orders
```

Strategies are **not** sklearn estimators because their semantic is
"behave over time", not "fit then transform" — but they're still plain
picklable Python classes, so they round-trip through joblib and the
Catalog serialisation cleanly.

## Multi-asset DCA

Invest across several assets each period by passing a `weights` dict. Weights are
normalised so they sum to 1; the `amount` is then split proportionally.

```python
from fundcloud.strategies import DCA
from fundcloud.sim import Simulator
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
idx = pd.bdate_range("2022-01-03", periods=504)

def _asset(price0, vol):
    c = price0 + np.cumsum(rng.normal(0, vol, len(idx)))
    return {"open": c, "high": c+0.5, "low": c-0.5, "close": c, "volume": 1e6}

bars = pd.concat(
    {"SPY": pd.DataFrame(_asset(400, 2.0), index=idx),
     "QQQ": pd.DataFrame(_asset(280, 2.5), index=idx),
     "BND": pd.DataFrame(_asset(75,  0.3), index=idx),
     "GLD": pd.DataFrame(_asset(180, 1.2), index=idx)},
    axis=1,
).pipe(lambda df: df.set_axis(df.columns.swaplevel(), axis=1)).sort_index(axis=1)

strategy = DCA(
    amount=2_000,          # $2,000 deployed each week
    horizon="weekly",
    weights={"SPY": 0.40, "QQQ": 0.30, "BND": 0.20, "GLD": 0.10},
)
result = Simulator(bars, cash=200_000).run_strategy(strategy)
print(result.portfolio.summary())
```

**Per-asset amounts** bypass the `weights` argument entirely — handy when you
have fixed per-account contribution rules:

```python
DCA(amount={"SPY": 800, "QQQ": 600, "BND": 400, "GLD": 200}, horizon="weekly")
```

## Hold with rebalancing

A buy-and-hold that rebalances quarterly when any asset drifts more than 5 % from
target harvests the rebalancing premium without churning every bar.

```python
from fundcloud.strategies import Hold, RebalanceSpec

# Quarterly rebalance, skip if all assets are within 5 % of target.
strategy = Hold(
    weights={"SPY": 0.60, "BND": 0.40},
    rebalance=RebalanceSpec(horizon="monthly", tolerance=0.05),
)
```

| `tolerance` | Effect |
|---|---|
| `0.0` | Rebalance every scheduled bar (calendar-only) |
| `0.05` | Skip rebalancing when all weights within 5 % of target |
| `0.10` | More permissive — fewer trades, more drift |

Rebalancing is most valuable in **sideways or mean-reverting markets**: it
systematically sells outperformers and buys underperformers, capturing the
volatility premium even when the long-run returns are similar.

## DCA vs Hold — comparing on the same capital

The most common question: "would I have been better off lump-summing on day 1?"
Use a common `cash` pool and compare the two `SimResult` portfolios:

```python
from fundcloud.sim import Simulator
from fundcloud.strategies import DCA, Hold

WEIGHTS = {"SPY": 0.60, "BND": 0.40}
TOTAL_CASH = 120_000

dca_result  = Simulator(bars, cash=TOTAL_CASH).run_strategy(
    DCA(amount=1_000, horizon="weekly", weights=WEIGHTS)
)
hold_result = Simulator(bars, cash=TOTAL_CASH).run_strategy(
    Hold(weights=WEIGHTS)
)

# Side-by-side comparison
import pandas as pd
comparison = pd.concat(
    {"DCA": dca_result.portfolio.summary(),
     "Hold": hold_result.portfolio.summary()},
    axis=1,
)
print(comparison)
```

Focus on **Calmar ratio** (CAGR / max drawdown) rather than raw Sharpe — DCA
typically wins on Calmar in volatile markets because it avoids the single worst
entry point, at the cost of lower total return when markets trend steadily upward.

```
Key metrics to compare:
  cagr             — DCA usually lower in bull markets (cash drag)
  max_drawdown     — DCA usually shallower (staggered entries)
  calmar           — DCA often competitive or better
  ann_volatility   — DCA usually lower
```

## Reading SimResult

After `Simulator.run_strategy()` you get a `SimResult` object with four attributes:

```python
result.portfolio    # Portfolio object — all metrics, tear sheets
result.equity_curve # pd.Series of cumulative equity (dollars, not returns)
result.trades       # pd.DataFrame — one row per fill (asset, side, qty, price, …)
result.orders       # pd.DataFrame — one row per emitted order (filled or not)
```

Useful queries:

```python
# How many buys did the DCA execute? (positive qty = buy, negative = sell)
buys = result.trades[result.trades["qty"] > 0]
print(f"Total purchases:  {len(buys)}")
print(f"Avg purchase qty: {buys['qty'].mean():.1f} shares")

# Total deployed capital (sum of all buy notional)
deployed = buys["notional"].sum()
print(f"Capital deployed: ${deployed:,.0f}")

# Equity curve peak
print(f"Peak equity: ${result.equity_curve.max():,.0f}")
print(f"Final equity: ${result.equity_curve.iloc[-1]:,.0f}")
```

To close all positions at the end of a DCA window, set `sell_on_end=True`:

```python
DCA(
    amount=500,
    horizon="weekly",
    weights={"SPY": 1.0},
    start="2024-01-01",
    end="2024-12-31",
    sell_on_end=True,   # liquidate on the last bar of the window
)
```
