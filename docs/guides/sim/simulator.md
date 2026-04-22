# Simulator

`Simulator` is the single object that turns an intention — whether expressed as a strategy, a target-weights matrix, a pair of entry/exit boolean panels, or a raw order blotter — into an executed `Portfolio`, and records every fill, order, and intra-bar cost along the way. Four entry points share one execution loop, one set of cost/slippage/timing assumptions, and one result shape, so swapping how you specify the strategy never changes how you read the output.

!!! note "Why four entry points, not four simulators"
    In practice, researchers move back and forth between specification styles: a target-weights matrix from an optimiser, then an explicit order list for a what-if, then a `BaseStrategy` for production. Keeping all four paths on the same simulator core means the execution semantics (fill timing, slippage, fees, cash accounting) are guaranteed to match, so comparing results across specifications is apples-to-apples.

```python
from fundcloud.sim import Simulator, FixedBps, HalfSpread, NextBarOpen

sim = Simulator(
    bars,
    cash=100_000,
    costs=FixedBps(bps=5),
    slippage=HalfSpread(spread_bps=2),
    execution=NextBarOpen(),
)
```

## Entry points

| Method | Use when |
|---|---|
| `run_strategy(strategy)` | You have a `BaseStrategy`. The simulator calls `init` once, `decide` per bar, `close` at the end. |
| `run_weights(weights_df)` | You have a wide `(date × asset)` target-weights frame. The simulator rebalances to the target row by row. |
| `run_signals(entries, exits, size=1.0)` | You have two boolean panels. On entry, allocate `size × cash` to the asset; on exit, close the position. |
| `run_orders(orders_df)` | You have an explicit long-format `[ts, asset, side, qty]` frame. Executes them as-is. |

Every entry point returns a `SimResult` with:

- `portfolio` — the post-sim snapshot (`Portfolio`).
- `trades` — long-format executed trades DataFrame.
- `orders` — long-format order history (including unfilled).
- `equity_curve` — per-bar equity in dollars.
- `.metrics()` / `.summary()` — shortcut to `portfolio.summary()`.

## Execution models

- **`NextBarOpen`** (default) — orders fire at the open of bar `t+1`.
  Avoids look-ahead.
- **`SameBarClose`** — orders fire at the close of bar `t`. Convenient but
  introduces a subtle bias.

## Cost + slippage

- Costs: `FixedBps(bps, minimum)`, `PerShare(rate, minimum)`, `NoCost`.
- Slippage: `NoSlippage`, `HalfSpread(spread_bps)`.

Both are simple protocols — supply your own class with the matching
`fee` / `apply` method to get custom behaviour.

## Worked example

```python
from fundcloud.sim import Simulator, FixedBps
from fundcloud.strategies import DCA

result = Simulator(bars, cash=100_000, costs=FixedBps(10)).run_strategy(
    DCA(amount=500, horizon="weekly", weights={"AAPL": 0.5, "MSFT": 0.5})
)

print(result.portfolio.sharpe())
print(result.trades[["ts", "asset", "qty", "price", "fee"]].head())
```

Feed the same `result.portfolio` into [Tear sheets](../reports/tearsheets.md)
for polished HTML/PDF/Excel output.
